import importlib
import math
import sys
import types

import torch
from torch import nn


class DummyAttention(nn.Module):
    def __init__(self, hidden_size: int = 8, num_heads: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def _reshape(self, states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = states.shape
        return states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        query = self._reshape(self.q_proj(hidden_states))
        key = self._reshape(self.k_proj(hidden_states))
        value = self._reshape(self.v_proj(hidden_states))

        if past_key_value is not None:
            past_key, past_value = past_key_value
            key = torch.cat([past_key, key], dim=2)
            value = torch.cat([past_value, value], dim=2)

        scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            scores = scores + attention_mask
        attn = torch.softmax(scores, dim=-1)
        output = attn @ value
        output = output.transpose(1, 2).contiguous().view(hidden_states.shape[0], hidden_states.shape[1], self.hidden_size)
        output = self.o_proj(output)

        present = (key, value) if use_cache else None
        weights = attn if output_attentions else None
        return output, weights, present


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = DummyAttention()


def test_huggingface_wrapper_replaces_attention_and_roundtrips_cache():
    from integrations.huggingface import (
        CompressedPastKeyValue,
        TurboQuantAttentionWrapper,
        apply_turboquant_to_hf_model,
    )

    model = apply_turboquant_to_hf_model(
        DummyModel(),
        sq_bits=4,
        qjl_dim=16,
        return_compressed_cache=True,
    )

    assert isinstance(model.self_attn, TurboQuantAttentionWrapper)
    assert model._turboquant_wrapped_modules == ["self_attn"]

    hidden_states = torch.randn(2, 3, 8)
    output, _, cache = model.self_attn(hidden_states, use_cache=True, output_attentions=True)

    assert output.shape == hidden_states.shape
    assert isinstance(cache, CompressedPastKeyValue)
    assert cache.seq_len == 3

    next_hidden_states = torch.randn(2, 1, 8)
    output2, _, cache2 = model.self_attn(
        next_hidden_states,
        past_key_value=cache,
        use_cache=True,
    )
    assert output2.shape == next_hidden_states.shape
    assert isinstance(cache2, CompressedPastKeyValue)
    assert cache2.seq_len == 4


def _install_fake_haystack():
    haystack_module = types.ModuleType("haystack")
    components_module = types.ModuleType("haystack.components")
    embedders_module = types.ModuleType("haystack.components.embedders")

    class FakeDocument:
        def __init__(self, content, embedding=None, meta=None):
            self.content = content
            self.embedding = embedding
            self.meta = meta or {}
            self.score = None

    class FakeSentenceTransformersDocumentEmbedder:
        def __init__(self, model="fake", **kwargs):
            self.model = model

        def run(self, documents):
            lookup = {
                "alpha": [1.0, 0.0, 0.0, 0.0],
                "beta": [0.0, 1.0, 0.0, 0.0],
            }
            for document in documents:
                document.embedding = lookup[document.content]
            return {"documents": documents}

    haystack_module.Document = FakeDocument
    embedders_module.SentenceTransformersDocumentEmbedder = FakeSentenceTransformersDocumentEmbedder
    components_module.embedders = embedders_module

    sys.modules["haystack"] = haystack_module
    sys.modules["haystack.components"] = components_module
    sys.modules["haystack.components.embedders"] = embedders_module

    return FakeDocument


def test_haystack_embedder_and_store_are_wired(monkeypatch):
    fake_document = _install_fake_haystack()

    for module_name in [
        "integrations.plugins.haystack_plugin",
        "turboquant.integrations.plugins.haystack_plugin",
    ]:
        sys.modules.pop(module_name, None)

    module = importlib.import_module("integrations.plugins.haystack_plugin")
    module = importlib.reload(module)

    documents = [fake_document("alpha"), fake_document("beta")]
    embedder = module.TurboQuantDocumentEmbedder(
        num_bits=4,
        qjl_dim=8,
        device="cpu",
    )
    result = embedder.run(documents)

    assert "turboquant_encoded" in result
    assert embedder.last_encoded is not None
    for document in result["documents"]:
        assert hasattr(document, "_turboquant_encoded")
        assert "turboquant" in document.meta

    store = module.TurboQuantDocumentStore(num_bits=4, qjl_dim=8, device="cpu")
    assert store.write_documents(result["documents"]) == 2

    matches = store.query_documents([1.0, 0.0, 0.0, 0.0], top_k=1)
    assert len(matches) == 1
    assert matches[0].content == "alpha"


def test_vllm_adapter_and_patch_helper_are_wired():
    from integrations.plugins.vllm_plugin import (
        TurboQuantVLLMAdapter,
        patch_vllm_with_turboquant,
    )

    adapter = TurboQuantVLLMAdapter(num_bits=4, qjl_dim=16, device="cpu")
    key_cache = torch.randn(2, 3, 4, 5)
    value_cache = torch.randn(2, 3, 4, 5)

    encoded = adapter.compress_kv_cache(key_cache, value_cache)
    assert len(encoded["encoded_heads"]) == 3

    query = torch.randn(2, 3, 4)
    output = adapter.compute_paged_attention(
        query=query,
        encoded_keys=encoded,
        scale=0.5,
    )
    assert output.shape == query.shape

    class DummyEngine:
        pass

    engine = patch_vllm_with_turboquant(
        DummyEngine(),
        num_bits=4,
        qjl_dim=16,
        device="cpu",
    )

    assert hasattr(engine, "turboquant_adapter")
    assert hasattr(engine, "compress_kv_cache")
    assert hasattr(engine, "compute_paged_attention")
