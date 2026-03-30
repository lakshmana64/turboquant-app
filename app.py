"""
TurboQuant Dashboard - Gradio Frontend

A comprehensive visual interface to benchmark TurboQuant's performance 
on real LLM embeddings and visualize compression vs. accuracy trade-offs.
"""

import gradio as gr
import torch
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from turboquant.sdk.optimize import TurboQuantizer
from core.adaptive import adaptive_quantize

# --- Helper Functions ---

def get_ollama_models():
    """Fetch list of all models from Ollama."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return [m["name"] for m in response.json()["models"]]
    except Exception:
        return ["No Ollama instance found"]

def run_quantization_demo(model_name, sq_bits, qjl_bits, prompt, adaptive_mode):
    """Run a live quantization test for a single prompt."""
    if model_name == "No Ollama instance found":
        return "Please start Ollama to run this demo.", None, "<div style='text-align: center;'>Waiting for Ollama</div>"
    
    # 1. Fetch Embeddings
    url = "http://localhost:11434/api/embeddings"
    try:
        res = requests.post(url, json={"model": model_name, "prompt": prompt}, timeout=10)
        vec = torch.tensor(res.json()["embedding"])
    except Exception as e:
        return f"Error fetching from Ollama: {e}", None, "<div style='text-align: center; color: #b91c1c;'>Embedding request failed</div>"

    d = vec.shape[-1]
    
    # 2. Setup Data (Self-Attention simulation)
    x = vec.unsqueeze(0) # (1, D)
    q = vec.unsqueeze(0) # (1, D)
    
    # 3. Quantize
    quantizer = TurboQuantizer(d, qjl_bits=int(qjl_bits), sq_bits=int(sq_bits))
    
    if adaptive_mode:
        # High bits = 8, Low bits = sq_bits
        x_hat, importance, _ = adaptive_quantize(x, low_bits=int(sq_bits), high_bits=8, rotation_matrix=quantizer.codec.rotation_matrix)
        # Manually compute Stage 2 for adaptive
        residual = x - x_hat
        _, r_signs, r_norm = quantizer.codec.estimator.encode_key(x, x_hat)
        est_dot = quantizer.codec.estimator.estimate(q.squeeze(0), x_hat.squeeze(0), r_signs[0], r_norm[0]).item()
        
        importance_pct = importance.float().mean().item() * 100
        stats_prefix = f"### ✨ Adaptive Mode Active\n- **Important Dims (8-bit)**: {importance_pct:.1f}%\n"
    else:
        encoded = quantizer.encode(x)
        est_dot = quantizer.estimate(q.squeeze(0), encoded).item()
        stats_prefix = ""
    
    # 4. Estimation Accuracy
    true_dot = (q * x).sum().item()
    
    # 5. Build Visualization
    # Feature distribution chart
    df_vec = pd.DataFrame({"Feature Index": range(d), "Value": vec.numpy()})
    fig_dist = px.histogram(df_vec, x="Value", title="Embedding Coordinate Distribution (Stage 1 Input)", 
                            color_discrete_sequence=['#636EFA'])
    fig_dist.update_layout(template="plotly_white")

    # Metrics
    comp_ratio = quantizer.compression_factor
    mse = (true_dot - est_dot)**2
    delta = abs(true_dot - est_dot)
    
    stats_md = f"""
    {stats_prefix}
    ### 📊 Benchmark Results
    - **Original Dimension**: {d}
    - **Avg Bits Per Dim**: {sq_bits + (qjl_bits/d):.2f}
    - **Compression Factor**: **{comp_ratio:.1f}x**
    - **True Inner Product**: {true_dot:.4f}
    - **TurboQuant Estimate**: {est_dot:.4f}
    - **MSE**: {mse:.6e}
    """

    meter_html = f"""
    <div style='text-align:center; padding: 12px; border: 1px solid #dbe4f0; border-radius: 12px; background: #f8fbff;'>
      <div style='font-size: 16px; font-weight: 600;'>Inner Product Accuracy Check</div>
      <div style='font-size: 13px; margin-top: 6px;'>Absolute error: <strong>{delta:.6f}</strong></div>
      <div style='font-size: 13px; margin-top: 4px;'>True: <strong>{true_dot:.4f}</strong> | Estimated: <strong>{est_dot:.4f}</strong></div>
    </div>
    """

    return stats_md, fig_dist, meter_html

def calculate_savings(memory_gb, context_len, bits):
    """Calculate VRAM savings for a given context window."""
    # Simplified KV Cache calculation
    # FP16 (16 bits) vs TurboQuant (bits)
    savings = memory_gb * (1 - (bits / 16))
    new_memory = memory_gb - savings
    
    fig = go.Figure(data=[
        go.Bar(name='Original (FP16)', x=['VRAM Usage'], y=[memory_gb], marker_color='#EF553B'),
        go.Bar(name='TurboQuant', x=['VRAM Usage'], y=[new_memory], marker_color='#00CC96')
    ])
    fig.update_layout(title=f"VRAM Savings for {context_len}k Context", yaxis_title="GB", barmode='group')
    
    return fig, f"By switching to {bits:.1f} bits, you save **{savings:.2f} GB** of VRAM."

# --- Gradio UI Layout ---

theme = gr.themes.Soft(primary_hue="blue", secondary_hue="slate")

with gr.Blocks() as demo:
    gr.Markdown("# 🔗 TurboQuant Dashboard")
    gr.Markdown("### Unbiased Online Vector Quantization with Near-optimal Distortion")
    
    with gr.Tabs():
        # Tab 1: Live Model Benchmarking
        with gr.TabItem("🚀 Live Ollama Benchmarker"):
            with gr.Row():
                with gr.Column(scale=1):
                    model_dd = gr.Dropdown(choices=get_ollama_models(), label="Select Ollama Model", value=get_ollama_models()[0])
                    sq_slider = gr.Slider(minimum=1, maximum=8, step=1, value=2, label="Stage 1 Bits (Scalar)")
                    qjl_slider = gr.Slider(minimum=0, maximum=512, step=32, value=64, label="Stage 2 Bits (QJL Correction)")
                    adaptive_mode = gr.Checkbox(label="Enable Adaptive Bit-Rate (ABR)", value=False)
                    text_input = gr.Textbox(placeholder="Enter prompt to embed...", label="Test Prompt", value="Quantum computing is transforming the future of cryptography.")
                    run_btn = gr.Button("Quantize & Benchmark", variant="primary")
                
                with gr.Column(scale=2):
                    stats_output = gr.Markdown("Click the button to run the benchmark.")
                    dist_chart = gr.Plot(label="Coordinate Distribution")
            
            with gr.Row():
                ip_meter = gr.HTML("<div style='text-align: center; font-size: 20px;'>Inner Product Accuracy Check</div>")

        # Tab 2: VRAM Savings Calculator
        with gr.TabItem("💾 KV Cache Calculator"):
            gr.Markdown("Estimate how much VRAM you save when scaling context windows with TurboQuant.")
            with gr.Row():
                with gr.Column():
                    base_vram = gr.Number(value=24, label="Current KV Cache VRAM Usage (GB)")
                    ctx_len = gr.Number(value=128, label="Target Context Length (k tokens)")
                    bits_radio = gr.Radio(choices=[2.0, 3.0, 4.0], value=2.0, label="TurboQuant Bits Per Dim")
                    calc_btn = gr.Button("Calculate Savings")
                
                with gr.Column():
                    calc_chart = gr.Plot()
                    calc_text = gr.Markdown("")

        # Tab 3: Documentation & Theory
        with gr.TabItem("📖 How it Works"):
            gr.Markdown("""
            ### The Two-Stage Architecture
            
            1. **Stage 1 (Random Rotation + SQ)**:
               We apply a random orthogonal rotation to the vector. This spreads out "spike" features, making the data follow a concentrated Gaussian-like distribution. We then apply optimal Lloyd-Max scalar quantization.
            
            2. **Stage 2 (QJL Residual Correction)**:
               Standard quantization is biased. TurboQuant computes the residual (error) and projects it using a 1-bit Quantized Johnson-Lindenstrauss (QJL) transform. This bit of extra info makes the final inner product estimate **unbiased**.
            
            ### Advanced Optimizations
            - **Adaptive Bit-Rate (ABR)**: Automatically uses higher precision (8-bit) for high-variance dimensions and lower precision (2-bit) for others.
            - **Triton Kernels**: Fused GPU kernels for single-pass quantization and packing.
            
            ### Why This Matters for LLMs
            KV caches grow linearly with context length. TurboQuant allows you to fit **8x-12x more tokens** into the same VRAM while maintaining the attention precision needed for complex reasoning.
            """)

    # Event handlers
    run_btn.click(
        run_quantization_demo, 
        inputs=[model_dd, sq_slider, qjl_slider, text_input, adaptive_mode], 
        outputs=[stats_output, dist_chart, ip_meter]
    )
    
    calc_btn.click(
        calculate_savings,
        inputs=[base_vram, ctx_len, bits_radio],
        outputs=[calc_chart, calc_text]
    )

if __name__ == "__main__":
    demo.launch(theme=theme)
