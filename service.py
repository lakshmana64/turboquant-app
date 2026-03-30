"""
TurboQuant High-Performance API Service.

Exposes TurboQuant quantization and search as a FastAPI microservice.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import torch
import numpy as np
from turboquant.sdk.optimize import TurboQuantizer

app = FastAPI(title="TurboQuant API", description="Unbiased Vector Quantization Service")

# --- Models ---

class VectorRequest(BaseModel):
    vector: List[float]
    sq_bits: int = 4
    qjl_bits: int = 64
    pack_bits: bool = True

class BatchVectorRequest(BaseModel):
    vectors: List[List[float]]
    sq_bits: int = 4
    qjl_bits: int = 64

class SearchRequest(BaseModel):
    query: List[float]
    encoded_keys: Dict[str, Any]
    sq_bits: int = 4
    qjl_bits: int = 64

# --- API Endpoints ---

@app.get("/")
async def root():
    return {"status": "online", "engine": "TurboQuant", "version": "1.0.1"}

@app.post("/encode")
async def encode_vector(req: VectorRequest):
    """Encode a single vector and return the compressed representation."""
    try:
        vec = torch.tensor(req.vector)
        dim = vec.shape[-1]
        
        quantizer = TurboQuantizer(dim, qjl_bits=req.qjl_bits, sq_bits=req.sq_bits, pack_bits=req.pack_bits)
        encoded = quantizer.encode(vec.unsqueeze(0))
        
        # Convert tensors to lists for JSON serialization
        serializable_encoded = {}
        for k, v in encoded.items():
            if isinstance(v, torch.Tensor):
                serializable_encoded[k] = v.tolist()
            else:
                serializable_encoded[k] = v
                
        return {
            "encoded": serializable_encoded,
            "dim": dim,
            "compression_factor": quantizer.compression_factor
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search(req: SearchRequest):
    """Estimate inner product between a query and compressed keys."""
    try:
        q = torch.tensor(req.query)
        dim = q.shape[-1]
        
        # Restore encoded tensors
        encoded = {}
        for k, v in req.encoded_keys.items():
            if isinstance(v, list):
                encoded[k] = torch.tensor(v)
            else:
                encoded[k] = v
        
        quantizer = TurboQuantizer(dim, qjl_bits=req.qjl_bits, sq_bits=req.sq_bits)
        scores = quantizer.estimate_batch(q.unsqueeze(0), encoded)
        
        return {"scores": scores[0].tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
