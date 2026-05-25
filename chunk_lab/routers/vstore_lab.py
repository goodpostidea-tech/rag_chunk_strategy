"""Vector store benchmark routes — mounted at /vstore on chunk_api."""

from fastapi import APIRouter, HTTPException

from chunk_lab.config import get_settings
from chunk_lab.models import embedding_info
from chunk_lab.vectorstore import vectorstore_info
from vstore_lab.benchmark import list_vstore_backends, run_vstore_benchmark, vstore_response_to_dict
from vstore_lab.types import VStoreBenchRequest

router = APIRouter(prefix="/vstore", tags=["vstore"])


@router.get("/health")
def vstore_health():
    cfg = get_settings()
    return {
        "status": "ok",
        "service": "vstore_lab",
        "vector_store": vectorstore_info(cfg),
        "embedding": embedding_info(cfg),
    }


@router.get("/backends")
def backends_list():
    return {"backends": list_vstore_backends()}


@router.post("/benchmark")
def vstore_benchmark(req: VStoreBenchRequest):
    try:
        resp = run_vstore_benchmark(
            vector_counts=req.vector_counts,
            providers=req.providers,
        )
        return vstore_response_to_dict(resp)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
