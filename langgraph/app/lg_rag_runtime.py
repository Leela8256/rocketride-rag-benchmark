"""Entry point of the service: the standard FastAPI app with the two RAG pipelines (ingestion and search) added to
the registry, and a read-only report of what the process runs with (/__bench/runtime). Nothing here adds a lock,
a limit or a profiler.
"""
import asyncio
import importlib.metadata
import os
import sys
from pathlib import Path

from service.app import create_app, default_builders


def _rag():
    from pipelines.document_pdf_rag.adapter import build_document_pdf_rag_pipeline
    return build_document_pdf_rag_pipeline()


def _search():
    from pipelines.search_pdf_rag.adapter import build_search_pdf_rag_pipeline
    return build_search_pdf_rag_pipeline()


def builders():
    out = dict(default_builders())
    out['document-pdf-rag-v1'] = _rag
    out['search-pdf-rag-v1'] = _search
    return out


app = create_app(builders=builders())

PACKAGES = ['torch', 'torchvision', 'sentence-transformers', 'transformers', 'numpy', 'tokenizers',
            'langgraph', 'langchain-core', 'langchain-text-splitters', 'fastapi', 'uvicorn',
            'langchain-postgres', 'sqlalchemy', 'psycopg', 'psycopg-binary', 'psycopg-pool', 'asyncpg', 'pgvector']
MATH_KEYS = ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
             'TORCH_THREADS', 'TORCH_INTEROP_THREADS', 'TOKENIZERS_PARALLELISM']


@app.get('/__bench/runtime')
async def runtime():
    import torch
    from workload.document.embed import load_model
    from workload.document.store_pg import store_info
    pool = getattr(asyncio.get_running_loop(), '_default_executor', None)
    versions = {}
    for p in PACKAGES:
        try:
            versions[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            versions[p] = None
    profiler = sys.modules.get('yappi')
    return {'pipelines': list(app.state.settings.pipelines),
            'torch_intraop': torch.get_num_threads(), 'torch_interop': torch.get_num_interop_threads(),
            'cpu_count': os.cpu_count(), 'affinity': sorted(os.sched_getaffinity(0)),
            'executor_max_workers': getattr(pool, '_max_workers', None), 'versions': versions,
            'math_environment': {k: os.environ[k] for k in MATH_KEYS if k in os.environ},
            'model_device': str(load_model().device), 'model_max_seq_length': load_model().max_seq_length,
            'model_snapshots': [str(p) for p in Path(os.environ['HF_HOME']).glob('hub/models--*/snapshots/*')],
            'yappi_loaded': profiler is not None, 'yappi_running': profiler.is_running() if profiler else False,
            'sys_profile_active': sys.getprofile() is not None,
            'store': store_info(),
            'inference_policy': 'standard shared SentenceTransformer.encode; no added inference lock',
            'store_policy': 'langchain-postgres PGVectorStore at library defaults; no added pool, batch or lock',
            'parallel_info': torch.__config__.parallel_info()}
