import logging
import os
import random
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Annotated

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request

from model import ARCHITECTURES
from tasks import MAX_LENGTH, TASKS, make_example, to_sequence

load_dotenv()

APP_VERSION = "0.1.0"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
DEFAULT_SAMPLE_LENGTH = int(os.getenv("DEFAULT_SAMPLE_LENGTH", "8"))

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("rope_or_nope")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting rope-or-nope %s (log level %s)", APP_VERSION, LOG_LEVEL)
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="rope-or-nope",
    description="Tiny transformers on cpu",
    version=APP_VERSION,
    lifespan=lifespan,
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info("%s %s -> %d (%.1f ms)", request.method, request.url.path, response.status_code, elapsed_ms)
    return response

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/version")
def version():
    return {"version": APP_VERSION}

@app.get("/")
def root():
    return {"name": "rope-or-nope", "docs": "/docs", "health": "/health"}

@app.get("/architectures")
def list_architectures():
    return [{"name": name, "description": cfg.description} for name, cfg in ARCHITECTURES.items()]


@app.get("/architectures/{name}")
def get_architecture(name: str):
    if name not in ARCHITECTURES:
        logger.warning("Unknown architecture requested: %s", name)
        raise HTTPException(
            status_code=404,
            detail=f"Unknown architecture '{name}'. Available: {', '.join(ARCHITECTURES)}",
        )
    return {"name": name, **asdict(ARCHITECTURES[name])}


@app.get("/tasks/{name}/sample")
def sample_task(
    name: str,
    length: Annotated[int, Query(ge=1, le=MAX_LENGTH)] = DEFAULT_SAMPLE_LENGTH,
    seed: int | None = None,
):
    if name not in TASKS:
        logger.warning("Unknown task requested: %s", name)
        raise HTTPException(
            status_code=404,
            detail=f"Unknown task '{name}'. Available: {', '.join(TASKS)}",
        )
    xs, ys = make_example(name, length, random.Random(seed))
    logger.info("Sampled task=%s length=%d seed=%s", name, length, seed)
    return {"task": name, "input": xs, "target": ys, "sequence": " ".join(to_sequence(xs, ys))}
