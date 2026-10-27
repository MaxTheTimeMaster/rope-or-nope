from dataclasses import asdict
from model import ARCHITECTURES
import random
from typing import Annotated
from fastapi import FastAPI, HTTPException, Query
from tasks import MAX_LENGTH, TASKS, make_example, to_sequence


APP_VERSION = "0.1.0"

app = FastAPI(
    title="rope-or-nope",
    description="Tiny transformers on cpu",
    version=APP_VERSION,
)

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
        raise HTTPException(
            status_code=404,
            detail=f"Unknown architecture '{name}'. Available: {', '.join(ARCHITECTURES)}",
        )
    return {"name": name, **asdict(ARCHITECTURES[name])}


@app.get("/tasks/{name}/sample")
def sample_task(
    name: str,
    length: Annotated[int, Query(ge=1, le=MAX_LENGTH)] = 8,
    seed: int | None = None,
):
    if name not in TASKS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown task '{name}'. Available: {', '.join(TASKS)}",
        )
    xs, ys = make_example(name, length, random.Random(seed))
    return {"task": name, "input": xs, "target": ys, "sequence": " ".join(to_sequence(xs, ys))}
