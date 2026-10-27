from fastapi import FastAPI
from dataclasses import asdict
from fastapi import FastAPI, HTTPException
from model import ARCHITECTURES

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
