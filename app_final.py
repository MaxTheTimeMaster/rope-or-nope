from fastapi import FastAPI

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
