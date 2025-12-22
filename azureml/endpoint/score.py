# Optional minimal AML-style scoring file.
# If you expose vLLM directly, you may not need this.

from fastapi import FastAPI

app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}
