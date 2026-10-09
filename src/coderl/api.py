import os
from fastapi import FastAPI, HTTPException
from .schema import CompileRequest, RunRequest, CompareRequest
from .worker import Worker


def create_app(worker=None):
    worker = worker or Worker(os.getenv("CODERL_WORKER_DIR", ".worker"))
    app = FastAPI(title="CodeRL Compiler and Execution Tools", version="2.0.0")

    @app.get("/health")
    def health():
        return {"ok": True, **worker.environment()}

    @app.post("/compile")
    def compile_code(request: CompileRequest):
        return worker.compile(request.source)

    @app.post("/run")
    def run_code(request: RunRequest):
        try:
            return worker.run(request.artifact_id, request.stdin, request.timeout)
        except FileNotFoundError:
            raise HTTPException(404, "artifact not found")

    @app.post("/compare")
    def compare(request: CompareRequest):
        try:
            return worker.compare(**request.model_dump(exclude={"tests"}), tests=request.tests)
        except FileNotFoundError:
            raise HTTPException(404, "artifact not found")

    return app
