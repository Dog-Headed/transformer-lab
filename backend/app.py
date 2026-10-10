"""Run: python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000"""
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from .config import GenerateRequest, LoadRequest, PredictRequest, ResumeRequest, TrainRequest
from .training import ServiceError, TrainingManager


def create_app(storage: Path | None = None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.manager = TrainingManager(storage or Path(os.getenv(
            "TRAINING_STORAGE_DIR", str(Path(__file__).parent / "runs"))))
        yield
        app.state.manager.close()

    application = FastAPI(title="Transformer Training Lab", version="1.0.0", lifespan=lifespan)
    origins = os.getenv("TRAINING_ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
    application.add_middleware(CORSMiddleware, allow_origins=origins.split(","),
                               allow_methods=["GET", "POST"], allow_headers=["Content-Type"])

    @application.exception_handler(ServiceError)
    async def service_error(request, exc):
        return JSONResponse(status_code=exc.status, content={"detail": exc.detail})

    @application.exception_handler(ValueError)
    async def invalid_data(request, exc):
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    def manager(request: Request):
        return request.app.state.manager

    @application.get("/health")
    def health():
        return {"status": "ok", "torch_threads": 2, "max_concurrent_training": 1,
                "max_runtime_seconds": 180, **TrainingManager.hardware()}

    @application.post("/tasks", status_code=201)
    def create_task(body: TrainRequest, request: Request):
        return manager(request).create(body)

    @application.get("/tasks")
    def list_tasks(request: Request):
        service = manager(request)
        with service.lock:
            return [service.snapshot(task_id) for task_id in service.jobs]

    @application.get("/tasks/{task_id}")
    def get_task(task_id: str, request: Request):
        return manager(request).snapshot(task_id)

    @application.post("/tasks/{task_id}/stop")
    def stop_task(task_id: str, request: Request):
        return manager(request).stop(task_id)

    @application.post("/tasks/{task_id}/resume")
    def resume_task(task_id: str, body: ResumeRequest, request: Request):
        return manager(request).resume(task_id, body.max_steps)

    @application.post("/tasks/{task_id}/checkpoint", status_code=201)
    def save_checkpoint(task_id: str, request: Request):
        return manager(request).save(task_id)

    @application.post("/tasks/{task_id}/generate")
    def generate(task_id: str, body: GenerateRequest, request: Request):
        return manager(request).generate(task_id, body)

    @application.post("/tasks/{task_id}/predict")
    def predict(task_id: str, body: PredictRequest, request: Request):
        return manager(request).predict(task_id, body)

    @application.post("/tasks/{task_id}/step")
    def step(task_id: str, body: PredictRequest, request: Request):
        return manager(request).step(task_id, body)

    @application.get("/checkpoints")
    def list_checkpoints(request: Request):
        return manager(request).checkpoints()

    @application.post("/checkpoints/{checkpoint_id}/load", status_code=201)
    def load_checkpoint(checkpoint_id: str, request: Request, body: LoadRequest = LoadRequest()):
        return manager(request).load(checkpoint_id, body.device)

    @application.get("/checkpoints/{checkpoint_id}/download")
    def download_checkpoint(checkpoint_id: str, request: Request):
        return FileResponse(manager(request).checkpoint_file(checkpoint_id),
                            filename=f"transformer-{checkpoint_id}.pt", media_type="application/octet-stream")

    return application


app = create_app()
