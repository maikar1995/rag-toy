# src/main.py
from fastapi import FastAPI
from .api.routers import rag

app = FastAPI()
app.include_router(rag.router, prefix="/api/v1")