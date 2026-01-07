# src/main.py
from fastapi import FastAPI
from .api.routers import users, items

app = FastAPI()
app.include_router(users.router, prefix="/api/v1")
app.include_router(items.router)
