from contextlib import asynccontextmanager
from motor.motor_asyncio import AsyncIOMotorClient
from fastapi import FastAPI
from backend.app.core.config import MONGO_URI, DB_NAME

client = None
db = None
collection = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global client, db, collection
    client = AsyncIOMotorClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db["predictions"]
    print(f"MongoDB connected: {DB_NAME}")
    yield
    client.close()
    print("MongoDB disconnected")


def get_collection():
    return collection
