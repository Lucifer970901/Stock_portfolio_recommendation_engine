from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api.routes import router
from app.services.recommender import recommender
from app.core.logger import get_logger
from fastapi.staticfiles import StaticFiles

log = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting up — building recommender...")
    recommender.build()
    yield
    log.info("Shutting down")

app = FastAPI(
    title='Stock Recommender API',
    version='1.0.0',
    lifespan=lifespan
)

app.include_router(router)
app.mount("/static", StaticFiles(directory="static"), name="static")