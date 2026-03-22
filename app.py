from fastapi import FastAPI
from src.routes.predict_routes import router as predict_router

app = FastAPI(
    title="Predictive Maintenance API",
    version="1.0"
)

app.include_router(predict_router, prefix="/api")

@app.get("/")
def home():
    return {"message": "API is running"}

