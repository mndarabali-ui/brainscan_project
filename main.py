import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from src.config import OUTPUT_DIR
from src.api import router as api_router

app = FastAPI(
    title="BrainScan AI Framework API",
    description="API untuk analisis otomatis CT-Scan & MRI",
    version="1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(os.path.join(OUTPUT_DIR, "figures"), exist_ok=True)
os.makedirs("temp_uploads", exist_ok=True)

app.include_router(api_router)

@app.get("/health")
def health():
    return {"status": "ok"}

app.mount(
    "/outputs/figures",
    StaticFiles(directory=os.path.join(OUTPUT_DIR, "figures")),
    name="figures",
)

app.mount(
    "/",
    StaticFiles(directory="src/static", html=True),
    name="static",
)
