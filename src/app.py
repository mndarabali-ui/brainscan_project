"""
app.py
------
Entry point aplikasi BrainScan AI.
Menjalankan FastAPI backend + UI static dalam satu service.
"""

import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from src.config import OUTPUT_DIR
from src.api import router as api_router

app = FastAPI(
    title="BrainScan AI Framework API",
    description="API untuk analisis otomatis CT-Scan & MRI menggunakan arsitektur Hybrid CNN-Transformer",
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

# WAJIB: API routes harus didaftarkan SEBELUM static UI
app.include_router(api_router)

# Route tambahan untuk cek server
@app.get("/health")
def health():
    return {"status": "ok"}

# Static heatmap/output
app.mount(
    "/outputs/figures",
    StaticFiles(directory=os.path.join(OUTPUT_DIR, "figures")),
    name="figures",
)

# WAJIB PALING BAWAH: UI frontend
app.mount(
    "/",
    StaticFiles(directory="src/static", html=True),
    name="static",
)
