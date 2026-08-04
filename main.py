from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI()

# Serve semua file di folder static (css/js/gambar)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Buka UI dari root "/"
@app.get("/")
def home():
    return FileResponse("static/index.html")

# Contoh endpoint API
@app.get("/api/health")
def health():
    return {"status": "ok"}
