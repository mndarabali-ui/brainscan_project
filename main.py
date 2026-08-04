from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI()

# serve folder src/static sebagai /static
app.mount("/static", StaticFiles(directory="static"), name="static")

# root "/" tampilkan UI
@app.get("/")
def home():
    return FileResponse("static/index.html")
