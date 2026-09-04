from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import resident  
from .routers import vehicle
 
app = FastAPI(
    title="ANPR Autogate API",
    description="REST API untuk sistem Residence Autogate ANPR (mulai dari CRUD Vehicle).",
    version="0.1.0",
)
 
# Izinkan Streamlit (default port 8501) memanggil API ini dari browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
app.include_router(vehicle.router)
app.include_router(resident.router)
 
 
@app.get("/", tags=["Health"])
def health_check():
    return {"status": "ok", "service": "anpr-api"}