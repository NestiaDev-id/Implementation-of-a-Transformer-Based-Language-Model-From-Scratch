import asyncio
import random
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Inisialisasi aplikasi FastAPI
app = FastAPI(
    title="LLM From Scratch API",
    description="API untuk menyimulasikan proses inferensi dan memberikan insight untuk visualizer.",
    version="1.0.0",
)
origins = [
    "http://localhost:5173",  # Port default Vite/React
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"], 
)

@app.get("/")
def read_root():
    return {"status": "online", "message": "Welcome to the LLM From Scratch API!"}

@app.get("/healthcheck")
def healthcheck():
    return {"status": "online"}