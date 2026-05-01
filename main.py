from src.api.db.db_engine import get_db, init_db
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routers.ml_routes import router as api_router

    
# Initialize the database tables
init_db()


app = FastAPI(
    title="Kinetic Core API",
    description="API for the Kinetic Core ML Service",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the endpoints from the controllers
app.include_router(api_router, prefix="/api")

@app.get("/")
def root():
    return {"message": "Welcome to the Kinetic Core API. Go to /docs for the API documentation."}

if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)