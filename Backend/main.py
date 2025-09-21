from fastapi import FastAPI
from routes.vector_query import neo
from crud.orchestrator import test


app = FastAPI()
app.include_router(neo)
app.include_router(test)



@app.get("/health")
async def health_check():
    return {"status": "healthy"}

