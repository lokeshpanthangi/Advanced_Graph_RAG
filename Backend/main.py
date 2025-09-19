from fastapi import FastAPI
from routes.neo import neo


app = FastAPI()
app.include_router(neo)



@app.get("/health")
async def health_check():
    return {"status": "healthy"}

