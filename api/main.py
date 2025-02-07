from fastapi import FastAPI

# ✅ Create a FastAPI instance
app = FastAPI()

# ✅ Define a basic test route to check if the API is running
@app.get("/")
async def root():
    return {"message": "Dental Crown API is running"}

# ✅ Add a test endpoint
@app.get("/api/test")
async def test():
    return {"message": "This is a test endpoint"}
