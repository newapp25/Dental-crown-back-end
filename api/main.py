import sys
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api.ai_depth_estimation import predict_depth
from api.photogrammetry import reconstruct_3d
from api.utils import validate_image, preprocess_image, handle_error

# Ensure correct module imports on Vercel
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup event for debugging
@app.on_event("startup")
async def startup_event():
    print("🚀 FastAPI app has started!")

@app.get("/")
async def root():
    return {"message": "Dental Crown Design API is running"}

@app.post("/api/depth")
async def depth_estimation(file: UploadFile = File(None)):  # Fixed File(...)
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    try:
        content = await file.read()
        image = validate_image(content)
        preprocessed_image = preprocess_image(image)
        depth_map = predict_depth(preprocessed_image)
        return {"depth_map": depth_map}
    except Exception as e:
        return handle_error(str(e))

@app.post("/api/reconstruct")
async def reconstruct(files: list[UploadFile] = File(None)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    try:
        images = [validate_image(await file.read()) for file in files]
        point_cloud = reconstruct_3d(images)
        return {"point_cloud": point_cloud}
    except Exception as e:
        return handle_error(str(e))
