from fastapi import FastAPI, UploadFile, File
import shutil

from app.ml.utils import validate_image
from app.ml.model import predict_image

app = FastAPI()

@app.get("/")
def home():
    return {"message": "AI Architecture Recognition System"}


@app.post("/upload")
def upload_image(file: UploadFile = File(...)):

    allowed_types = ["image/jpeg", "image/png"]

    if file.content_type not in allowed_types:
        return {"error": "Invalid file types"}

    file_path = f"uploads/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    is_valid = validate_image(file_path)

    if not is_valid:
        return {"error": "Corrupted image"}
    
    prediction = predict_image(file_path)

    return {
        "filename": file.filename,
        "prediction": prediction
    }