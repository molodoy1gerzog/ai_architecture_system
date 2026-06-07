import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.ml.utils import validate_image
from app.ml.model import predict_image


app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

UPLOAD_DIR = PROJECT_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static"
)

app.mount(
    "/uploads",
    StaticFiles(directory=str(UPLOAD_DIR)),
    name="uploads"
)

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


def render_index(request: Request, **context):
    context["request"] = request

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context=context
    )


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return render_index(request)


@app.post("/predict", response_class=HTMLResponse)
def predict(request: Request, file: UploadFile = File(...)):
    allowed_types = ["image/jpeg", "image/png", "image/webp"]

    if file.content_type not in allowed_types:
        return render_index(
            request,
            error="Недопустимый формат файла. Загрузите JPG, PNG или WEBP."
        )

    file_extension = Path(file.filename).suffix.lower()
    safe_filename = f"{uuid4().hex}{file_extension}"
    file_path = UPLOAD_DIR / safe_filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    is_valid = validate_image(file_path)

    if not is_valid:
        return render_index(
            request,
            error="Файл поврежден или не является изображением."
        )

    prediction = predict_image(str(file_path))

    return render_index(
        request,
        filename=file.filename,
        image_url=f"/uploads/{safe_filename}",
        prediction=prediction
    )


@app.post("/upload")
def upload_image(file: UploadFile = File(...)):
    allowed_types = ["image/jpeg", "image/png", "image/webp"]

    if file.content_type not in allowed_types:
        return {"error": "Invalid file type"}

    file_path = UPLOAD_DIR / file.filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    is_valid = validate_image(file_path)

    if not is_valid:
        return {"error": "Corrupted image"}

    prediction = predict_image(str(file_path))

    return {
        "filename": file.filename,
        "prediction": prediction
    }