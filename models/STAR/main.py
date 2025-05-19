from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from pathlib import Path
from hair_swap import HairFast, get_parser
from torchvision.utils import save_image
import shutil

app = FastAPI()

# אתחול המודל פעם אחת בלבד בעת עליית השרת
model_args = get_parser().parse_args([])
hair_fast = HairFast(model_args)

# תיקיות עבודה קבועות
INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output")
INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

@app.post("/swap")
async def swap_hair(
    face: UploadFile = File(...),
    shape: UploadFile = File(...),
    color: UploadFile = File(...)
):
    # שמירת קבצים זמניים
    face_path = INPUT_DIR / "face.png"
    shape_path = INPUT_DIR / "shape.png"
    color_path = INPUT_DIR / "color.png"

    with open(face_path, "wb") as f:
        shutil.copyfileobj(face.file, f)
    with open(shape_path, "wb") as f:
        shutil.copyfileobj(shape.file, f)
    with open(color_path, "wb") as f:
        shutil.copyfileobj(color.file, f)

    # הרצת המודל
    result = hair_fast.swap(face_path, shape_path, color_path)

    # שמירת התוצאה
    output_path = OUTPUT_DIR / "result.png"
    save_image(result, output_path)

    return FileResponse(output_path, media_type="image/png")

