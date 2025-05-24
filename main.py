import argparse
import os
import sys
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, Header
import shutil
from hair_swap import HairFast, get_parser

app = FastAPI()

API_KEY = os.getenv("API_KEY")  # קריאה למפתח מהמשתנים הסביבתיים

def verify_api_key(api_key: str = Header(...)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")  # חסימה במקרה של מפתח שגוי

@app.post("/swap", dependencies=[Depends(verify_api_key)])
async def swap_images(face: UploadFile = File(...), shape: UploadFile = File(...), color: UploadFile = File(...)):
    model_args = get_parser().parse_args([])

    # שמירת הקבצים הזמניים
    temp_dir = Path("temp_images")
    temp_dir.mkdir(exist_ok=True)
    face_path = temp_dir / face.filename
    shape_path = temp_dir / shape.filename
    color_path = temp_dir / color.filename

    for file, path in [(face, face_path), (shape, shape_path), (color, color_path)]:
        with open(path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

    hair_fast = HairFast(model_args)
    output_image = hair_fast.swap(face_path, shape_path, color_path)

    output_path = temp_dir / "result.png"
    save_image(output_image, output_path)

    return {"message": "Success", "output_image": str(output_path)}
