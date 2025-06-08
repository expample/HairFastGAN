import os
import base64
import io
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
from PIL import Image
from hair_swap import HairFast, get_parser

app = FastAPI()
API_KEY = os.getenv("MY_SECRET_API_KEY")

def verify_api_key(api_key: str = Header(...)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")

class ImageInput(BaseModel):
    faceImageBase64: str
    shapeImageBase64: str
    colorImageBase64: str

@app.post("/swap", dependencies=[Depends(verify_api_key)])
async def swap_images(data: ImageInput):
    model_args = get_parser().parse_args([])
    hair_fast = HairFast(model_args)

    # Decode base64 images
    def decode_base64_image(base64_string):
        image_data = base64.b64decode(base64_string.split(",")[-1])
        return Image.open(io.BytesIO(image_data))

    face_img = decode_base64_image(data.faceImageBase64)
    shape_img = decode_base64_image(data.shapeImageBase64)
    color_img = decode_base64_image(data.colorImageBase64)

    # Convert PIL to numpy format expected by hair_fast.swap
    import numpy as np
    face_np = np.array(face_img) / 255.0
    shape_np = np.array(shape_img) / 255.0
    color_np = np.array(color_img) / 255.0

    result = hair_fast.swap(face_np, shape_np, color_np)

    # Encode result back to base64
    result_img = Image.fromarray((result * 255).astype("uint8"))
    buf = io.BytesIO()
    result_img.save(buf, format="PNG")
    base64_str = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "message": "Success",
        "image_base64": f"data:image/png;base64,{base64_str}"
    }
