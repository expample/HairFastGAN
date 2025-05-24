from fastapi import FastAPI
from hair_swap import HairFast, get_parser
from pathlib import Path
from torchvision.utils import save_image
from pydantic import BaseModel
import torch

app = FastAPI()

model_args = get_parser().parse_args([])
hair_fast = HairFast(model_args)

class ImagePaths(BaseModel):
    face_path: str
    shape_path: str
    color_path: str

@app.post("/swap/")
def swap_images(paths: ImagePaths):
    face = Path(paths.face_path)
    shape = Path(paths.shape_path)
    color = Path(paths.color_path)

    result = hair_fast.swap(face, shape, color)
    output_path = Path("output") / f"{face.stem}_{shape.stem}_{color.stem}.png"
    output_path.parent.mkdir(exist_ok=True)
    save_image(result, output_path)

    return {"output_path": str(output_path)}
