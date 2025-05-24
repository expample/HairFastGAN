import argparse
import os
import sys
from pathlib import Path

from torchvision.utils import save_image
from tqdm.auto import tqdm

from hair_swap import HairFast, get_parser


def main(model_args, args):
    hair_fast = HairFast(model_args)

    experiments: list[str | tuple[str, str, str]] = []
    if args.file_path is not None:
        with open(args.file_path, 'r') as file:
            experiments.extend(file.readlines())

    if all(path is not None for path in (args.face_path, args.shape_path, args.color_path)):
        experiments.append((args.face_path, args.shape_path, args.color_path))

    for exp in tqdm(experiments):
        if isinstance(exp, str):
            file_1, file_2, file_3 = exp.split()
        else:
            file_1, file_2, file_3 = exp

        face_path = args.input_dir / file_1
        shape_path = args.input_dir / file_2
        color_path = args.input_dir / file_3

        base_name = '_'.join([path.stem for path in (face_path, shape_path, color_path)])
        exp_name = base_name if model_args.save_all else None

        if isinstance(exp, str) or args.result_path is None:
            os.makedirs(args.output_dir, exist_ok=True)
            output_image_path = args.output_dir / f'{base_name}.png'
        else:
            os.makedirs(args.result_path.parent, exist_ok=True)
            output_image_path = args.result_path

        final_image = hair_fast.swap(face_path, shape_path, color_path, benchmark=args.benchmark, exp_name=exp_name)
        save_image(final_image, output_image_path)


if __name__ == "__main__":
    model_parser = get_parser()
    parser = argparse.ArgumentParser(description='HairFast evaluate')
    parser.add_argument('--input_dir', type=Path, default='', help='The directory of the images to be inverted')
    parser.add_argument('--benchmark', action='store_true', help='Calculates the speed of the method during the session')

    # Arguments for a set of experiments
    parser.add_argument('--file_path', type=Path, default=None,
                        help='File with experiments with the format "face_path.png shape_path.png color_path.png"')
    parser.add_argument('--output_dir', type=Path, default=Path('output'), help='The directory for final results')

    # Arguments for single experiment
    parser.add_argument('--face_path', type=Path, default=None, help='Path to the face image')
    parser.add_argument('--shape_path', type=Path, default=None, help='Path to the shape image')
    parser.add_argument('--color_path', type=Path, default=None, help='Path to the color image')
    parser.add_argument('--result_path', type=Path, default=None, help='Path to save the result')

    args, unknown1 = parser.parse_known_args()
    model_args, unknown2 = model_parser.parse_known_args()

    unknown_args = set(unknown1) & set(unknown2)
    if unknown_args:
        file_ = sys.stderr
        print(f"Unknown arguments: {unknown_args}", file=file_)

        print("\nExpected arguments for the model:", file=file_)
        model_parser.print_help(file=file_)

        print("\nExpected arguments for evaluate:", file=file_)
        parser.print_help(file=file_)

        sys.exit(1)

    main(model_args, args)
from fastapi import FastAPI, File, UploadFile
from pathlib import Path
from hair_swap import HairFast, get_parser
import shutil

app = FastAPI()

@app.post("/swap")
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
    output_image.save(output_path)
    
    return {"message": "Success", "output_image": str(output_path)}
    import os
from fastapi import Depends, HTTPException, Header

API_KEY = os.getenv("API_KEY")  # קריאה למפתח מהמשתנים הסביבתיים

def verify_api_key(api_key: str = Header(...)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")  # חסימה במקרה של מפתח שגוי

@app.post("/swap", dependencies=[Depends(verify_api_key)])
async def swap_images(face: UploadFile = File(...), shape: UploadFile = File(...), color: UploadFile = File(...)):
    # כאן נמצא הקוד הרגיל של עיבוד התמונה
