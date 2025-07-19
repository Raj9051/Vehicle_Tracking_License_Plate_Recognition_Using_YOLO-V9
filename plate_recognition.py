import argparse


def custom_args(parser):
    parser.epilog = (parser.epilog or "") + "\nTo analyse the image for redaction: python number_plate_redaction.py --api-key MY_API_KEY --split-image /tmp/car.jpg"

    parser.add_argument(
        "--split-image",
        action="store_true",
        help="Do extra lookups on parts of the image. Useful on high resolution images.",
    )
    parser.add_argument(
        "--show-boxes",
        action="store_true",
        help="Display the resulting blurred image.",
    )
    parser.add_argument(
        "--save-blurred",
        action="store_true",
        help="Blur license plates and save image in filename_blurred.jpg.",
    )
    parser.add_argument(
        "--ignore-regexp",
        action="append",
        help="Plate regex to ignore during blur. Usually invalid plate numbers.",
    )
    parser.add_argument(
        "--ignore-no-bb",
        action="store_true",
        help="Ignore detections without a vehicle bounding box during blur.",
    )
    parser.add_argument(
        "--detection-threshold",
        type=float,
        default=0.2,
        help="Keep all detections above this threshold. Between 0 and 1.",
    )
    parser.add_argument(
        "--ocr-threshold",
        type=float,
        default=0.5,
        help="Keep all plates if the characters reading score is above this threshold. Between 0 and 1.",
    )


def parse_arguments(custom_args_fn=None):
    parser = argparse.ArgumentParser(
        description="Mock plate recognition API interface",
        epilog="Example: python number_plate_redaction.py image.jpg"
    )
    parser.add_argument("files", nargs="+", help="Image files to process")
    parser.add_argument("--api-key", type=str, required=False, help="API Key (mock)")
    parser.add_argument("--regions", nargs="*", default=["us"], help="Regions (mock)")
    parser.add_argument("--sdk-url", type=str, default="", help="SDK URL (mock)")
    
    if custom_args_fn:
        custom_args_fn(parser)
    
    return parser.parse_args()

from paddleocr import PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

def recognition_api(image_bytes, regions, api_key, sdk_url, config=None):
    from PIL import Image
    import numpy as np
    image = Image.open(image_bytes).convert('RGB')
    image_np = np.array(image)
    results = ocr.ocr(image_np, cls=True)

    detected_plates = []
    for line in results[0]:
        box, (text, score) = line
        xmin = int(min([pt[0] for pt in box]))
        ymin = int(min([pt[1] for pt in box]))
        xmax = int(max([pt[0] for pt in box]))
        ymax = int(max([pt[1] for pt in box]))

        detected_plates.append({
            "plate": text,
            "score": score,
            "box": {
                "xmin": xmin, "ymin": ymin,
                "xmax": xmax, "ymax": ymax
            },
            "vehicle": {
                "score": 0.9,  # optional
                "box": {
                    "xmin": xmin - 5, "ymin": ymin - 5,
                    "xmax": xmax + 5, "ymax": ymax + 5
                }
            }
        })

    return {"results": detected_plates}



def draw_bb(image, results):
    from PIL import ImageDraw
    draw = ImageDraw.Draw(image)
    for res in results:
        b = res["box"]
        draw.rectangle([b["xmin"], b["ymin"], b["xmax"], b["ymax"]], outline="red", width=2)
        draw.text((b["xmin"], b["ymin"] - 10), res.get("plate", "N/A"), fill="red")
    return image




from paddleocr import PaddleOCR
from PIL import Image
import numpy as np

ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Open image using PIL
img = Image.open(r"C:\Users\shubh\OneDrive\Desktop\test\carDetection\images\photo_1.jpg")
 # Replace with actual path
img_np = np.array(img)

# OCR directly on NumPy image
result = ocr.ocr(img_np)
print(result)
