import cv2
import numpy as np
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang='en')

def extract_text_from_plate(cropped_img):
    result = ocr.predict(np.array(cropped_img))
    plates = []
    for item in result:
        for line in item:
            text = line[1][0]
            score = line[1][1]
            plates.append((text, score))
    return plates

def process_frame_with_ocr(frame, yolo_model, plate_class_id=0, conf=0.3):
    results = yolo_model.predict(frame, conf=conf)
    boxes = results[0].boxes
    annotated = frame.copy()

#     for box in boxes:
#         cls_id = int(box.cls[0].item())
#         if cls_id != plate_class_id:
#             continue  # Only process license plate class
# # 
    for box in boxes:
        cls_id = int(box.cls[0])
        print(f"Detected class ID: {cls_id}")  # ✅ MOVE IT HERE

        if cls_id != plate_class_id:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        plate_crop = frame[y1:y2, x1:x2]
        plates = extract_text_from_plate(plate_crop)

        for text, score in plates:
            label = f"{text} ({score:.2f})"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    return annotated


