from ultralytics import YOLO
import cv2
import easyocr
import os
import re

# CONFIGURATION
PLATE_IMAGE_PATH = r"C:\Users\sidat\yolov12\car_dataset\CarLongPlateGen2078.jpg"
MODEL_PATH = r"C:\Users\sidat\yolov12\Models\license_plate_detector.pt"
OUTPUT_DIR = r"C:\Users\sidat\yolov12\outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# OCR Parameters
OCR_PARAMS = {
    'contrast_ths': 0.3,
    'text_threshold': 0.7,
    'low_text': 0.4,
    'width_ths': 0.7,
    'height_ths': 0.7,
    'decoder': 'beamsearch',
    'detail': 1,
    'paragraph': False
}

# Detection Settings
MIN_PLATE_AREA = 500
DETECTION_CONF = 0.5
DETECTION_IOU = 0.45
PADDING = 10

# Pipeline for Processing YOLOv8 + easyOCR
class LicensePlateReader:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.reader = easyocr.Reader(['en'], gpu=True)
        print("[INFO] License Plate Reader initialized")

    def read_plate_text(self, cropped_plate):
        #Perform OCR on a cropped license plate image
        gray = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY) if len(cropped_plate.shape) == 3 else cropped_plate
        resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        try:
            results = self.reader.readtext(resized, **OCR_PARAMS)
            if results:
                results = sorted(results, key=lambda x: x[0][0][0])  # maintain reading order
                text = "".join([r[1] for r in results])
                clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
                return clean_text
        except Exception as e:
            print(f"[ERROR] OCR failed: {e}")
        return ""

    def detect_and_read(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            print(f"[ERROR] Could not load image: {image_path}")
            return [], None

        results = self.model(img, conf=DETECTION_CONF, iou=DETECTION_IOU, verbose=False)
        detected_plates = []

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])

            area = (x2 - x1) * (y2 - y1)
            if area < MIN_PLATE_AREA:
                continue

            y1, y2 = max(0, y1 - PADDING), min(img.shape[0], y2 + PADDING)
            x1, x2 = max(0, x1 - PADDING), min(img.shape[1], x2 + PADDING)

            cropped_plate = img[y1:y2, x1:x2]
            plate_text = self.read_plate_text(cropped_plate)

            if plate_text:
                detected_plates.append({'text': plate_text, 'confidence': confidence})
                print(f"[✅] Plate detected: {plate_text} (conf: {confidence:.2f})")
                self.annotate_plate(img, x1, y1, x2, y2, plate_text)
            else:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        return detected_plates, img
    # Post-Processing Pipeline
    def annotate_plate(self, img, x1, y1, x2, y2, text):
        # Draw bounding box and label on the image
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(text, font, 0.9, 2)
        cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw + 10, y1), (0, 255, 0), -1)
        cv2.putText(img, text, (x1 + 5, y1 - 5), font, 0.9, (0, 0, 0), 2)

def main():
    lpr = LicensePlateReader(MODEL_PATH)

    print(f" Processing: {PLATE_IMAGE_PATH}")
    plates, annotated_img = lpr.detect_and_read(PLATE_IMAGE_PATH)

    if annotated_img is not None:
        output_path = os.path.join(OUTPUT_DIR, "result.jpg")
        cv2.imwrite(output_path, annotated_img)
        print("DETECTION RESULTS")
        if plates:
            for i, plate in enumerate(plates, 1):
                print(f"{i}. {plate['text']} (confidence: {plate['confidence']:.1%})")
        else:
            print("No license plates detected")
        print("=" * 50)
        print(f"Output saved: {output_path}")

if __name__ == "__main__":
    main()