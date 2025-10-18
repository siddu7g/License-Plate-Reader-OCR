License Plate Detection and Reader (YOLO + EasyOCR)

This project detects and reads vehicle license plates using a **custom-trained YOLO model** and **EasyOCR** for text extraction.  
---
Prerequisite
1. Activate a Conda environment and install dependencies
 
```bash
conda create -n plate_ocr python=3.9 -y
conda activate plate_ocr

pip install ultralytics easyocr opencv-python
