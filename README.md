License Plate Detection and Reader (YOLO + EasyOCR)

This project detects and reads vehicle license plates using a **custom-trained YOLO model** and **EasyOCR** for text extraction.  
---
Prerequisite
1. Activate a Conda environment and install dependencies
 
```bash
conda create -n plate_ocr python=3.9 -y
conda activate plate_ocr

pip install ultralytics easyocr opencv-python

```
2. Dataset

This project was tested on the **License Plate Detection Dataset** from [Roboflow Universe](https://app.roboflow.com/sidg/cars-plates-sqjgv-jvvfe/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true).

- **Source**: Roboflow
- **Dataset Version**: v3
- **License**: CC BY 4.0
All dataset credits belong to the original contributors on Roboflow.
