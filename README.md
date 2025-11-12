# 🚗 AI Accident Detection Using YOLOv8  

An AI-powered accident detection system built using **YOLOv8** (You Only Look Once), capable of identifying road accidents from CCTV images and videos in real time.  
The model is trained on a custom dataset of accident and non-accident images to enable smart and efficient traffic monitoring.

---

## 📘 Project Overview

This project demonstrates how **deep learning and computer vision** can be applied to improve road safety by detecting accidents automatically from visual footage.  
We utilized **YOLOv8**, a state-of-the-art object detection algorithm, to detect accident regions within frames and raise automated alerts.

**Key Goals:**
- Detect accident scenes in real-time video feeds  
- Differentiate between accident and non-accident frames  
- Enable future integration with IoT alert systems  

---

## 🧠 Concept Behind the Project

YOLOv8 (by Ultralytics) is a **one-stage object detector** that directly predicts bounding boxes and class probabilities from full images in a single forward pass.  
This makes it ideal for real-time detection tasks such as **road accident detection** from CCTV footage.

**Working Principle:**
1. The input image/video is divided into grid cells.  
2. Each cell predicts bounding boxes and class probabilities.  
3. Non-Max Suppression (NMS) removes overlapping boxes.  
4. Final output shows bounding boxes labeled as *“Accident”* or *“No Accident”* with confidence scores.

---

## 🧩 Features

✅ Detects accidents in real time from images or videos  
✅ Trained on custom annotated dataset (1200+ samples)  
✅ Built using YOLOv8 small model for fast performance  
✅ Compatible with Google Colab GPU for training  
✅ Accurate and lightweight model ready for deployment  

---

## ⚙️ Setup and Usage

### 1️⃣ Clone Repository
```bash
git clone https://github.com/<your-username>/Accident-Detection-Model.git
cd Accident-Detection-Model
```
2️⃣ Install Dependencies
```bash
pip install ultralytics==8.0.20
```
3️⃣ Verify GPU (for Colab users)
```bash
!nvidia-smi
```
4️⃣ Train Model
```bash
!yolo task=detect mode=train model=yolov8s.pt data=data.yaml epochs=25 imgsz=640 plots=True
```
5️⃣ Validate Model
```bash
!yolo task=detect mode=val model=runs/detect/train/weights/best.pt data=data.yaml
```

6️⃣ Run Predictions
```bash
!yolo task=detect mode=predict model=runs/detect/train/weights/best.pt source='data/test/images' conf=0.25
```

📊 Dataset Details - https://drive.google.com/drive/u/0/folders/1WJm1ozQRfHYQwyNyYUm9mr8TpDQeUDuM

Total Images: 1200+ (Accident & Non-Accident)

Collected from: YouTube, Google Images, Kaggle

Annotated using: Roboflow

Format: YOLOv8 (with .txt label files)

Split Ratio: Train 80%, Validation 10%, Test 10%

Classes:

0: Accident
1: NoAccident

🧠 Deep Learning Concepts Used
| Concept                       | Description                                             |
| ----------------------------- | ------------------------------------------------------- |
| **YOLOv8**                    | Single-stage CNN-based object detector                  |
| **Transfer Learning**         | Used pretrained YOLOv8 weights fine-tuned for accidents |
| **Bounding Box Regression**   | Predicts (x, y, width, height) for detected regions     |
| **Non-Max Suppression (NMS)** | Eliminates overlapping bounding boxes                   |
| **Evaluation Metrics**        | mAP, Precision, Recall, F1-score used for validation    |


🎯 Results
| Metric    | Score (example) |
| --------- | --------------- |
| mAP50     | 0.89            |
| Precision | 0.91            |
| Recall    | 0.87            |

Example Output:

✅ Accident Detected [Confidence: 0.94]

https://github.com/user-attachments/assets/df17ac07-b159-4978-b3f0-6e35250c589b

![car crash output](https://github.com/user-attachments/assets/08f53ed0-c25c-418b-9785-c9ee039e759d)
![car crash output  2jpg](https://github.com/user-attachments/assets/fcfa7855-2af4-4a2e-aadb-705e05693224)


🚀 Future Scope

- Real-time deployment on CCTV networks

- Integration with IoT-based emergency alert systems

- Multi-class accident severity detection (fire, collision types)

- Web dashboard using Flask/Streamlit for live monitoring
