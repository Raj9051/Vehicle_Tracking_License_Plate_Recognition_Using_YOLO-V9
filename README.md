Vehicle Detection and License Plate Recognition

A real-time system that detects vehicles and recognizes license plates using YOLOv9 for object detection and PaddleOCR for optical character recognition, integrated into a user-friendly Streamlit interface.

🚀 Features

Vehicle detection using YOLOv9 (Ultralytics)
License plate text recognition using PaddleOCR
Real-time webcam, RTSP, YouTube, or uploaded file input
Interactive frontend built with Streamlit
Tracking with SORT algorithm
Frame-wise annotations and result logging (CSV)

📁 Project Structure

├── carDetection/
│   ├── plate_video_utils.py
│   ├── vehicle_detect.py
│   ├── streamlit_app.py
│   ├── models/
│   └── images/
├── requirements.txt
├── README.md
└── LICENSE

⚙️ Setup Instructions

1. Clone the Repository

git clone https://github.com/your-username/vehicle-plate-recognition.git
cd vehicle-plate-recognition

2. Create and Activate Virtual Environment

python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

3. Install Dependencies

pip install -r requirements.txt

▶️ Run the Application

Streamlit App:
streamlit run streamlit_app.py
From Script (Example):
python plate_video_utils.py --input video.mp4

📦 Dependencies

Python 3.9+
Ultralytics (YOLOv9)
PaddleOCR
OpenCV
Streamlit
NumPy, Pandas, Matplotlib, Seaborn

📊 Output

Live video or image feed with vehicle and plate annotations
Exportable CSV log of detected plate numbers and timestamps
Real-time confidence score overlay.
<img width="1920" height="1080" alt="Screenshot (1)" src="https://github.com/user-attachments/assets/a10accff-997d-4435-a59b-c6aaeacc10a2" />
<img width="263" height="79" alt="Screenshot 2025-07-01 013621" src="https://github.com/user-attachments/assets/a62e1e1b-aa68-4103-8802-090b0335f40b" />
<img width="1839" height="829" alt="Screenshot 2025-07-01 013522" src="https://github.com/user-attachments/assets/7f83e18e-69d9-40b6-859e-2069b3da23d9" />
<img width="1784" height="840" alt="Screenshot 2025-04-21 150317" src="https://github.com/user-attachments/assets/49e8496d-7d35-460e-886d-724b15ab7c01" />

📌 Future Improvements

Add multi-language license plate support
Night vision and low-light enhancements
Cloud deployment with API endpoint
Face detection and vehicle type classification

🧑‍💻 Contributors
Shubham Raj – Final Year B.Tech CSE (Data Science)

🙌 Acknowledgements
Ultralytics YOLOv9
PaddleOCR
OpenCV
Streamlit

