# 🚗 Smart Car Parking & Registration System

An **AI-powered smart parking system** that automatically detects cars, manages parking slots, and supports vehicle registration using computer vision and deep learning.

This project is designed for **smart cities, compounds, malls, and private parking facilities**, focusing on automation, accuracy, and scalability.

---

## ✨ Features

### 🚘 Vehicle Detection

* Real-time car detection using **YOLOv8**
* Accurate detection in different lighting and angles

### 🅿️ Smart Parking Slot Management

* Uses **predefined parking masks** to detect occupied vs free slots
* Supports high-resolution parking layouts (Full HD)

### 🪪 Vehicle Registration (Extensible)

* Prepared architecture for car/plate registration
* Easy integration with future **license plate recognition (LPR)** modules

### 🌐 API & App Separation

* Modular design (`api.py`, `app.py`, `main.py`)
* Ready for FastAPI or Streamlit-based deployment

---

## 📁 Project Structure

```text
Smat_Car_Parking_and_registration_system/
│
├── api.py                # API endpoints (parking status, detection)
├── app.py                # Application logic
├── main.py               # Entry point
├── util.py               # Helper functions
│
├── yolov8n.pt            # YOLOv8 pretrained model
├── mask_1920_1080.png    # Parking slot mask (Full HD)
├── mask_crop.png         # Cropped mask for ROI
│
├── requirements.txt      # Project dependencies
├── LICENSE.md            # License information
└── README.md             # Project documentation
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Smat_Car_Parking_and_registration_system.git
cd Smat_Car_Parking_and_registration_system
```

Install dependencies:

```bash
pip install -r requirements.txt
```

> ⚠️ Python 3.8+ is recommended.

---

## ▶️ Usage

Run the system:

```bash
python main.py
```

The system will:

1. Load YOLOv8 model
2. Process camera/video frames
3. Detect vehicles
4. Determine parking slot availability

---

## 🧠 How It Works

### Detection Pipeline

1. Input frame from camera or video
2. YOLOv8 detects vehicles
3. Parking mask defines valid slot areas
4. Intersection logic decides:

   * 🟥 Occupied slot
   * 🟩 Free slot

### Mask-Based Slot Detection

* Masks represent parking slot locations
* Pixel overlap with detected bounding boxes determines occupancy

---

## 📊 Model & Assets

| File                 | Description              |
| -------------------- | ------------------------ |
| `yolov8n.pt`         | Vehicle detection model  |
| `mask_1920_1080.png` | Full parking layout mask |
| `mask_crop.png`      | Region of interest mask  |

---

## 🛠️ Technologies Used

* Python
* OpenCV
* YOLOv8 (Ultralytics)
* NumPy
* FastAPI / Streamlit (optional deployment)

---

## 🧪 Possible Extensions

* License Plate Recognition (LPR)
* Database integration (slots, users, history)
* Web dashboard for admins
* Mobile app integration
* Multi-camera support

---

## 🎯 Use Cases

* Smart compounds
* Shopping malls
* Universities
* Private & public parking garages

---

## 👨‍💻 Author

**Mohamed Elsawy**
AI Engineering Student – Mansoura University

---

## 📜 License

This project is licensed under the terms described in `LICENSE.md`.

---

⭐ If you find this project useful, don’t forget to star the repository!

