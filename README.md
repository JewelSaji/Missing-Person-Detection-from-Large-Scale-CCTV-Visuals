# 🧠 CNN-Based Framework for Missing Person Identification and Violence Detection

A deep learning-powered surveillance system that identifies missing persons and detects violence in real-time video streams — enhanced with distributed processing for scalability.

---

## 🚀 Project Highlights

This project provides a robust AI-based solution for two critical challenges in public safety:

- **🔍 Missing Person Identification:** Detects and recognizes missing individuals using CNN-based facial recognition.
- **🛡️ Violence Detection:** Uses a trained CNN to analyze video feeds and detect violent activity.
- **⚙️ Distributed Architecture:** Processes large-scale data efficiently using Apache Spark.
- **🖥️ User-Friendly Interface:** Offers an easy-to-use GUI for media input and result viewing.
- **📄 Automated Reporting:** Generates detailed summaries and PDF reports of detections.

---

## 🧠 System Architecture

```plaintext
               ┌────────────────────────────┐
               │       User Interface       │
               │       (Tkinter GUI)        │
               └────────────┬───────────────┘
                            │
               ┌────────────▼───────────────┐
               │    Media Input (Images,    │
               │       Video Files)         │
               └────────────┬───────────────┘
                            │
        ┌───────────────────▼────────────────────┐
        │         Preprocessing Pipeline         │
        │ (Resizing, Normalization, Face Cropping)│
        └───────────────────┬────────────────────┘
                            │
 ┌──────────────────────────▼──────────────────────────┐
 │               Deep Learning Inference               │
 │  ┌────────────────────────┐   ┌───────────────────┐ │
 │  │  Face Recognition (CNN)│   │ Violence Detection│ │
 │  └────────────────────────┘   └───────────────────┘ │
 └──────────────────────────┬──────────────────────────┘
                            │
              ┌─────────────▼─────────────┐
              │ Distributed Processing    │
              │     via Apache Spark      │
              └─────────────┬─────────────┘
                            │
              ┌─────────────▼─────────────┐
              │     Report Generation     │
              │     (PDF, Stats, Logs)    │
              └─────────────┬─────────────┘
                            │
              ┌─────────────▼─────────────┐
              │       Output Results      │
              │  (GUI, Logs, PDF Reports) │
              └───────────────────────────┘
```

---

## 🛠️ Components

| File | Description |
|------|-------------|
| `ui.py` | GUI for user interaction |
| `missing_person_detection.py` | CNN-based face recognition logic |
| `violence_detection.py` | CNN-based violence classification logic |
| `spark_processing.py` | Spark-powered distributed processing |
| `report_generation.py` | Generates detailed reports and logs |
| `start_cluster.py` | Launches Apache Spark cluster |

---

## 🧰 Technologies Used

- **Languages:** Python  
- **Deep Learning:** TensorFlow / Keras, PyTorch (facenet-pytorch)  
- **Computer Vision:** OpenCV  
- **Distributed Computing:** Apache Spark (PySpark)  
- **UI Framework:** Tkinter  
- **Utilities:** NumPy, Matplotlib, FPDF  

---

## ⚙️ Setup Instructions

### 🔹 Prerequisites

Make sure the following are installed:

- Python 3.8+
- Java 8+
- Apache Spark
- `virtualenv` (recommended)

### 🔹 Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/yourproject.git
   cd yourproject
   ```

2. Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Unix/Linux
   .\venv\Scripts\activate   # Windows
   ```

3. Install all dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Launch the Spark cluster:
   ```bash
   python start_cluster.py
   ```

5. Run the application:
   ```bash
   python ui.py
   ```

---

## 📬 Contribution

Contributions are welcome! Feel free to fork the repository, make changes, and submit a pull request. For major changes, please open an issue to discuss what you'd like to improve.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙌 Credits

**Developed by:** Jewel Saji  
**Degree Program:** B.Tech in Artificial Intelligence & Machine Learning  
**Email:** [jewelsaji026@gmail.com](mailto:jewelsaji026@gmail.com)

---
