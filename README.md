# DEPI-Facial-Recognition-Project: Intelligent Shop Security

A state-of-the-art, Full-Stack Facial Recognition System for authentication and identification of individuals using live video streams. Built for commercial security, access control, and smart visitor archiving.

This system upgrades standard facial recognition by integrating a **React.js Dashboard**, a **Node.js/Socket.io Backend**, and a highly optimized **Python/DeepFace AI Engine** with multi-threading and dynamic thresholds.

---

## 🏗️ Project Structure

```text
DEPI-Facial-Recognition-Project/
├── ai_engine/                 # 🧠 Python AI & Computer Vision Service
│   ├── core_logic.py          # DeepFace/FaceNet wrapper & embedding logic
│   └── stream_scanner.py      # Flask video feed, HAAR cascades, & Thread Locks
├── web_app/
│   ├── backend/               # ⚙️ Node.js API & WebSocket Server
│   │   ├── server.js          # Handles file system (FS) and event broadcasting
│   │   └── package.json       # Backend dependencies
│   └── frontend/              # 💻 React.js Interactive UI
│       ├── public/
│       ├── src/               # React components, pages (Dashboard, LiveStream), App.js
│       └── package.json       # Frontend dependencies
├── data/                      # 🗄️ Local File System Database
│   ├── blacklist_db/          # Stored images of blacklisted suspects
│   ├── raw_dataset/
│   ├── processed_dataset/
│   └── visitors_db/           # Auto-generated daily visitor archives
├── models/                    # 🤖 Pre-trained Models
│   ├── haarcascade_frontalface_default.xml
│   └── facenet_keras.h5
├── notebooks/                 # 📓 Jupyter notebooks for early exploration & testing
│   ├── milestone1_data_exploration.ipynb
│   ├── milestone2_model_development.ipynb
│   └── milestone3_realtime_testing.ipynb
├── reports/                   # 📄 Project documentation and milestone reports
│   ├── milestone1_dataset_exploration/
│   ├── milestone2_model_evaluation/
│   ├── milestone3_testing/
│   └── milestone4_mlops/
├── start.bat                  # 🚀 1-Click Startup Script for all microservices
└── requirements.txt           # Python dependencies
```

---

## 🎯 Milestones

### Milestone 1: Data Collection, Exploration, and Preprocessing ✅ Completed

- Obtain labeled facial datasets (LFW, VGGFace).
- Analyze dataset composition, quality, and diversity.
- Preprocess images: resize to 160×160, normalize pixel values, face detection/cropping, augmentation.

### Milestone 2: Facial Recognition Model Development ✅ Completed

- Select and fine-tune a model: FaceNet, VGG-Face, DeepFace, or custom CNN.
- Train using transfer learning.
- Evaluate with Accuracy, Precision, Recall, F1-score, and False Acceptance Rate (FAR).

### Milestone 3: Deployment and Real-Time Testing ✅ Completed

- Deploy the model via Flask API.
- Build a Full-Stack architecture with React.js and Node.js.
- Integrate with live video streams for real-time recognition.
- Test under various conditions (lighting, angles, expressions) with Thread-Safe Sync.

### Milestone 4: MLOps and Monitoring 🔄 In Progress

- Set up MLflow or Kubeflow for experiment tracking.
- Implement a retraining pipeline.
- Monitor FAR and trigger alerts on performance degradation.

### Milestone 5: Final Documentation and Presentation ⏳ Pending

- Full project report covering data, model, deployment, and monitoring.
- Presentation of system architecture and real-world impact.

---

## ⚙️ Prerequisites

Before running the project, ensure your environment has the following installed:

- [Node.js](https://nodejs.org/) (LTS version)
- [Anaconda / Miniconda](https://www.anaconda.com/) (For isolated Python environments)
- [Git](https://git-scm.com/)

---

## 🚀 Getting Started & Installation

### 1. Set up the AI Engine (Python)

Open Anaconda Prompt and create the environment for the AI pipeline:

```bash
conda create -n AI python=3.10 -y
conda activate AI
pip install -r requirements.txt
# Or install manually:
# pip install opencv-python deepface tensorflow scipy flask flask-cors requests numpy
```

### 2. Set up the Backend (Node.js)

Open a terminal in the project root directory and install backend dependencies:

```bash
cd web_app/backend
npm install
```

### 3. Set up the Frontend (React.js)

In the same or a new terminal, install the frontend dependencies:

```bash
cd web_app/frontend
npm install
```

---

## 🏃‍♂️ Running the System

To launch the complete system (AI Stream + Backend API + React Dashboard), simply run the batch script from your project root:

```bash
start.bat
```
```bash
start.bat
```
