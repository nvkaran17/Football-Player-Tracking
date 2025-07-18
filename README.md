
# ⚽ Football Player Tracking Using YOLOv11 🧠🎥

This project detects and tracks football players in a video using the YOLOv11 object detection model. It ensures **smooth, real-time tracking**, assigns **unique IDs to each player**, and displays the **total number of players in the frame** — all while keeping the bounding boxes minimal and clean.

---

## 🔍 Features

- 🎯 **Player Detection** using YOLOv11
- 🔁 **Simple & Stable Tracking** – one box per player, no duplicates
- 🧠 **Retains Player ID** even if the player leaves and re-enters the frame
- ⚫ **Clean Black Bounding Boxes** for a professional look
- 🔢 **Player Count** shown live on screen
- 🖥️ Built with **OpenCV + PyTorch**

---

## 📦 Model Download

The YOLOv11 model file (`best.pt`) is too large for GitHub.  
Please download it manually from Google Drive and place it in the `yolov11_model/` directory:

👉 [Download best.pt (Google Drive)](# ⚽ Football Player Tracking Using YOLOv11 🧠🎥

This project detects and tracks football players in a video using the YOLOv11 object detection model. It ensures **smooth, real-time tracking**, assigns **unique IDs to each player**, and displays the **total number of players in the frame** — all while keeping the bounding boxes minimal and clean.

---

## 🔍 Features

- 🎯 **Player Detection** using YOLOv11
- 🔁 **Simple & Stable Tracking** – one box per player, no duplicates
- 🧠 **Retains Player ID** even if the player leaves and re-enters the frame
- ⚫ **Clean Black Bounding Boxes** for a professional look
- 🔢 **Player Count** shown live on screen
- 🖥️ Built with **OpenCV + PyTorch**

---

## 📦 Model Download

The YOLOv11 model file (`best.pt`) is too large for GitHub.  
Please download it manually from Google Drive and place it in the `yolov11_model/` directory:

👉 [Download best.pt (Google Drive)](https://drive.google.com/file/d/1yxuC79etWqpEJdy9o-RnEj7I0nxF3CQK/view?usp=sharing)

---

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/nvkaran17/Football-Player-Tracking.git
cd Football-Player-Tracking
)

---

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/nvkaran17/Football-Player-Tracking.git
cd Football-Player-Tracking
2. Create Virtual Environment (Recommended)

python -m venv venv
venv\Scripts\activate  # On Windows
3. Install Dependencies

pip install -r requirements.txt
🚀 Usage

python main.py
This will open a video window that:

Detects all players in each frame

Tracks them with consistent black bounding boxes

Displays a live count of total players currently on screen

📁 Project Structure

Football-Player-Tracking/
│
├── yolov11_model/
│   └── best.pt                # Download and place here manually
├── utils/
│   ├── tracker.py             # Simple object tracker
│   └── helpers.py             # Drawing + preprocessing helpers
├── main.py                    # Main detection & tracking script
├── requirements.txt
└── README.md
🎥 Sample Output
Replace this with your own video/gif later


👤 Author
Karan NV
📧 nvkaran33@gmail.com
🌐 GitHub Profile - https://github.com/nvkaran17

🏁 Future Improvements
 Team-based colored tracking (Red/Blue)

 Ball tracking

 Player statistics overlay (speed, distance run, etc.)


