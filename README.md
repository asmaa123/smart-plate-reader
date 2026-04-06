# 🚗 Smart Plate Reader

> Real-time license plate detection, tracking, and OCR for Arabic & English plates — built with YOLO26 + EasyOCR + OpenCV.

---

## ✨ Features

- 🎯 **Vehicle Detection** — YOLO26 custom-trained model for accurate plate localization
- 🔁 **Multi-Vehicle Tracking** — persistent ID per vehicle across all frames
- 🌍 **Multilingual OCR** — reads Arabic and English license plates
- 🗳️ **Voting System** — aggregates OCR results over 8 frames for higher accuracy
- 🖥️ **Live Dashboard** — real-time overlay showing vehicle count, plates found & timestamp
- 🛤️ **Trailing Path** — color-coded movement trail per vehicle
- 📊 **CSV Export** — auto-saves every detected plate with vehicle ID, timestamp & confidence

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| YOLO26 | Vehicle & plate detection |
| EasyOCR | Arabic + English text recognition |
| OpenCV | Video processing & visualization |
| Google Colab | Training & inference environment |
| NumPy / Pandas | Data handling |

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/smart-plate-reader.git
cd smart-plate-reader
```

### 2. Install dependencies
```bash
pip install ultralytics easyocr opencv-python-headless numpy pandas
```

### 3. Add your model
Place your trained `best.pt` file in the project root or update `MODEL_PATH` in the script.

### 4. Run on Google Colab
Upload `main.py` to Colab, mount your Drive, and run — the script will prompt you to upload a video.

---

## ⚙️ Configuration

```python
MODEL_PATH   = "/content/drive/MyDrive/best.pt"  # path to your YOLO model
CONF_THRESH  = 0.4     # detection confidence threshold
OCR_CONF     = 0.25    # OCR confidence threshold
TRAIL_LEN    = 30      # length of vehicle movement trail
VOTE_FRAMES  = 8       # frames used for OCR voting
LANGUAGES    = ['ar', 'en']  # supported OCR languages
```

---

## 📦 Output

After processing, you get:

- **`output_pro.mp4`** — annotated video with boxes, IDs, trails, plate text & dashboard
- **`plates_log.csv`** — full log of detected plates

```
frame | time_sec | vehicle_id | plate_text | confidence
------|----------|------------|------------|----------
120   | 4.0      | 3          | أ ب ج 1234 | 0.87
```

---

## 🧠 How the Voting System Works

Raw OCR on a single frame is unreliable due to blur and motion. Instead, the system:

1. Runs OCR every 5 frames per tracked vehicle
2. Stores the last 8 OCR results per vehicle ID
3. Picks the **most frequent result** as the final plate text

This significantly improves accuracy on low-resolution or moving plates.

---

## 📋 Preprocessing Pipeline

Each plate crop goes through the following before OCR:

```
Original crop → Grayscale → 2x Upscale → Bilateral Filter → Otsu Threshold → EasyOCR
```

---

## 📌 Roadmap

- [ ] Speed estimation per vehicle
- [ ] Web interface (Streamlit or Gradio)
- [ ] Support for more languages
- [ ] Export to JSON alongside CSV
- [ ] Docker support

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

---

## 📄 License

MIT License — feel free to use, modify, and share.

---

<p align="center">Built with ❤️ using Python & Computer Vision</p>
