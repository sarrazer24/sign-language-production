# Sign to Text — Local Setup Guide

This guide explains how to run the **Sign to Text** feature locally on your machine.

---

## Prerequisites

Make sure you have the following installed:

- Python 3.9 or higher → https://www.python.org/downloads/
- Flutter SDK → https://docs.flutter.dev/get-started/install
- Git → https://git-scm.com/downloads
- Google Chrome

---

## Step 1 — Clone the repository

```bash
git clone https://github.com/sarrazer24/sign-language-production.git
cd sign-language-production
```

---

## Step 2 — Get the model files

The model files are too large for GitHub. Download them from the shared Google Drive link (ask Sara) and place them inside the `sign_to_text/` folder:

```
sign-language-production/
└── sign_to_text/
    ├── best.pt           ← download this
    ├── label_map.txt     ← download this
    ├── local_api.py      ← already in repo
    └── detect_asl_realtime.py  ← already in repo
```

---

## Step 3 — Install Python dependencies

```bash
cd sign_to_text
pip install flask flask-cors ultralytics opencv-python numpy
```

> On Windows, if `pip` doesn't work, try `pip3` or `python -m pip install ...`

---

## Step 4 — Start the local detection API

Open a terminal in `sign_to_text/` and run:

```bash
cd sign_to_text
python local_api.py
```

You should see:
```
Model loaded — 34 classes
Running on http://0.0.0.0:5000
```

**Keep this terminal open** while using the app.

> On Windows, if the camera doesn't work, make sure no other app is using it (Teams, Zoom, etc.)

---

## Step 5 — Install Flutter dependencies

Open a new terminal:

```bash
cd app/sign
flutter pub get
```

---

## Step 6 — Run the Flutter app

```bash
cd app/sign
flutter run -d chrome --web-browser-flag "--disable-web-security"
```

The app will open in Chrome automatically.

---

## Step 7 — Use Sign to Text

1. In the app, click **Sign to Text**
2. Click **Start Recording**
3. Show a sign in front of your webcam
4. The detected sign appears on screen in real time

**Controls during detection:**
- `Q` — quit the detector window
- `C` — clear sign history
- `S` — save a screenshot

---

## Running both terminals at once (summary)

| Terminal | Command | Location |
|----------|---------|----------|
| Terminal 1 | `python local_api.py` | `sign_to_text/` |
| Terminal 2 | `flutter run -d chrome --web-browser-flag "--disable-web-security"` | `app/sign/` |

---

## Troubleshooting

**Camera not opening**

On Windows, edit `detect_asl_realtime.py` and change:
```python
cap = cv2.VideoCapture(0)
# to
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
```

**Port 5000 already in use**

```bash
# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# Mac/Linux
lsof -i :5000
kill -9 <PID>
```

**Module not found errors**

```bash
pip install flask flask-cors ultralytics opencv-python numpy
```

**Flutter web security error**

Make sure you always launch Flutter with:
```bash
flutter run -d chrome --web-browser-flag "--disable-web-security"
```

**Model not detecting anything**

- Make sure your hand is clearly visible in the frame
- Make sure the lighting is good
- Try moving closer to the camera
- The model detects: A-Z letters + Hello, Goodbye, Please, Sorry, ThankYou, Yes, No, ILoveYou

---

## Project Structure

```
sign-language-production/
├── app/
│   └── sign/                        ← Flutter app
│       ├── lib/
│       │   ├── screens/
│       │   │   └── sign_to_text_screen.dart
│       │   └── services/
│       │       └── detection_service.dart
│       └── pubspec.yaml
├── sign_to_text/                    ← ML model + API
│   ├── best.pt                      ← YOLOv8 model (not in Git)
│   ├── label_map.txt                ← class names
│   ├── local_api.py                 ← Flask API (port 5000)
│   └── detect_asl_realtime.py      ← standalone detector
├── phase1_text_to_pose/
├── phase2_pose_to_video/
└── README.md
```

---

## Detected Classes (34 total)

**Letters:** A B C D E F G H I J K L M N O P Q R S T U V W X Y Z

**Words:** Hello · Goodbye · Please · Sorry · ThankYou · Yes · No · ILoveYou
