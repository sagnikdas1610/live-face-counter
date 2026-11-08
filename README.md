# Live Face Counter

A local web app that detects faces from a webcam or an uploaded photo using OpenCV YuNet.

## Requirements

- Python 3.10 or later
- A webcam for live camera mode (optional)

## Run on Windows

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python .\FaceCounter.py
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser. The app attempts to open the page automatically. If webcam access is unavailable, use Browser Camera Mode or upload a photo.

The YuNet model file is included in this repository. If it is missing, the app attempts to download it from the OpenCV Zoo repository.

## Stop

Click **Stop Server** in the page or press `Ctrl+C` in the terminal.
