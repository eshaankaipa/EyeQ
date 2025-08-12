# Real-time Text Detection Camera Application

This application provides a real-time camera feed with advanced text detection capabilities. It can automatically detect and use your iPhone 14 via Continuity Camera for 60fps video feed, making it perfect for capturing and analyzing math problems and other text content.

## Features

- **Continuity Camera Support**: Automatically detects and uses your iPhone 14 camera for 60fps feed
- **Real-time Text Detection**: Advanced OCR with multiple engines (EasyOCR, Tesseract, Google Vision)
- **High-Quality Video**: Supports up to 1920x1080 resolution at 60fps
- **Smart Camera Detection**: Automatically finds the best available camera
- **Manual Camera Selection**: Option to manually select specific cameras
- **Image Capture**: Timestamp-based image capture with text detection overlays
- **AI Integration**: Built-in support for sending images to LLMs for problem solving

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Continuity Camera Setup

To use your iPhone 14 camera for 60fps video feed:

1. **Enable Continuity Camera on your iPhone:**
   - Go to Settings > General > AirPlay & Handoff
   - Turn on 'Continuity Camera'

2. **Ensure both devices are signed into the same Apple ID**

3. **Position your iPhone:**
   - Place it on a stand or hold it steady
   - Point it at the text you want to analyze
   - Ensure good lighting for better text detection

4. **The app will automatically detect your iPhone camera**

## Usage

1. Run the camera application:
```bash
python text_detection_camera.py
```

2. **Camera Controls:**
   - Press `c` to solve (capture crop + send to LLM)
   - Press `t` to toggle text detection on/off
   - Press `a` to toggle accurate mode (slower but more accurate)
   - Press `g` to cycle granularity (paragraph/line/word)
   - Press `f` to show/save full recognized text
   - Press `q` to quit

3. **Camera Selection:**
   - The app automatically detects and uses your iPhone camera if available
   - If iPhone camera is not detected, you can manually select from available cameras
   - Manual selection is offered when iPhone camera is not found

## File Structure

```
aiglasses/
├── text_detection_camera.py  # Main camera application with text detection
├── requirements.txt          # Python dependencies
├── README.md                # This file
└── captured_images/         # Directory where captured images are stored
    ├── captured_image_YYYYMMDD_HHMMSS.jpg
    ├── text_detection_YYYYMMDD_HHMMSS.jpg
    └── ocr_crop_YYYYMMDD_HHMMSS.jpg
```

## Requirements

- Python 3.7+
- Webcam or camera device (iPhone 14 via Continuity Camera recommended)
- OpenCV
- NumPy
- EasyOCR or Tesseract (for text detection)
- Google Cloud Vision (optional, for enhanced OCR)

## Troubleshooting

### Camera Issues
- **iPhone camera not detected**: Ensure Continuity Camera is enabled on your iPhone
- **Camera permissions**: On macOS, grant camera permissions to your terminal/IDE
- **Multiple cameras**: Use manual camera selection if auto-detection fails

### Text Detection Issues
- **Poor OCR accuracy**: Try accurate mode ('a' key) for better results
- **No text detected**: Ensure good lighting and steady camera positioning
- **Missing dependencies**: Install OCR libraries with `pip install easyocr` or `pip install pytesseract`

### General Issues
- **Import errors**: Install all dependencies with `pip install -r requirements.txt`
- **Performance issues**: Use fast mode for higher FPS, accurate mode for better detection
- **iPhone connection**: Ensure both devices are on the same WiFi network and signed into the same Apple ID

## Tips for Best Results

1. **iPhone Positioning**: Keep your iPhone steady on a stand or tripod
2. **Lighting**: Ensure good, even lighting on the text
3. **Text Quality**: Use clear, printed text rather than handwritten text
4. **Camera Distance**: Position the camera 6-12 inches from the text
5. **Stability**: Avoid motion blur by keeping the camera steady 