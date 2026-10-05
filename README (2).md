# AeroMind AI

### Your Hand Is The Interface

AeroMind AI is a touchless human-computer interaction system that allows users to write and interact with a computer using hand movements captured through a webcam.

Instead of using a keyboard or touchscreen for basic character input, the user can draw letters or numbers in the air with their index finger. The system tracks the hand using MediaPipe, converts the movement into a drawing, preprocesses the drawing, and uses TensorFlow Lite models to recognize the character.

## Features

- Touchless air-writing using hand tracking
- Real-time index-finger tracking
- AI-based letter recognition
- Numeric recognition mode
- Prediction confidence display
- Automatic prediction after the user stops drawing
- Voice input
- Text-to-speech readback
- Save recognized text as an image
- Gesture-based color palette
- Gesture-controlled game menu
- Hand-controlled Snake game
- Hand-controlled Tic Tac Toe with AI opponent

## Technology Stack

- Python
- OpenCV
- MediaPipe
- TensorFlow Lite
- NumPy
- SpeechRecognition
- pyttsx3
- Pygame

## How It Works

```text
Webcam
   ↓
MediaPipe Hand Detection
   ↓
Index Finger Tracking
   ↓
Air Drawing Canvas
   ↓
Image Preprocessing
   ↓
28 × 28 Model Input
   ↓
TensorFlow Lite Recognition
   ↓
Letter / Number Prediction
   ↓
Text Output / Save / Voice Readback
```

## AI Recognition

AeroMind AI uses two TensorFlow Lite models:

- `combined_letters_fallback_cnn.tflite` — letter recognition
- `digits_fallback_cnn.tflite` — digit recognition

The drawing is converted to grayscale, thresholded, cropped around the written character, resized to 28 × 28 pixels, normalized, and passed to the selected model.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/dhokanekrushna467-web/AeroMind-AI.git
cd AeroMind-AI
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv airdraw_env
.irdraw_env\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

If PowerShell blocks activation, you can run the program after installing packages with:

```powershell
airdraw_env\Scripts\python.exe airdrawfull.py
```

### 4. Run AeroMind AI

```powershell
python airdrawfull.py
```

A working webcam is required.

## Controls

### Main interface

- `Text` — letter recognition mode
- `Numeric` — digit recognition mode
- `Save` — save recognized text and read it aloud
- `C` — clear current drawing
- `R` — reset drawing and recognized text
- `Space` — insert a space
- `Backspace` — remove the last character
- `Enter` — insert a new line
- `V` — voice input
- `P` — save and read text
- `1 / 2 / 3 / 4` — select drawing color
- `Q` or `Esc` — exit

### Hand gestures

- Three fingers held up — open the color palette
- Five fingers held up — open the game menu
- Index finger — draw / interact

## Project Structure

```text
AeroMind-AI/
│
├── airdrawfull.py
├── combined_letters_fallback_cnn.tflite
├── digits_fallback_cnn.tflite
├── requirements.txt
├── README.md
└── .gitignore
```

## Dataset

The repository contains the trained inference models used by the application. The full training dataset is not included in this repository.

## Future Improvements

- Better recognition accuracy for different handwriting styles
- Continuous word and sentence recognition
- Improved gesture stability
- Multi-hand interaction
- Personalized handwriting adaptation
- Better performance under different lighting conditions
- Integration with external applications
- Multimodal gesture + voice interaction

## Project Goal

AeroMind AI explores a more natural way of interacting with computers by making the user's hand the interface.

**Your Hand Is The Interface.**
