🚀 Gemma 4 Audio Transcriber

A modern, lightweight desktop application built with Python and PySide6 for high-precision audio transcription.

The main breakthrough of this project is that it eliminates the need for traditional speech-to-text tools such as Whisper. Instead, it directly harnesses the power of LiteRT-LM and Gemma 4 (2B), taking full advantage of the model's native ability to process and transcribe audio.

✨ Key Features

100% Native AI: Leverages Gemma 4's advanced multimodal capabilities to process audio natively, without relying on complex external speech-to-text dependencies.

Fast and Accurate Multilingual Transcription: Delivers excellent performance and multilingual accuracy, particularly for languages beyond English.

Smart Audio Chunking: Analyzes silence within the audio to automatically split long recordings into optimal segments, making it possible to transcribe lengthy files smoothly.

Modern Interface: Features a clean, sleek, and responsive dark-mode UI built with a professional QSS theme.

Asynchronous Processing: Model loading and transcription run in background threads (QThread), keeping the application UI smooth and fully responsive.

📦 Installation

Follow these steps to set up and run the project locally.

1. Clone the Repository

Clone the repository and navigate to the project directory:

git clone https://github.com/Daniel3282/python-gemma4-audio-transcriber.git
cd python-gemma4-audio-transcriber

2. Install the Required Python Dependencies
pip install huggingface_hub litert-lm pydub PySide6

3. Install FFmpeg

FFmpeg is required for audio processing.

Windows: Download FFmpeg from the official website or install it using Chocolatey:

choco install ffmpeg


macOS: Install it using Homebrew:

brew install ffmpeg


Linux (Ubuntu/Debian): Install it using APT:

sudo apt install ffmpeg

4. Run the Application
python main.py