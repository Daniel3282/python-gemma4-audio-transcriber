🚀 Gemma 4 Audio Transcriber

A modern, lightweight desktop application built with Python and PySide6 for high-precision audio transcription.

The main breakthrough of this project is that it eliminates the need for traditional speech-to-text tools such as Whisper. Instead, it directly harnesses the power of LiteRT-LM and Gemma 4 (2B), taking full advantage of the model's native ability to process and transcribe audio.

✨ Key Features

100% Local AI: All audio processing and transcription are performed locally on your computer using Gemma 4 and LiteRT-LM. Your audio does not need to be uploaded to a cloud service or sent to an external transcription API.

Unlimited Transcription: Transcribe as many hours of audio as you want. There are no artificial limits on transcription time, number of files, or number of transcriptions. The only practical limitations are your computer's available storage, memory, and processing power.

Fast and Accurate Multilingual Transcription: Delivers excellent performance and multilingual accuracy, particularly for languages beyond English.

Smart Audio Chunking: Analyzes silence within the audio to automatically split long recordings into optimal segments, making it possible to transcribe lengthy files smoothly.

Modern Interface: Features a clean, sleek, and responsive dark-mode UI built with a professional QSS theme.

Asynchronous Processing: Model loading and transcription run in background threads (QThread), keeping the application UI smooth and fully responsive.

Privacy-Friendly: Because transcription happens entirely on your machine, your audio files and transcriptions remain under your control instead of being sent to third-party cloud services.

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
python main.py# python-gemma4-audio-transcriber
Lightweight Python &amp; PySide6 desktop app for multi-language audio transcription. Powered by LiteRT-LM and Gemma 4 (2B), it eliminates external tools like Whisper, delivering faster and more accurate native multilingual results with smart audio chunking and a modern dark UI.
