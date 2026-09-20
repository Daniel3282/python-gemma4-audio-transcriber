import os
import sys
from huggingface_hub import hf_hub_download
import litert_lm
from pydub import AudioSegment
from pydub.silence import split_on_silence
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QFrame,
)

# Modern Visual Theme (QSS)
MODERN_STYLE = """
QMainWindow {
    background-color: #0f172a;
}
QWidget {
    color: #f8fafc;
    font-family: 'Segoe UI', -apple-system, sans-serif;
    font-size: 14px;
}
QLabel {
    color: #cbd5e1;
    font-weight: 600;
}
QLineEdit {
    background-color: #1e293b;
    border: 2px solid #334155;
    border-radius: 8px;
    padding: 10px 14px;
    color: #f8fafc;
    font-size: 14px;
}
QLineEdit:focus {
    border: 2px solid #38bdf8;
}
QTextEdit {
    background-color: #1e293b;
    border: 2px solid #334155;
    border-radius: 8px;
    padding: 12px;
    color: #f8fafc;
    selection-background-color: #38bdf8;
    selection-color: #0f172a;
}
QTextEdit:focus {
    border: 2px solid #38bdf8;
}
QPushButton {
    background-color: #334155;
    color: #f8fafc;
    border: none;
    border-radius: 8px;
    padding: 10px 18px;
    font-weight: 600;
}
QPushButton:hover {
    background-color: #475569;
}
QPushButton:pressed {
    background-color: #1e293b;
}
QPushButton#btn_transcribe {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0284c7, stop:1 #0ea5e9);
    color: white;
    font-size: 15px;
    padding: 12px;
    border-radius: 8px;
}
QPushButton#btn_transcribe:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0369a1, stop:1 #0284c7);
}
QPushButton#btn_transcribe:disabled {
    background-color: #1e293b;
    color: #64748b;
}
QFrame#card {
    background-color: #1e293b;
    border-radius: 12px;
    border: 1px solid #334155;
}
"""

class ModelLoaderWorker(QThread):
    loaded = Signal(str)
    status_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.repo_id = "litert-community/gemma-4-E2B-it-litert-lm"
        self.filename = "gemma-4-E2B-it.litertlm"

    def run(self):
        self.status_changed.emit("⏳ Loading Gemma 4 model in the background...")
        try:
            model_path = hf_hub_download(
                repo_id=self.repo_id, filename=self.filename, local_dir="./models"
            )
            self.loaded.emit(model_path)
            self.status_changed.emit("✅ Model ready! Select an audio file to begin.")
        except Exception as e:
            self.status_changed.emit(f"❌ Error loading model: {e}")


class LongAudioTranscriptionWorker(QThread):
    status_changed = Signal(str)
    partial_text_received = Signal(str)

    def __init__(self, audio_path, prompt_text, model_path, parent=None):
        super().__init__(parent)
        self.audio_path = audio_path
        self.prompt_text = prompt_text
        self.model_path = model_path

    def run(self):
        try:
            self.status_changed.emit("✂️ Analyzing silences to split audio...")
            audio = AudioSegment.from_file(self.audio_path)

            raw_chunks = split_on_silence(
                audio,
                min_silence_len=500,
                silence_thresh=audio.dBFS - 16,
                keep_silence=250,
            )

            if not raw_chunks:
                raw_chunks = [
                    audio[i : i + 25000] for i in range(0, len(audio), 25000)
                ]

            chunks = []
            current_chunk = AudioSegment.empty()
            for chunk in raw_chunks:
                if len(current_chunk) + len(chunk) < 30000:
                    current_chunk += chunk
                else:
                    if len(current_chunk) > 0:
                        chunks.append(current_chunk)
                    current_chunk = chunk
            if len(current_chunk) > 0:
                chunks.append(current_chunk)

            total_chunks = len(chunks)

            with litert_lm.Engine(
                self.model_path,
                backend=litert_lm.Backend.GPU(),
                audio_backend=litert_lm.Backend.CPU(),
            ) as engine:

                for index, chunk in enumerate(chunks):
                    self.status_changed.emit(
                        f"⚙️ Transcribing part {index + 1} of {total_chunks}..."
                    )

                    temp_chunk_path = f"temp_chunk_{index}.wav"
                    chunk.export(temp_chunk_path, format="wav")

                    try:
                        with engine.create_conversation() as conversation:
                            contents = [
                                self.prompt_text,
                                litert_lm.Content.AudioFile(absolute_path=temp_chunk_path),
                            ]
                            response = conversation.send_message(
                                litert_lm.Contents.of(*contents)
                            )
                            partial_text = response["content"][0]["text"]
                            self.partial_text_received.emit(partial_text)
                    finally:
                        if os.path.exists(temp_chunk_path):
                            try:
                                os.remove(temp_chunk_path)
                            except:
                                pass

            self.status_changed.emit("✅ Transcription successfully completed!")

        except Exception as e:
            self.partial_text_received.emit(f"\n[Error during process: {e}]")
            self.status_changed.emit("❌ Execution error.")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Long Audio Transcriber • Gemma 4")
        self.resize(850, 720)
        self.model_path = None

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(24, 24, 24, 24)
        main_layout.setSpacing(16)

        # App Title
        title_label = QLabel("✨ Smart Transcriber with Gemma 4")
        title_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #38bdf8;")
        main_layout.addWidget(title_label)

        # Input Section (File + Prompt)
        card_frame = QFrame()
        card_frame.setObjectName("card")
        card_layout = QVBoxLayout(card_frame)
        card_layout.setContentsMargins(20, 20, 20, 20)
        card_layout.setSpacing(14)

        # Audio File
        card_layout.addWidget(QLabel("📂 Audio File (WAV / MP3 / AAC / FLAC)"))
        file_layout = QHBoxLayout()
        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText("Select or drag and drop audio file...")
        file_layout.addWidget(self.path_input)

        self.btn_browse = QPushButton("Browse...")
        self.btn_browse.setCursor(Qt.PointingHandCursor)
        self.btn_browse.clicked.connect(self.select_audio_file)
        file_layout.addWidget(self.btn_browse)
        card_layout.addLayout(file_layout)

        # Instruction Prompt
        card_layout.addWidget(QLabel("💬 AI Instruction / Prompt"))
        self.prompt_input = QTextEdit()
        self.prompt_input.setFixedHeight(75)
        self.prompt_input.setText(
            "Accurately transcribe this audio segment into text. "
            "Output only the transcription without extra comments."
        )
        card_layout.addWidget(self.prompt_input)

        main_layout.addWidget(card_frame)

        # Execution Button (Disabled until model loads)
        self.btn_transcribe = QPushButton("🚀 Start Smart Transcription")
        self.btn_transcribe.setObjectName("btn_transcribe")
        self.btn_transcribe.setCursor(Qt.PointingHandCursor)
        self.btn_transcribe.setEnabled(False)
        self.btn_transcribe.clicked.connect(self.start_transcription)
        main_layout.addWidget(self.btn_transcribe)

        # Status Bar
        self.status_label = QLabel("⚡ Initializing system...")
        self.status_label.setStyleSheet("color: #94a3b8; font-style: italic; font-weight: 500;")
        main_layout.addWidget(self.status_label)

        # Output Box
        main_layout.addWidget(QLabel("📝 Real-Time Output"))
        self.text_output = QTextEdit()
        self.text_output.setPlaceholderText("Transcribed chunks will appear here progressively...")
        main_layout.addWidget(self.text_output)

        # Start model loading in the background as soon as the window opens
        self.init_model_loader()

    def init_model_loader(self):
        self.loader_worker = ModelLoaderWorker()
        self.loader_worker.status_changed.connect(self.status_label.setText)
        self.loader_worker.loaded.connect(self.on_model_loaded)
        self.loader_worker.start()

    def on_model_loaded(self, path):
        self.model_path = path
        self.btn_transcribe.setEnabled(True)

    def select_audio_file(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio",
            "",
            "Audio Files (*.wav *.mp3 *.m4a *.flac)",
        )
        if file_name:
            self.path_input.setText(file_name)

    def append_transcription_part(self, text):
        current_text = self.text_output.toPlainText()
        if current_text:
            self.text_output.setText(current_text + " " + text)
        else:
            self.text_output.setText(text)

    def start_transcription(self):
        audio_path = self.path_input.text().strip()
        prompt_text = self.prompt_input.toPlainText().strip()

        if not audio_path or not os.path.exists(audio_path):
            self.status_label.setText("⚠️ Please select a valid audio file before continuing.")
            return

        if not self.model_path:
            self.status_label.setText("⚠️ The model is still loading, please wait a moment.")
            return

        self.btn_transcribe.setEnabled(False)
        self.text_output.clear()

        self.worker = LongAudioTranscriptionWorker(audio_path, prompt_text, self.model_path)
        self.worker.status_changed.connect(self.status_label.setText)
        self.worker.partial_text_received.connect(self.append_transcription_part)
        self.worker.finished.connect(lambda: self.btn_transcribe.setEnabled(True))
        self.worker.start()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(MODERN_STYLE)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())