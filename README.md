# Audio Censor App Documentation

## Overview
Audio Censor App is a modern desktop application for detecting and censoring explicit content in audio files (MP3, WAV, M4A, AAC, OGG, FLAC) using OpenAI's Whisper model for transcription and pydub for audio processing. It supports English, Hindi, and custom profanity detection, batch processing, and provides a professional PyQt5-based GUI with audio preview, waveform visualization, and advanced settings.

---

## Features
- **Transcribe audio** using Whisper (word-level timestamps)
- **Detect profanity** in English, Hindi, and custom lists (fully editable)
- **Censor audio** by replacing profane words with a beep, muting, or reversing them
- **Batch processing**: Process multiple files in one go
- **Custom profanity management**: Add, remove, import, and export custom words
- **Modern GUI** built with PyQt5 (drag-and-drop, dark mode, responsive layout)
- **Audio preview** and **waveform visualization** (matplotlib)
- **Whisper model selection** (tiny, base, small, medium, large)
- **Detailed logging and error handling**
- **Settings saved automatically**
- **Progress and status updates** during processing

---

## Requirements
- Python 3.8+
- Packages: `openai-whisper`, `pydub`, `PyQt5`, `matplotlib`, `numpy`
- FFmpeg (must be installed and in your system PATH)

---

## Installation
1. **Install Python packages:**
   ```powershell
   pip install openai-whisper pydub PyQt5 matplotlib numpy
   ```
2. **Install FFmpeg:**
   - Download from https://ffmpeg.org/download.html (Windows: use gyan.dev builds)
   - Extract and add the `bin` folder to your system PATH
   - Verify with: `ffmpeg -version`

---

## Usage
1. **Run the app:**
   ```powershell
   python audio_censor_app.py
   ```
2. **Select input audio file(s)** (MP3, WAV, M4A, AAC, OGG, FLAC)
3. **Choose output location** (auto-generated if left blank)
4. **Select Whisper model** (for accuracy/speed tradeoff)
5. **Choose censoring method:**
   - Beep: Replace profane words with a beep sound
   - Mute: Silence the profane words
   - Reverse: Reverse the profane audio segment
6. **Manage profanity words:**
   - Add, remove, import, or export custom words for any language
7. **Preview audio and view waveform** before and after censoring
8. **Start processing:**
   - Click "Start Processing" and wait for completion
   - Progress, logs, and detected profanities are shown in the app

---

## How It Works
1. **Transcription:**
   - Uses Whisper to transcribe the audio and get word-level timestamps
2. **Profanity Detection:**
   - Checks each word against English, Hindi, and custom profanity lists
   - Custom words are saved in `~/.audio_censor/english_profanity.txt`, `hindi_profanity.txt`, or `custom_profanity.txt`
3. **Censoring:**
   - For each detected profane word, replaces the audio segment with a beep, silence, or reversed audio
   - The censored audio is saved to the specified output file

---

## GUI Components
- **File Selection:** Input/output file pickers, drag-and-drop support
- **Options:** Whisper model, censoring method, language selection, advanced settings (padding, beep frequency)
- **Custom Words:** Add/remove/import/export profanity words
- **Audio Preview:** Play original/censored audio, seek with slider
- **Waveform Visualization:** See audio waveform and profanity markers
- **Status:** Progress bar, log area, and detected profanities tab

---

## Troubleshooting
- **Missing dependencies:**
  - Install with `pip install openai-whisper pydub PyQt5 matplotlib numpy`
- **FFmpeg not found:**
  - Ensure FFmpeg is installed and the `bin` folder is in your PATH
- **App not starting:**
  - Run from terminal and check for error messages
- **Audio not playing:**
  - Ensure your system supports the audio format and FFmpeg is working

---

## Credits
- Uses [OpenAI Whisper](https://github.com/openai/whisper) for transcription
- Uses [pydub](https://github.com/jiaaro/pydub) for audio processing
- GUI built with [PyQt5](https://riverbankcomputing.com/software/pyqt/intro)
- Waveform visualization with [matplotlib](https://matplotlib.org/)

---

## License
This project is for educational and personal use. Please respect copyright and privacy laws when processing audio files.
