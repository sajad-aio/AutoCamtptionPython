# AutoCamtptionPython
Automatic caption generation system for images and videos using deep learning (python/STT) to create descriptive captions.
------------
# AutoCaption 🎬

## Overview
AutoCaption is an AI-powered web and Telegram bot application that automatically generates subtitles (captions) for videos.  
It extracts audio from uploaded videos, reduces background noise, performs speech-to-text transcription, translates if required, and generates `.srt` subtitle files.  

The system includes:
- 🌐 Web interface (Flask)
- 🤖 Telegram bot interface
- 🎤 Speech recognition (with `speech_recognition`)
- 🌍 Translation (via GoogleTranslator)
- 🔊 Noise reduction and audio preprocessing
- 🗄 User/project management with SQLite

---

## Features
- User authentication (Register/Login)
- Upload videos (mp4, avi, mov, mkv)
- Automatic speech-to-text conversion
- Noise reduction and silence detection
- Subtitle generation in `.srt` format
- Automatic translation to target language
- Save and manage projects (per user)
- Telegram bot integration for quick use

---

## Project Structure
autocaption/
│── sourcecodes/
│ ├── app.py # Flask web app
│ ├── bot.py # Telegram bot
│ ├── static/ # JS, CSS
│ ├── templates/ # HTML templates
│ ├── uploads/ # Uploaded videos (auto-created)
│ └── processed/ # Processed outputs (auto-created)
│── requirements.txt

---

## Requirements
- Python 3.8+
- Flask, Flask-Login, Flask-SQLAlchemy
- MoviePy
- Pydub
- SpeechRecognition
- Deep-Translator
- Noisereduce
- Librosa, SoundFile
- OpenCV
- python-telegram-bot
- SQLite (built-in)

Install dependencies:
```bash
pip install -r requirements.txt
Usage
Run Flask Web App
cd sourcecodes
python app.py
```
Visit: http://127.0.0.1:5000
```bash
Run Telegram Bot

Edit your BOT_TOKEN inside bot.py, then run:

cd sourcecodes
python bot.py
```
How it works

User uploads a video (via web or Telegram bot).
System extracts audio, removes background noise.
Speech Recognition → converts speech to text.
Optional: Translate text to target language.
Generate subtitles in .srt format.
Save project and allow download.

Future Improvements
Multi-speaker diarization
Support for YouTube links
Improved translation quality
Cloud deployment (Heroku, Docker)

Contributing
Fork the repo
Create a feature branch
Open a Pull Request with details

License
MIT License

Contact
Author: Sajad Ali Bakhshi
📧 Email: sajadalibakhshi1389@gmail.com
