# Hebrew Whisper Transcription Service

Standalone Hebrew speech-to-text service using faster-whisper and ivrit-ai models.

## Features

- 🎙️ **File transcription** - Transcribe WAV, MP3, and other audio files
- 📡 **Live RTP transcription** - Connect to EC2 relay for live radio
- 🚀 **GPU acceleration** - CUDA support for fast transcription
- 🇮🇱 **Hebrew optimized** - Uses ivrit-ai fine-tuned models
- 📓 **Google Colab ready** - Includes Colab notebook

## Files

```
whisper-standalone/
├── main.py                 # Main entry point
├── whisper_transcriber.py  # Whisper transcription logic
├── rtp_client.py           # RTP/TCP client for live audio
├── requirements.txt        # Python dependencies
├── .env.example            # Environment config template
├── colab_notebook.ipynb    # Google Colab notebook
└── README.md               # This file
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Configuration

```bash
cp .env.example .env
# Edit .env with your settings
```

### Usage

**Interactive mode:**
```bash
python main.py
```

**Transcribe single file:**
```bash
python main.py --file audio.wav
```

**Live RTP transcription:**
```bash
python main.py --live --rtp-host YOUR_EC2_IP --rtp-port 5005
```

**All options:**
```bash
python main.py --help
```

## Google Colab

1. Open `colab_notebook.ipynb` in Google Colab
2. Enable GPU runtime (Runtime → Change runtime type → GPU)
3. Run the cells to install and use

## TCP Commands

Send file paths via TCP:
```bash
echo '/path/to/audio.wav' | nc localhost 9999
```

## Models

Default model: `ivrit-ai/whisper-large-v3-turbo-ct2`

Other options:
- `ivrit-ai/whisper-large-v3-ct2` - Higher quality, slower
- `openai/whisper-small` - Faster, lower quality
- Local path to downloaded model

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| WHISPER_MODEL | ivrit-ai/whisper-large-v3-turbo-ct2 | Model path or HF ID |
| WHISPER_DEVICE | auto | cpu, cuda, or auto |
| WHISPER_COMPUTE_TYPE | int8 | int8, float16, float32 |
| RTP_HOST | | EC2 relay host for live audio |
| RTP_PORT | 5005 | EC2 relay port |
| NC_HOST | 0.0.0.0 | TCP listener host |
| NC_PORT | 9999 | TCP listener port |
