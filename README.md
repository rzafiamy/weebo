# Weebo

A real-time speech-to-speech chatbot powered by Whisper Small, Llama 3.2, and Kokoro-82M.

Works on Apple Silicon.

Learn more [here](https://amanvir.com/weebo).

## Features

- Continuous speech recognition using Whisper MLX
- Natural language responses via Llama
- Real-time text-to-speech synthesis with Kokoro-82M
- Support for different voices
- Streaming response generation

## Setup

Download required models:

- [`kokoro-v0_19.onnx`](https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx) (TTS model)
- Pull the llama3.2 model using Ollama

## Usage

Run the chatbot:

```bash
python main.py
```

The program will start listening for voice input. Speak naturally and wait for a brief pause - the bot will respond with synthesized speech. Press Ctrl+C to stop.
