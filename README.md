# 🚀 Weebo

A fork of Weebo, which is a real-time speech-to-speech chatbot powered by:
- ⚡ Fast Whisper
- 🤖 Ollama models (`gemma3:1B`)
- 🗣️ Kokoro-82M

The original Weebo is great for Apple Silicon, but not optimized for CUDA-based systems. This fork aims to provide a better experience on NVIDIA GPUs. 🎯

---

## ✨ Features

✅ Continuous speech recognition using Fast Whisper  
✅ Natural language responses via Ollama models  
✅ Real-time text-to-speech synthesis with Kokoro-82M  
✅ Support for different voices 🎙️  
✅ Streaming response generation 📡  

---

## 🔧 Setup

### 1️⃣ Download KOKORO-82M from GitHub Repository

```bash
wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx
```

---

### 2️⃣ Install Ollama from the Official Website

To install Ollama on your Linux system, follow these steps:

#### 🔹 a. Download and Install Ollama:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

#### 🔹 b. Verify the installation:

```bash
ollama --version
```

#### 🔹 c. Download Gemma3:1B model

```bash
ollama pull gemma3:1B
```

---

### 3️⃣ Install Fast Whisper and Dependencies

Fast Whisper is an optimized implementation of Whisper designed for real-time speech recognition. It requires **cuDNN 9** to function properly. 🚀

#### 🏗️ Step 1: Download and Install the CUDA Keyring

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
```

#### 🏗️ Step 2: Install cuDNN

```bash
sudo apt-get -y install cudnn
```

#### 🏗️ Step 3: Install Specific CUDNN Version

For CUDA 11:
```bash
sudo apt-get -y install cudnn-cuda-11
```
For CUDA 12:
```bash
sudo apt-get -y install cudnn-cuda-12
```

---

### 4️⃣ Configure Environment 🌍

#### 🔹 a. Update `.bashrc`

Add the following lines to your `.bashrc` file (adjust for your CUDA version):

```bash
export PATH=/usr/local/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:/usr/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.6
```

Then, apply the changes:
```bash
source ~/.bashrc
```

#### 🔹 b. Virtual Environment Activation 🐍

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### 🔹 c. Install Python Requirements 📦

```bash
pip install -r requirements.txt
```

---

## 🎙️ Usage

Run the chatbot:

```bash
python main.py
```

The program will start listening for voice input. Speak naturally and wait for a brief pause - the bot will respond with synthesized speech. 🗣️🎧 Press **Ctrl+C** to stop. ❌

---

## 🛠️ Troubleshooting

If you encounter any issues, try the following:
- Ensure your GPU drivers and CUDA are correctly installed ⚙️
- Verify that `ollama` and `Fast Whisper` are installed 🔍
- Check the dependencies in `requirements.txt` 📜
- Restart your terminal session after updating `.bashrc` 🔄

---

## 💡 Contributing

Contributions are welcome! Feel free to fork this repository, submit issues, or create pull requests. 🙌

---

## 📜 License

This project is licensed under the MIT License. 📄

---

## ❤️ Acknowledgments

Special thanks to the creators of the original Weebo and the developers of Fast Whisper, Ollama, and Kokoro-82M. 👏

