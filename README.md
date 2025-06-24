# 🎤 Live TTS Translator - Whisper AI

A real-time speech recognition and translation tool powered by **Whisper AI** with **WebGPU acceleration**. Perfect for streamers who need live translation overlays.

## ✨ Features

- **🤖 Whisper AI Integration**: Uses Xenova/transformers.js with WebGPU acceleration
- **🌐 Real-time Translation**: Automatic speech-to-text with instant translation
- **🎮 Streaming Ready**: Auto-starts listening, perfect for OBS overlays
- **📱 No Server Required**: Runs entirely in the browser on GitHub Pages
- **🔧 Customizable**: Adjustable font size, opacity, and languages
- **🎯 Multi-language Support**: Auto-detect input, translate to any target language

## 🚀 Quick Start for Streamers

### Option 1: Use GitHub Pages (Recommended)

1. **Fork this repository** to your GitHub account
2. **Enable GitHub Pages** in your repository settings
3. **Access your live translator** at: `https://yourusername.github.io/LiveTTS`
4. **For streaming overlay**: Add `#overlay` to the URL: `https://yourusername.github.io/LiveTTS#overlay`

### Option 2: Direct Use

Simply open `index.html` in a modern browser that supports WebGPU (Chrome, Edge).

## 🎬 Streaming Setup

### For OBS Studio:
1. Add a **Browser Source**
2. Set URL to: `https://yourusername.github.io/LiveTTS#overlay`
3. Set Width: `800`, Height: `400` (adjust as needed)
4. Check **"Shutdown source when not visible"** for better performance
5. The translator will **automatically start listening** when loaded

### Features for Streamers:
- ✅ **Auto-starts listening** - No manual intervention needed
- ✅ **Overlay mode** - Minimalist UI perfect for streaming
- ✅ **Customizable appearance** - Adjust font size and background opacity
- ✅ **Multiple languages** - Auto-detect input, translate to any language
- ✅ **Text-to-speech** - Optional voice output of translations

## 🛠️ Browser Requirements

- **Chrome 113+** or **Edge 113+** (for WebGPU support)
- **Modern browser** with microphone access
- **HTTPS connection** (automatic with GitHub Pages)

## 🎯 How It Works

1. **Model Loading**: Automatically downloads and caches Whisper-base model (~200MB)
2. **WebGPU Acceleration**: Uses GPU when available for faster processing
3. **Real-time Processing**: Processes 5-second audio segments continuously
4. **Translation**: Uses free translation APIs (MyMemory, LibreTranslate)
5. **Speech Output**: Optional text-to-speech for translations

## 🔧 Customization

Access settings by visiting the main page (without `#overlay`):
- **Font Size**: Adjust text size for better visibility
- **Background Opacity**: Control overlay transparency
- **Languages**: Set input (auto-detect) and output languages
- **Text-to-Speech**: Enable/disable voice output

## 📋 Language Support

**Input**: Auto-detect or specify (English, Spanish, French, German, Italian, Portuguese, Russian, Japanese, Korean, Chinese, Arabic, Hindi)

**Output**: All supported languages above

## 🎤 Audio Processing

- **Sample Rate**: 16kHz (optimal for Whisper)
- **Channels**: Mono
- **Processing**: 5-second segments with echo cancellation
- **Latency**: ~2-3 seconds from speech to translation

## 🔒 Privacy

- **100% Client-side**: Everything runs in your browser
- **No data sent to servers**: Except for translation (using free public APIs)
- **Local caching**: Model downloads once and caches for future use
- **Microphone access**: Required for speech recognition

## 💡 Tips for Best Results

- **Use a good microphone** for clearer speech recognition
- **Speak clearly** and at normal pace
- **WebGPU browsers** provide better performance
- **Stable internet** needed for initial model download
- **Allow microphone access** when prompted

## 🆘 Troubleshooting

**Model won't load**: Check internet connection, try refreshing
**No microphone access**: Check browser permissions
**Poor transcription**: Ensure clear audio input and minimal background noise
**Slow performance**: Use Chrome/Edge with WebGPU support

## 📄 License

MIT License - Feel free to use for streaming, modify, and distribute.

---

Perfect for streamers who want **real-time translation** without complex setups! 🎮✨ 