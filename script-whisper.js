class LiveTranslator {
    constructor() {
        this.pipeline = null;
        this.mediaRecorder = null;
        this.audioStream = null;
        this.isListening = false;
        this.currentLanguage = 'auto';
        this.targetLanguage = 'en';
        this.apiKey = localStorage.getItem('googleTranslateApiKey') || '';
        this.history = JSON.parse(localStorage.getItem('translationHistory') || '[]');
        this.audioChunks = [];
        this.isProcessing = false;
        this.lastDetectedLanguage = null;
        this.isModelLoaded = false;
        
        this.initializeElements();
        this.setupLoadingScreen();
        this.initializeEventListeners();
        this.loadSettings();
        this.displayHistory();
        
        // Check for overlay mode
        if (window.location.hash === '#overlay') {
            document.body.classList.add('overlay-mode');
        }
        
        // Auto-load the model on startup
        this.autoLoadModel();
    }
    
    initializeElements() {
        this.elements = {
            // Loading screen elements
            loadingScreen: document.getElementById('loadingScreen'),
            mainApp: document.getElementById('mainApp'),
            loadingProgress: document.getElementById('loadingProgress'),
            progressFill: document.querySelector('.progress-fill'),
            progressText: document.querySelector('.progress-text'),
            
            // Main app elements
            startBtn: document.getElementById('startBtn'),
            stopBtn: document.getElementById('stopBtn'),
            statusText: document.getElementById('statusText'),
            micStatus: document.getElementById('micStatus'),
            originalText: document.getElementById('originalText'),
            translatedText: document.getElementById('translatedText'),
            inputLanguage: document.getElementById('inputLanguage'),
            outputLanguage: document.getElementById('outputLanguage'),
            historyContainer: document.getElementById('historyContainer'),
            clearHistory: document.getElementById('clearHistory'),
            fontSize: document.getElementById('fontSize'),
            fontSizeValue: document.getElementById('fontSizeValue'),
            bgOpacity: document.getElementById('bgOpacity'),
            bgOpacityValue: document.getElementById('bgOpacityValue'),
            autoSpeak: document.getElementById('autoSpeak'),
            overlayMode: document.getElementById('overlayMode'),
            apiKey: document.getElementById('apiKey'),
            saveApiKey: document.getElementById('saveApiKey')
        };
    }
    
    setupLoadingScreen() {
        // Show loading screen initially
        this.elements.loadingScreen.style.display = 'flex';
        this.elements.mainApp.style.display = 'none';
    }
    
    async autoLoadModel() {
        // Auto-load the model immediately on page load
        setTimeout(() => {
            this.loadWhisperModel();
        }, 500); // Small delay to ensure DOM is ready
    }
    
    async loadWhisperModel() {
        try {
            this.elements.loadingProgress.style.display = 'block';
            this.updateLoadingProgress(0, 'Initializing Whisper AI with WebGPU...');
            
            console.log('Starting Whisper model loading...');
            
            // Import Transformers.js dynamically with the latest version
            this.updateLoadingProgress(5, 'Loading Transformers.js library...');
            console.log('Importing Transformers.js...');
            
            const { pipeline, env } = await import('https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2');
            console.log('Transformers.js imported successfully');
            
            // Configure environment for optimal WebGPU performance (like HF demo)
            env.allowRemoteModels = true;
            env.allowLocalModels = true;
            
            // Check for WebGPU support
            if (navigator.gpu) {
                console.log('WebGPU is supported - using GPU acceleration');
                env.backends.onnx.wasm.proxy = false;
                this.updateLoadingProgress(10, 'WebGPU detected - optimizing for GPU acceleration...');
            } else {
                console.log('WebGPU not supported - using optimized WASM');
                env.backends.onnx.wasm.numThreads = navigator.hardwareConcurrency || 4;
                env.backends.onnx.wasm.simd = true;
                env.backends.onnx.wasm.proxy = false;
                this.updateLoadingProgress(10, 'Using optimized WASM backend...');
            }
            
            this.updateLoadingProgress(15, 'Loading Whisper model...');
            console.log('Starting pipeline creation...');
            
            // Add timeout for model loading
            const modelPromise = pipeline('automatic-speech-recognition', 'Xenova/whisper-base', {
                dtype: {
                    encoder_model: 'fp16',
                    decoder_model_merged: 'q4', // Use quantized model for better performance
                },
                device: navigator.gpu ? 'webgpu' : 'wasm',
                progress_callback: (progress) => {
                    console.log('Loading progress:', progress);
                    if (progress.status === 'downloading') {
                        const percent = 15 + Math.round((progress.loaded / progress.total) * 70);
                        this.updateLoadingProgress(percent, `Downloading: ${Math.round((progress.loaded / progress.total) * 100)}%`);
                    } else if (progress.status === 'loading') {
                        this.updateLoadingProgress(90, 'Loading model into memory...');
                    } else if (progress.status === 'ready') {
                        this.updateLoadingProgress(95, 'Model ready - finalizing...');
                    }
                }
            });
            
            // Add 30 second timeout
            const timeoutPromise = new Promise((_, reject) => {
                setTimeout(() => reject(new Error('Model loading timeout - falling back to Web Speech API')), 30000);
            });
            
            this.pipeline = await Promise.race([modelPromise, timeoutPromise]);
            console.log('Pipeline created successfully');
            
            this.updateLoadingProgress(100, 'Model loaded successfully!');
            
            // Wait a moment then show main app and auto-start listening
            setTimeout(() => {
                this.isModelLoaded = true;
                this.showMainApp();
                
                // Auto-start listening for streamers
                setTimeout(() => {
                    this.startListening();
                }, 1000);
            }, 1000);
            
            console.log('Whisper pipeline loaded successfully with WebGPU support');
            
        } catch (error) {
            console.error('Failed to load Whisper pipeline:', error);
            this.updateLoadingProgress(0, 'Failed to load Whisper - switching to Web Speech API...');
            
            // Wait then show main app with fallback
            setTimeout(() => {
                this.initializeFallbackSpeechRecognition();
                this.showMainApp();
                
                // Auto-start listening even in fallback mode
                setTimeout(() => {
                    this.startListening();
                }, 1000);
            }, 2000);
        }
    }
    
    updateLoadingProgress(percent, text) {
        this.elements.progressFill.style.width = `${percent}%`;
        this.elements.progressText.textContent = text;
    }
    
    showMainApp() {
        // Hide loading screen and show main app
        this.elements.loadingScreen.style.display = 'none';
        this.elements.mainApp.style.display = 'block';
        
        // Enable start button
        this.elements.startBtn.disabled = false;
        
        // Update status
        if (this.isModelLoaded) {
            this.updateStatus('Whisper AI ready - Starting automatically...', '⏳');
        } else {
            this.updateStatus('Fallback mode ready - Starting automatically...', '⏳');
        }
    }
    
    async startListening() {
        try {
            // Request microphone access
            this.audioStream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    sampleRate: 16000, // Whisper optimal sample rate
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                } 
            });
            
            if (this.isModelLoaded) {
                // Use Whisper AI
                this.startWhisperListening();
            } else if (this.recognition) {
                // Use Web Speech API fallback
                this.startWebSpeechListening();
            } else {
                throw new Error('No speech recognition available');
            }
            
        } catch (error) {
            console.error('Microphone access error:', error);
            this.updateStatus('Microphone access denied', '❌');
            alert('Microphone access is required. Please allow microphone access and try again.');
        }
    }
    
    startWhisperListening() {
        // Set up MediaRecorder for Whisper AI with optimal settings
        this.mediaRecorder = new MediaRecorder(this.audioStream, {
            mimeType: 'audio/webm;codecs=opus'
        });
        
        this.audioChunks = [];
        this.isListening = true;
        
        this.mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                this.audioChunks.push(event.data);
            }
        };
        
        this.mediaRecorder.onstop = () => {
            if (this.audioChunks.length > 0 && !this.isProcessing && this.isListening) {
                this.processAudioWithWhisper();
            }
        };
        
        // Update UI
        this.elements.startBtn.disabled = true;
        this.elements.stopBtn.disabled = false;
        this.updateStatus('Listening...', '🎤');
        
        // Start recording in segments for real-time processing
        this.startRecordingSegments();
    }
    
    startWebSpeechListening() {
        this.isListening = true;
        try {
            this.recognition.start();
            console.log('Web Speech API started');
        } catch (error) {
            console.error('Failed to start recognition:', error);
            this.updateStatus('Failed to start speech recognition', '❌');
        }
    }
    
    startRecordingSegments() {
        if (!this.isListening) return;
        
        // Record 5-second segments for real-time processing
        this.mediaRecorder.start();
        
        setTimeout(() => {
            if (this.isListening && this.mediaRecorder.state === 'recording') {
                this.mediaRecorder.stop();
                
                // Start next segment after a brief pause
                setTimeout(() => {
                    if (this.isListening) {
                        this.audioChunks = []; // Clear previous chunks
                        this.startRecordingSegments();
                    }
                }, 100);
            }
        }, 5000); // 5-second segments for better accuracy
    }
    
    async processAudioWithWhisper() {
        if (this.isProcessing || this.audioChunks.length === 0 || !this.isModelLoaded) return;
        
        this.isProcessing = true;
        
        try {
            // Convert audio chunks to blob
            const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm;codecs=opus' });
            
            // Convert blob to audio buffer
            const arrayBuffer = await audioBlob.arrayBuffer();
            const audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
            const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
            
            // Get audio data as Float32Array
            let audio = audioBuffer.getChannelData(0);
            
            // Ensure audio is the right length (pad or trim to 30 seconds max)
            const maxLength = 16000 * 30; // 30 seconds at 16kHz
            if (audio.length > maxLength) {
                audio = audio.slice(0, maxLength);
            }
            
            // Transcribe with Whisper
            this.updateStatus('Transcribing...', '🔄');
            
            const result = await this.pipeline(audio, {
                language: this.currentLanguage === 'auto' ? null : this.currentLanguage,
                task: 'transcribe',
                return_timestamps: false,
                chunk_length_s: 30,
                stride_length_s: 5,
            });
            
            // Handle the result
            if (result && result.text && result.text.trim()) {
                const transcription = result.text.trim();
                
                // Update detected language if available
                if (result.language) {
                    this.lastDetectedLanguage = result.language;
                    console.log(`Whisper detected language: ${result.language}`);
                }
                
                // Display transcription
                this.elements.originalText.textContent = transcription;
                
                // Translate if needed
                this.translateText(transcription);
                
                console.log('Transcription:', transcription);
            }
            
        } catch (error) {
            console.error('Audio processing error:', error);
            this.updateStatus('Transcription error', '❌');
        } finally {
            this.isProcessing = false;
        }
    }
    
    stopListening() {
        this.isListening = false;
        
        if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
            this.mediaRecorder.stop();
        }
        
        if (this.recognition) {
            this.recognition.stop();
        }
        
        if (this.audioStream) {
            this.audioStream.getTracks().forEach(track => track.stop());
            this.audioStream = null;
        }
        
        this.updateStatus('Stopped listening', '🎤 Off');
        document.body.classList.remove('listening');
        this.elements.startBtn.disabled = false;
        this.elements.stopBtn.disabled = true;
    }
    
    // Fallback to Web Speech API if Whisper fails
    initializeFallbackSpeechRecognition() {
        if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
            this.updateStatus('Speech recognition not supported in this browser', '❌');
            return;
        }
        
        this.recognition = 'webkitSpeechRecognition' in window 
            ? new webkitSpeechRecognition() 
            : new SpeechRecognition();
        
        this.recognition.continuous = true;
        this.recognition.interimResults = true;
        
        // Set language based on current selection
        this.updateRecognitionLanguage();
        
        this.recognition.onstart = () => {
            this.isListening = true;
            this.updateStatus('Listening (Web Speech API mode)...', '🎤 On');
            document.body.classList.add('listening');
            this.elements.startBtn.disabled = true;
            this.elements.stopBtn.disabled = false;
        };
        
        this.recognition.onresult = (event) => {
            let finalTranscript = '';
            let interimTranscript = '';
            
            for (let i = event.resultIndex; i < event.results.length; i++) {
                const transcript = event.results[i][0].transcript;
                
                if (event.results[i].isFinal) {
                    finalTranscript += transcript;
                } else {
                    interimTranscript += transcript;
                }
            }
            
            const displayText = finalTranscript || interimTranscript;
            this.elements.originalText.textContent = displayText;
            
            if (finalTranscript && finalTranscript.trim().length > 2) {
                this.translateText(finalTranscript);
            }
        };
        
        this.recognition.onend = () => {
            if (this.isListening) {
                // Auto-restart recognition
                setTimeout(() => {
                    if (this.isListening) {
                        try {
                            this.recognition.start();
                        } catch (error) {
                            console.error('Failed to restart recognition:', error);
                            this.stopListening();
                        }
                    }
                }, 100);
            }
        };
        
        this.recognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
            
            if (event.error === 'not-allowed') {
                this.updateStatus('Microphone access denied', '❌');
            } else if (event.error === 'no-speech') {
                // Ignore no-speech errors
                return;
            } else {
                this.updateStatus(`Speech recognition error: ${event.error}`, '❌');
            }
        };
    }
    
    updateRecognitionLanguage() {
        if (this.recognition) {
            const langCode = this.currentLanguage === 'auto' ? 'en-US' : this.getProperLanguageCode(this.currentLanguage);
            this.recognition.lang = langCode;
        }
    }

    initializeEventListeners() {
        this.elements.startBtn.addEventListener('click', () => {
            this.startListening();
        });
        
        this.elements.stopBtn.addEventListener('click', () => {
            this.stopListening();
        });
        
        this.elements.clearHistory.addEventListener('click', () => this.clearHistory());
        this.elements.saveApiKey.addEventListener('click', () => this.saveApiKey());
        
        this.elements.inputLanguage.addEventListener('change', (e) => {
            this.currentLanguage = e.target.value;
            this.lastDetectedLanguage = null;
            this.updateRecognitionLanguage();
        });
        
        this.elements.outputLanguage.addEventListener('change', (e) => {
            this.targetLanguage = e.target.value;
        });
        
        this.elements.fontSize.addEventListener('input', (e) => {
            const size = e.target.value;
            this.elements.fontSizeValue.textContent = `${size}px`;
            document.documentElement.style.setProperty('--font-size', `${size}px`);
            this.saveSettings();
        });
        
        this.elements.bgOpacity.addEventListener('input', (e) => {
            const opacity = e.target.value / 100;
            this.elements.bgOpacityValue.textContent = `${e.target.value}%`;
            document.documentElement.style.setProperty('--bg-opacity', opacity);
            this.saveSettings();
        });
        
        this.elements.autoSpeak.addEventListener('change', () => this.saveSettings());
        
        // Overlay mode functionality
        this.elements.overlayMode.addEventListener('change', (e) => {
            if (e.target.checked) {
                document.body.classList.add('overlay-mode');
                // Minimize UI for streaming
                this.minimizeForOverlay();
            } else {
                document.body.classList.remove('overlay-mode');
                // Restore full UI
                this.restoreFromOverlay();
            }
            this.saveSettings();
        });
    }
    
    minimizeForOverlay() {
        // Hide unnecessary elements for streaming overlay
        const elementsToHide = [
            '.header',
            '.controls',
            '.settings-panel',
            '.history-section',
            '.status-bar'
        ];
        
        elementsToHide.forEach(selector => {
            const element = document.querySelector(selector);
            if (element) {
                element.style.display = 'none';
            }
        });
        
        // Style the translation display for overlay
        const container = document.querySelector('.container');
        if (container) {
            container.style.maxWidth = '100%';
            container.style.padding = '10px';
        }
        
        // Make text boxes more compact
        const textBoxes = document.querySelectorAll('.text-box');
        textBoxes.forEach(box => {
            box.style.minHeight = '60px';
            box.style.margin = '5px 0';
        });
    }
    
    restoreFromOverlay() {
        // Restore all hidden elements
        const elementsToShow = [
            '.header',
            '.controls',
            '.settings-panel',
            '.history-section',
            '.status-bar'
        ];
        
        elementsToShow.forEach(selector => {
            const element = document.querySelector(selector);
            if (element) {
                element.style.display = '';
            }
        });
        
        // Restore normal container styling
        const container = document.querySelector('.container');
        if (container) {
            container.style.maxWidth = '';
            container.style.padding = '';
        }
        
        // Restore normal text box styling
        const textBoxes = document.querySelectorAll('.text-box');
        textBoxes.forEach(box => {
            box.style.minHeight = '';
            box.style.margin = '';
        });
    }
    
    async translateText(text) {
        if (!text.trim()) return;
        
        this.updateStatus('Translating...', '🔄');
        
        try {
            let translatedText = '';
            let serviceUsed = 'Unknown';
            
            // Determine source language for translation
            let sourceLanguage = this.currentLanguage;
            if (this.currentLanguage === 'auto' && this.lastDetectedLanguage) {
                sourceLanguage = this.lastDetectedLanguage;
            }
            
            if (this.apiKey) {
                translatedText = await this.translateWithGoogleAPI(text, sourceLanguage);
                serviceUsed = 'Google Translate API';
            } else {
                // Try free services
                const services = [
                    { fn: () => this.translateWithMyMemory(text, sourceLanguage), name: 'MyMemory' },
                    { fn: () => this.translateWithLibreTranslate(text, sourceLanguage), name: 'LibreTranslate' },
                    { fn: () => this.translateWithFallback(text, sourceLanguage), name: 'Fallback' }
                ];
                
                for (const service of services) {
                    try {
                        const result = await service.fn();
                        if (result && result.trim() && !result.includes('[Translation failed]')) {
                            translatedText = result;
                            serviceUsed = service.name;
                            break;
                        }
                    } catch (error) {
                        console.warn(`${service.name} failed:`, error.message);
                        continue;
                    }
                }
                
                if (!translatedText) {
                    translatedText = `[Translation failed] ${text}`;
                    serviceUsed = 'No service available';
                }
            }
            
            this.elements.translatedText.textContent = translatedText;
            this.addToHistory(text, translatedText);
            
            if (this.elements.autoSpeak.checked && !translatedText.includes('[Translation failed]')) {
                this.speakText(translatedText);
            }
            
            // Show which service was used
            const langInfo = this.currentLanguage === 'auto' && this.lastDetectedLanguage 
                ? ` (detected: ${this.getLanguageName(this.lastDetectedLanguage)})` 
                : '';
            this.updateStatus(`Translated via ${serviceUsed}${langInfo}`, '✅');
            
            setTimeout(() => {
                if (this.isListening) {
                    const mode = this.isModelLoaded ? 'Whisper AI' : 'Web Speech API';
                    this.updateStatus(`Listening with ${mode}...`, '🎤 On');
                }
            }, 2000);
            
        } catch (error) {
            console.error('Translation error:', error);
            
            const detectedLang = this.currentLanguage === 'auto' && this.lastDetectedLanguage 
                ? this.lastDetectedLanguage 
                : this.currentLanguage;
            const fallbackText = `[Original: ${this.getLanguageName(detectedLang)}] ${text}`;
            this.elements.translatedText.textContent = fallbackText;
            this.addToHistory(text, fallbackText);
            
            this.updateStatus('Translation failed', '⚠️');
        }
    }
    
    async translateWithGoogleAPI(text, sourceLanguage = null) {
        const url = `https://translation.googleapis.com/language/translate/v2?key=${this.apiKey}`;
        
        const response = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                q: text,
                target: this.targetLanguage,
                source: sourceLanguage === 'auto' ? undefined : sourceLanguage
            })
        });
        
        if (!response.ok) {
            throw new Error(`Google Translate API error: ${response.status}`);
        }
        
        const data = await response.json();
        return data.data.translations[0].translatedText;
    }
    
    async translateWithMyMemory(text, sourceLanguage = null) {
        const sourceLang = sourceLanguage || (this.currentLanguage === 'auto' ? 'en' : this.currentLanguage);
        const limitedText = text.length > 500 ? text.substring(0, 500) + '...' : text;
        
        const url = `https://api.mymemory.translated.net/get?q=${encodeURIComponent(limitedText)}&langpair=${sourceLang}|${this.targetLanguage}`;
        
        const response = await fetch(url);
        
        if (!response.ok) {
            throw new Error(`MyMemory error: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.responseStatus === 200 && data.responseData?.translatedText) {
            const translation = data.responseData.translatedText;
            if (translation.toLowerCase() !== limitedText.toLowerCase()) {
                return translation;
            }
        }
        
        throw new Error('MyMemory: No valid translation');
    }

    async translateWithLibreTranslate(text, sourceLanguage = null) {
        const url = 'https://libretranslate.de/translate';
        const sourceLang = sourceLanguage || (this.currentLanguage === 'auto' ? 'auto' : this.currentLanguage);
        
        const response = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                q: text,
                source: sourceLang,
                target: this.targetLanguage,
                format: 'text'
            })
        });
        
        if (!response.ok) {
            throw new Error(`LibreTranslate error: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.translatedText) {
            return data.translatedText;
        }
        
        throw new Error('LibreTranslate: No translation in response');
    }

    async translateWithFallback(text, sourceLanguage = null) {
        const sourceLang = sourceLanguage || this.currentLanguage;
        
        if (sourceLang === this.targetLanguage) {
            return text;
        }
        
        return `[Original: ${this.getLanguageName(sourceLang)}] ${text}`;
    }
    
    getLanguageName(code) {
        const names = {
            'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
            'it': 'Italian', 'pt': 'Portuguese', 'ru': 'Russian', 'ja': 'Japanese',
            'ko': 'Korean', 'zh': 'Chinese', 'ar': 'Arabic', 'hi': 'Hindi',
            'auto': 'Auto-detected'
        };
        return names[code] || code;
    }
    
    speakText(text) {
        if ('speechSynthesis' in window) {
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.lang = this.getProperLanguageCode(this.targetLanguage);
            utterance.rate = 0.9;
            utterance.volume = 0.8;
            speechSynthesis.speak(utterance);
        }
    }
    
    addToHistory(original, translation) {
        const historyItem = {
            original,
            translation,
            timestamp: new Date().toLocaleTimeString(),
            date: new Date().toLocaleDateString()
        };
        
        this.history.unshift(historyItem);
        
        if (this.history.length > 50) {
            this.history = this.history.slice(0, 50);
        }
        
        localStorage.setItem('translationHistory', JSON.stringify(this.history));
        this.displayHistory();
    }
    
    displayHistory() {
        this.elements.historyContainer.innerHTML = '';
        
        this.history.forEach(item => {
            const historyElement = document.createElement('div');
            historyElement.className = 'history-item';
            historyElement.innerHTML = `
                <div class="original">${item.original}</div>
                <div class="translation">${item.translation}</div>
                <div class="timestamp">${item.date} ${item.timestamp}</div>
            `;
            this.elements.historyContainer.appendChild(historyElement);
        });
    }
    
    clearHistory() {
        this.history = [];
        localStorage.removeItem('translationHistory');
        this.displayHistory();
    }
    
    saveApiKey() {
        this.apiKey = this.elements.apiKey.value.trim();
        if (this.apiKey) {
            localStorage.setItem('googleTranslateApiKey', this.apiKey);
            alert('API key saved successfully!');
        } else {
            localStorage.removeItem('googleTranslateApiKey');
            alert('API key removed!');
        }
    }
    
    saveSettings() {
        const settings = {
            fontSize: this.elements.fontSize.value,
            bgOpacity: this.elements.bgOpacity.value,
            autoSpeak: this.elements.autoSpeak.checked,
            overlayMode: this.elements.overlayMode.checked,
            inputLanguage: this.currentLanguage,
            outputLanguage: this.targetLanguage
        };
        
        localStorage.setItem('ttsTranslatorSettings', JSON.stringify(settings));
    }
    
    loadSettings() {
        const saved = localStorage.getItem('ttsTranslatorSettings');
        if (saved) {
            try {
                const settings = JSON.parse(saved);
                
                if (settings.fontSize) {
                    this.elements.fontSize.value = settings.fontSize;
                    this.elements.fontSizeValue.textContent = `${settings.fontSize}px`;
                    document.documentElement.style.setProperty('--font-size', `${settings.fontSize}px`);
                }
                
                if (settings.bgOpacity) {
                    this.elements.bgOpacity.value = settings.bgOpacity;
                    this.elements.bgOpacityValue.textContent = `${settings.bgOpacity}%`;
                    document.documentElement.style.setProperty('--bg-opacity', settings.bgOpacity / 100);
                }
                
                if (typeof settings.autoSpeak === 'boolean') {
                    this.elements.autoSpeak.checked = settings.autoSpeak;
                }
                
                if (typeof settings.overlayMode === 'boolean') {
                    this.elements.overlayMode.checked = settings.overlayMode;
                    if (settings.overlayMode) {
                        document.body.classList.add('overlay-mode');
                        setTimeout(() => this.minimizeForOverlay(), 100);
                    }
                }
                
                if (settings.inputLanguage) {
                    this.currentLanguage = settings.inputLanguage;
                    this.elements.inputLanguage.value = settings.inputLanguage;
                }
                
                if (settings.outputLanguage) {
                    this.targetLanguage = settings.outputLanguage;
                    this.elements.outputLanguage.value = settings.outputLanguage;
                }
                
            } catch (error) {
                console.error('Failed to load settings:', error);
            }
        }
        
        // Load API key
        const savedApiKey = localStorage.getItem('googleTranslateApiKey');
        if (savedApiKey) {
            this.apiKey = savedApiKey;
            this.elements.apiKey.value = savedApiKey;
        }
    }
    
    updateStatus(status, micStatus) {
        this.elements.statusText.textContent = status;
        this.elements.micStatus.textContent = micStatus;
    }

    getProperLanguageCode(language) {
        const languageMap = {
            'auto': 'en-US',
            'en': 'en-US',
            'es': 'es-ES',
            'fr': 'fr-FR',
            'de': 'de-DE',
            'it': 'it-IT',
            'pt': 'pt-BR',
            'ru': 'ru-RU',
            'ja': 'ja-JP',
            'ko': 'ko-KR',
            'zh': 'zh-CN',
            'ar': 'ar-SA',
            'hi': 'hi-IN'
        };
        
        return languageMap[language] || language;
    }
}

// Initialize the translator when the page loads
document.addEventListener('DOMContentLoaded', () => {
    const translator = new LiveTranslator();
    console.log('Live Translator with Whisper AI initialized for streaming.');
});

// Service Worker registration for PWA capabilities
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('./sw.js')
            .then(registration => {
                console.log('SW registered: ', registration);
            })
            .catch(registrationError => {
                console.log('SW registration failed: ', registrationError);
            });
    });
} 