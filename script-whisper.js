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
        this.whisperEnabled = false;
        this.audioBuffer = null;
        this.recordingInterval = null;
        
        this.initializeElements();
        this.setupLoadingScreen();
        this.initializeEventListeners();
        this.loadSettings();
        this.displayHistory();
        
        // Check for overlay mode
        if (window.location.hash === '#overlay') {
            document.body.classList.add('overlay-mode');
        }
        
        // Auto-load the model on startup with better error handling
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
            this.updateLoadingProgress(0, 'Checking WebGPU compatibility...');
            
            console.log('Starting Whisper model loading...');
            
            // Check if we're on HTTPS (required for WebGPU)
            if (location.protocol !== 'https:' && location.hostname !== 'localhost') {
                console.warn('HTTPS required for WebGPU - falling back to Web Speech API');
                throw new Error('HTTPS required for Whisper WebGPU support');
            }
            
            // Check WebGPU support with proper error handling
            let webgpuSupported = false;
            try {
                if (navigator.gpu) {
                    const adapter = await navigator.gpu.requestAdapter();
                    webgpuSupported = !!adapter;
                }
            } catch (error) {
                console.warn('WebGPU check failed:', error);
            }
            
            this.updateLoadingProgress(10, webgpuSupported ? 'WebGPU detected - loading 200MB Whisper model...' : 'Using CPU mode - loading 200MB model...');
            
            // Import Transformers.js with timeout and error handling
            console.log('Importing Transformers.js...');
            
            const importPromise = import('https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2');
            const timeout = new Promise((_, reject) => 
                setTimeout(() => reject(new Error('Import timeout')), 15000)
            );
            
            const { pipeline, env } = await Promise.race([importPromise, timeout]);
            console.log('Transformers.js imported successfully');
            
            // Configure environment with better settings
            env.allowRemoteModels = true;
            env.allowLocalModels = false; // Disable local models to avoid CORS issues
            env.useFSCache = false; // Disable filesystem cache for GitHub Pages
            
            if (webgpuSupported) {
                console.log('Configuring WebGPU backend...');
                env.backends.onnx.wasm.proxy = false;
            } else {
                console.log('Configuring optimized WASM backend...');
                env.backends.onnx.wasm.numThreads = Math.min(navigator.hardwareConcurrency || 4, 4);
                env.backends.onnx.wasm.simd = true;
                env.backends.onnx.wasm.proxy = false;
            }
            
            this.updateLoadingProgress(20, 'Loading Whisper-Base model (~200MB)...');
            console.log('Starting pipeline creation with whisper-base...');
            
            // Create pipeline with proper error handling and timeout - using whisper-base (200MB)
            const modelPromise = pipeline('automatic-speech-recognition', 'Xenova/whisper-base', {
                dtype: webgpuSupported ? {
                    encoder_model: 'fp16',
                    decoder_model_merged: 'q4',
                } : 'fp32',
                device: webgpuSupported ? 'webgpu' : 'wasm',
                progress_callback: (progress) => {
                    console.log('Model loading progress:', progress);
                    if (progress.status === 'downloading') {
                        const percent = 20 + Math.round((progress.loaded / progress.total) * 60);
                        this.updateLoadingProgress(percent, `Downloading: ${Math.round((progress.loaded / progress.total) * 100)}% (~200MB)`);
                    } else if (progress.status === 'loading') {
                        this.updateLoadingProgress(85, 'Loading model into memory...');
                    } else if (progress.status === 'ready') {
                        this.updateLoadingProgress(95, 'Model ready - finalizing...');
                    }
                }
            });
            
            // Extended timeout for larger model
            const modelTimeout = new Promise((_, reject) => {
                setTimeout(() => reject(new Error('Model loading timeout')), 45000);
            });
            
            this.pipeline = await Promise.race([modelPromise, modelTimeout]);
            console.log('Whisper-base pipeline created successfully');
            
            this.whisperEnabled = true;
            this.isModelLoaded = true;
            this.updateLoadingProgress(100, 'Whisper AI (200MB model) loaded successfully!');
            
            // Wait a moment then show main app and auto-start listening
            setTimeout(() => {
                this.showMainApp();
                
                // Auto-start listening for streamers
                setTimeout(() => {
                    this.startListening();
                }, 500);
            }, 1000);
            
            console.log('Whisper-base pipeline loaded successfully');
            
        } catch (error) {
            console.error('Failed to load Whisper pipeline:', error);
            this.whisperEnabled = false;
            this.updateLoadingProgress(0, 'Whisper unavailable - using Web Speech API...');
            
            // Wait then show main app with fallback
            setTimeout(() => {
                this.initializeFallbackSpeechRecognition();
                this.showMainApp();
                
                // Auto-start listening even in fallback mode
                setTimeout(() => {
                    this.startListening();
                }, 500);
            }, 1500);
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
            this.updateStatus('Whisper AI ready - Starting continuous listening...', '⏳');
        } else {
            this.updateStatus('Web Speech API ready - Starting continuous listening...', '⏳');
        }
    }
    
    async startListening() {
        try {
            // Request microphone access with better error handling
            this.audioStream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    sampleRate: 16000,
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                } 
            });
            
            this.isListening = true;
            this.updateStatus('Starting continuous listening...', '🔄');
            
            if (this.whisperEnabled && this.isModelLoaded) {
                this.startWhisperListening();
            } else {
                this.startWebSpeechListening();
            }
            
        } catch (error) {
            console.error('Failed to access microphone:', error);
            this.updateStatus('Microphone access denied. Please allow microphone access and refresh.', '❌');
            
            // Try fallback without microphone constraints
            try {
                this.audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
                this.isListening = true;
                this.startWebSpeechListening();
            } catch (fallbackError) {
                console.error('Fallback microphone access failed:', fallbackError);
                this.updateStatus('Unable to access microphone', '❌');
            }
        }
    }
    
    startWhisperListening() {
        try {
            this.updateStatus('Continuous listening with Whisper AI (8-sec segments)...', '🎤 On');
            document.body.classList.add('listening');
            this.elements.startBtn.disabled = true;
            this.elements.stopBtn.disabled = false;
            
            // Clear previous audio chunks
            this.audioChunks = [];
            
            // Setup MediaRecorder with proper error handling
            const options = {
                mimeType: 'audio/webm;codecs=opus',
                audioBitsPerSecond: 16000
            };
            
            // Fallback MIME types if webm is not supported
            if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                if (MediaRecorder.isTypeSupported('audio/webm')) {
                    options.mimeType = 'audio/webm';
                } else if (MediaRecorder.isTypeSupported('audio/mp4')) {
                    options.mimeType = 'audio/mp4';
                } else {
                    delete options.mimeType;
                }
            }
            
            this.mediaRecorder = new MediaRecorder(this.audioStream, options);
            
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.audioChunks.push(event.data);
                }
            };
            
            this.mediaRecorder.onstop = () => {
                if (this.audioChunks.length > 0) {
                    this.processAudioWithWhisper();
                }
            };
            
            this.mediaRecorder.onerror = (error) => {
                console.error('MediaRecorder error:', error);
                this.updateStatus('Recording error - switching to Web Speech API', '⚠️');
                this.startWebSpeechListening();
            };
            
            // Start continuous recording in segments
            this.startRecordingSegments();
            
        } catch (error) {
            console.error('Failed to start Whisper listening:', error);
            this.updateStatus('Whisper error - using Web Speech API', '⚠️');
            this.startWebSpeechListening();
        }
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
        
        // Clear any existing interval
        if (this.recordingInterval) {
            clearTimeout(this.recordingInterval);
        }
        
        // Use longer 8-second segments for better transcription accuracy
        this.mediaRecorder.start();
        console.log('Started recording segment...');
        
        this.recordingInterval = setTimeout(() => {
            if (this.isListening && this.mediaRecorder.state === 'recording') {
                this.mediaRecorder.stop();
                console.log('Stopped recording segment for processing...');
                
                // Immediately start next segment without waiting for processing to complete
                // This ensures continuous listening
                setTimeout(() => {
                    if (this.isListening) {
                        this.startRecordingSegments(); // Start next segment immediately
                    }
                }, 100); // Very short gap to allow MediaRecorder to reset
            }
        }, 8000); // 8-second segments for better accuracy
    }
    
    async processAudioWithWhisper() {
        // Process in parallel - don't block listening
        if (this.isProcessing) {
            console.log('Already processing audio, skipping...');
            return;
        }
        
        if (this.audioChunks.length === 0 || !this.isModelLoaded) return;
        
        this.isProcessing = true;
        
        // Copy chunks for processing and clear immediately to continue recording
        const chunksToProcess = [...this.audioChunks];
        this.audioChunks = [];
        
        try {
            this.updateStatus('Transcribing with Whisper AI...', '🔄');
            
            // Convert audio chunks to blob
            const audioBlob = new Blob(chunksToProcess, { 
                type: this.mediaRecorder.mimeType || 'audio/webm' 
            });
            
            // Convert blob to audio buffer with proper error handling
            const arrayBuffer = await audioBlob.arrayBuffer();
            
            if (arrayBuffer.byteLength === 0) {
                console.warn('Empty audio buffer received');
                this.isProcessing = false;
                return;
            }
            
            // Create audio context with proper sample rate
            const audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: 16000
            });
            
            let audioBuffer;
            try {
                audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
            } catch (decodeError) {
                console.error('Audio decode error:', decodeError);
                this.updateStatus('Listening with Whisper AI...', '🎤 On'); // Continue listening
                this.isProcessing = false;
                return;
            }
            
            // Get audio data as Float32Array
            let audio = audioBuffer.getChannelData(0);
            
            if (audio.length === 0) {
                console.warn('Empty audio data');
                this.isProcessing = false;
                return;
            }
            
            // Resample to 16kHz if needed (Whisper requirement)
            if (audioBuffer.sampleRate !== 16000) {
                const targetLength = Math.round(audio.length * 16000 / audioBuffer.sampleRate);
                const resampledAudio = new Float32Array(targetLength);
                
                for (let i = 0; i < targetLength; i++) {
                    const sourceIndex = Math.round(i * audioBuffer.sampleRate / 16000);
                    resampledAudio[i] = audio[Math.min(sourceIndex, audio.length - 1)];
                }
                
                audio = resampledAudio;
            }
            
            // Ensure minimum audio length (at least 1 second)
            if (audio.length < 16000) {
                console.warn('Audio too short for reliable transcription');
                this.isProcessing = false;
                return;
            }
            
            // Limit audio length (30 seconds max for performance)
            const maxLength = 16000 * 30;
            if (audio.length > maxLength) {
                audio = audio.slice(0, maxLength);
            }
            
            // Enhanced audio quality checks
            const rms = Math.sqrt(audio.reduce((sum, val) => sum + val * val, 0) / audio.length);
            if (rms < 0.002) { // Slightly more lenient threshold
                console.warn('Audio appears to be very quiet, but processing anyway...');
                // Don't return - still try to process quiet audio
            }
            
            // Check for digital noise/artifacts (very high RMS indicates corrupted audio)
            if (rms > 0.5) {
                console.warn('Audio appears to be corrupted or too loud');
                this.isProcessing = false;
                return;
            }
            
            // Transcribe with Whisper
            console.log(`Processing ${audio.length} samples (${audio.length / 16000}s), RMS: ${rms.toFixed(4)}`);
            
            const result = await this.pipeline(audio, {
                language: this.currentLanguage === 'auto' ? null : this.currentLanguage,
                task: 'transcribe',
                return_timestamps: false,
                chunk_length_s: 30,
                stride_length_s: 5,
                temperature: 0.0, // Use deterministic decoding
                condition_on_previous_text: false, // Prevent random outputs
                no_timestamps: true, // Disable timestamps to reduce artifacts
            });
            
            // Handle the result with enhanced validation
            if (result && result.text) {
                let transcription = result.text.trim();
                
                console.log('Raw Whisper transcription:', transcription);
                
                // Enhanced artifact filtering
                if (!this.isValidTranscription(transcription)) {
                    console.log('Filtered out transcription:', transcription);
                    this.isProcessing = false;
                    return;
                }
                
                console.log('Valid transcription accepted:', transcription);
                
                // Update detected language if available
                if (result.language) {
                    this.lastDetectedLanguage = result.language;
                    console.log(`Whisper detected language: ${result.language}`);
                }
                
                // Display transcription
                this.elements.originalText.textContent = transcription;
                console.log('Transcription displayed in UI');
                
                // Translate if needed
                if (this.targetLanguage !== this.currentLanguage && this.targetLanguage !== 'auto') {
                    console.log('Starting translation...');
                    this.translateText(transcription);
                } else {
                    console.log('No translation needed, displaying original');
                    this.elements.translatedText.textContent = transcription;
                }
                
                console.log('Processing complete for:', transcription);
            } else {
                console.log('No text in Whisper result:', result);
            }
            
            // Always return to listening state
            this.updateStatus('Listening with Whisper AI...', '🎤 On');
            
        } catch (error) {
            console.error('Audio processing error:', error);
            
            // Don't stop listening on processing errors - continue
            this.updateStatus('Listening with Whisper AI...', '🎤 On');
        } finally {
            this.isProcessing = false;
        }
    }
    
    // Enhanced transcription validation - made much less aggressive for international languages
    isValidTranscription(text) {
        if (!text || text.length < 1) {
            console.log('Validation failed: Empty text');
            return false;
        }
        
        console.log('Validating transcription:', JSON.stringify(text), 'Length:', text.length);
        
        // Only filter out obvious technical artifacts - be very specific
        const technicalArtifacts = [
            '[music]', '[applause]', '[noise]', '[background music]', '[laughter]', '[blank_audio]',
            'thanks for watching', 'thanks for listening', 'subscribe', 'like and subscribe'
        ];
        
        // Check for exact matches with technical artifacts only
        const lowerText = text.toLowerCase().trim();
        for (const artifact of technicalArtifacts) {
            if (lowerText === artifact || lowerText.includes(artifact)) {
                console.log('Validation failed: Contains technical artifact:', artifact);
                return false;
            }
        }
        
        // Only filter EXTREMELY obvious repetitive garbage - single character repeated many times
        if (/^(.)\1{15,}$/.test(text.trim())) {
            console.log('Validation failed: Single character repeated 15+ times');
            return false;
        }
        
        // Filter only completely empty or whitespace-only text
        if (text.trim().length === 0) {
            console.log('Validation failed: Only whitespace');
            return false;
        }
        
        // Accept ALL other text - international languages, short phrases, everything
        console.log('Validation PASSED for:', JSON.stringify(text));
        return true;
    }
    
    stopListening() {
        this.isListening = false;
        
        // Clear recording interval
        if (this.recordingInterval) {
            clearTimeout(this.recordingInterval);
            this.recordingInterval = null;
        }
        
        // Stop media recorder
        if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
            try {
                this.mediaRecorder.stop();
            } catch (error) {
                console.warn('Error stopping media recorder:', error);
            }
        }
        
        // Stop speech recognition
        if (this.recognition) {
            try {
                this.recognition.stop();
            } catch (error) {
                console.warn('Error stopping speech recognition:', error);
            }
        }
        
        // Stop audio stream
        if (this.audioStream) {
            this.audioStream.getTracks().forEach(track => {
                try {
                    track.stop();
                } catch (error) {
                    console.warn('Error stopping audio track:', error);
                }
            });
            this.audioStream = null;
        }
        
        // Clear any pending audio chunks
        this.audioChunks = [];
        
        // Reset processing flag
        this.isProcessing = false;
        
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
            
            // Skip translation if same language
            if (sourceLanguage === this.targetLanguage) {
                this.elements.translatedText.textContent = text;
                this.addToHistory(text, text);
                this.updateStatus('Same language - no translation needed', '✅');
                if (this.elements.autoSpeak.checked) {
                    this.speakText(text);
                }
                return;
            }
            
            if (this.apiKey) {
                try {
                    translatedText = await this.translateWithGoogleAPI(text, sourceLanguage);
                    serviceUsed = 'Google Translate API';
                } catch (error) {
                    console.warn('Google API failed:', error.message);
                    translatedText = await this.translateWithFallback(text, sourceLanguage);
                    serviceUsed = 'Fallback';
                }
            } else {
                // Try MyMemory with better language support
                try {
                    const result = await this.translateWithMyMemory(text, sourceLanguage);
                    if (result && result.trim() && 
                        !result.includes('[Translation failed]') && 
                        result.toLowerCase() !== text.toLowerCase()) {
                        translatedText = result;
                        serviceUsed = 'MyMemory';
                    } else {
                        throw new Error('No valid translation from MyMemory');
                    }
                } catch (error) {
                    console.warn('MyMemory failed:', error.message);
                    // Use simple translation service for common languages
                    try {
                        translatedText = await this.translateWithSimpleAPI(text, sourceLanguage);
                        serviceUsed = 'Simple Translation';
                    } catch (simpleError) {
                        console.warn('Simple translation failed:', simpleError.message);
                        translatedText = await this.translateWithFallback(text, sourceLanguage);
                        serviceUsed = 'Fallback Display';
                    }
                }
            }
            
            // Display translation
            this.elements.translatedText.textContent = translatedText;
            this.addToHistory(text, translatedText);
            
            // Speak only the translated text (not the language indicator)
            if (this.elements.autoSpeak.checked && serviceUsed !== 'Fallback Display') {
                const textToSpeak = translatedText.includes('[') && translatedText.includes(']') 
                    ? text // If it's a fallback with brackets, speak original
                    : translatedText; // Otherwise speak the translation
                this.speakText(textToSpeak);
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
            const fallbackText = await this.translateWithFallback(text, detectedLang);
            this.elements.translatedText.textContent = fallbackText;
            this.addToHistory(text, fallbackText);
            
            this.updateStatus('Translation failed - showing original', '⚠️');
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
        // Map language codes to MyMemory format
        const langMap = {
            'ja': 'ja',
            'ko': 'ko', 
            'zh': 'zh',
            'en': 'en',
            'es': 'es',
            'fr': 'fr',
            'de': 'de',
            'it': 'it',
            'pt': 'pt',
            'ru': 'ru',
            'ar': 'ar',
            'hi': 'hi'
        };
        
        const sourceLang = langMap[sourceLanguage] || (sourceLanguage || 'en');
        const targetLang = langMap[this.targetLanguage] || this.targetLanguage;
        const limitedText = text.length > 200 ? text.substring(0, 200) + '...' : text;
        
        // Use GET request to avoid CORS preflight issues
        const params = new URLSearchParams({
            q: limitedText,
            langpair: `${sourceLang}|${targetLang}`,
            de: 'translator@example.com', // Required email format
            mt: '1'
        });
        
        const url = `https://api.mymemory.translated.net/get?${params.toString()}`;
        
        const response = await fetch(url, {
            method: 'GET',
            headers: {
                'Accept': 'application/json',
                'User-Agent': 'LiveTTSTranslator/1.0'
            }
        });
        
        if (!response.ok) {
            throw new Error(`MyMemory error: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('MyMemory response:', data);
        
        if (data.responseStatus === 200 && data.responseData?.translatedText) {
            let translation = data.responseData.translatedText;
            
            // Clean up the translation
            translation = translation.replace(/MYMEMORY WARNING:.*$/gi, '').trim();
            translation = translation.replace(/\s*\[.*?\]\s*$/g, '').trim();
            
            // Check if translation is valid (not just the same text)
            if (translation && 
                translation.toLowerCase() !== limitedText.toLowerCase() && 
                !translation.includes('TRANSLATED BY') &&
                translation.length > 0) {
                return translation;
            }
        }
        
        throw new Error('MyMemory: No valid translation found');
    }

    // Simple translation for common phrases
    async translateWithSimpleAPI(text, sourceLanguage = null) {
        const sourceLang = sourceLanguage || 'auto';
        
        // Common Japanese phrases with English translations
        const commonTranslations = {
            'ja': {
                '本当にですか': 'Really?',
                '本当に': 'Really',
                'ですか': 'Is it?',
                'はい': 'Yes',
                'いいえ': 'No',
                'こんにちは': 'Hello',
                'ありがとう': 'Thank you',
                'ありがとうございます': 'Thank you very much',
                'すみません': 'Excuse me',
                'おはよう': 'Good morning',
                'こんばんは': 'Good evening',
                'さようなら': 'Goodbye',
                'お疲れ様': 'Good job',
                'がんばって': 'Good luck',
                'どうですか': 'How is it?',
                'そうですね': 'I agree',
                'わかりました': 'I understand',
                'わからない': 'I don\'t understand'
            },
            'ko': {
                '안녕하세요': 'Hello',
                '감사합니다': 'Thank you',
                '네': 'Yes',
                '아니요': 'No',
                '죄송합니다': 'Sorry',
                '잘 부탁드립니다': 'Please take care of me'
            }
        };
        
        const translations = commonTranslations[sourceLang];
        if (translations && translations[text]) {
            return translations[text];
        }
        
        // If not found, try a different API approach
        throw new Error('Simple translation not available for this text');
    }

    async translateWithFallback(text, sourceLanguage = null) {
        const sourceLang = sourceLanguage || this.currentLanguage;
        
        // If same language, return original
        if (sourceLang === this.targetLanguage) {
            return text;
        }
        
        // For fallback, just show the original text clearly labeled
        const sourceName = this.getLanguageName(sourceLang);
        return `${text}`;
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
            'pt': 'pt-PT',
            'ru': 'ru-RU',
            'ja': 'ja-JP',
            'ko': 'ko-KR',
            'zh': 'zh-CN',
            'ar': 'ar-SA',
            'hi': 'hi-IN'
        };
        
        return languageMap[language] || 'en-US';
    }
}

// Initialize the application when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    try {
        console.log('Initializing Live TTS Translator...');
        window.translator = new LiveTranslator();
        console.log('Live TTS Translator initialized successfully');
    } catch (error) {
        console.error('Failed to initialize Live TTS Translator:', error);
        
        // Show error message to user
        const statusElement = document.getElementById('statusText');
        if (statusElement) {
            statusElement.textContent = 'Initialization failed - please refresh page';
        }
        
        // Try to show error in overlay if available
        if (typeof showError === 'function') {
            showError('Failed to initialize application: ' + error.message);
        }
    }
});

// Handle page visibility changes to restart listening if needed
document.addEventListener('visibilitychange', () => {
    if (window.translator && !document.hidden && window.translator.isListening) {
        // Restart listening if page becomes visible and was previously listening
        setTimeout(() => {
            if (window.translator && window.translator.isListening) {
                console.log('Page became visible - ensuring listening is active');
            }
        }, 1000);
    }
});

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = LiveTranslator;
}

// Also make available globally
window.LiveTranslator = LiveTranslator; 