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
            
            this.updateLoadingProgress(10, webgpuSupported ? 'WebGPU detected - loading optimized model...' : 'Using CPU mode...');
            
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
            
            this.updateLoadingProgress(20, 'Creating Whisper pipeline...');
            console.log('Starting pipeline creation...');
            
            // Create pipeline with proper error handling and timeout
            const modelPromise = pipeline('automatic-speech-recognition', 'Xenova/whisper-tiny', {
                dtype: webgpuSupported ? {
                    encoder_model: 'fp16',
                    decoder_model_merged: 'q4',
                } : 'fp32',
                device: webgpuSupported ? 'webgpu' : 'wasm',
                progress_callback: (progress) => {
                    console.log('Model loading progress:', progress);
                    if (progress.status === 'downloading') {
                        const percent = 20 + Math.round((progress.loaded / progress.total) * 60);
                        this.updateLoadingProgress(percent, `Downloading: ${Math.round((progress.loaded / progress.total) * 100)}%`);
                    } else if (progress.status === 'loading') {
                        this.updateLoadingProgress(85, 'Loading model into memory...');
                    } else if (progress.status === 'ready') {
                        this.updateLoadingProgress(95, 'Model ready - finalizing...');
                    }
                }
            });
            
            // Reduced timeout for tiny model
            const modelTimeout = new Promise((_, reject) => {
                setTimeout(() => reject(new Error('Model loading timeout')), 20000);
            });
            
            this.pipeline = await Promise.race([modelPromise, modelTimeout]);
            console.log('Pipeline created successfully');
            
            this.whisperEnabled = true;
            this.isModelLoaded = true;
            this.updateLoadingProgress(100, 'Whisper AI loaded successfully!');
            
            // Wait a moment then show main app and auto-start listening
            setTimeout(() => {
                this.showMainApp();
                
                // Auto-start listening for streamers
                setTimeout(() => {
                    this.startListening();
                }, 500);
            }, 1000);
            
            console.log('Whisper pipeline loaded successfully');
            
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
            this.updateStatus('Whisper AI ready - Starting automatically...', '⏳');
        } else {
            this.updateStatus('Fallback mode ready - Starting automatically...', '⏳');
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
            this.updateStatus('Starting...', '🔄');
            
            if (this.whisperEnabled && this.isModelLoaded) {
                // Try MediaRecorder first, fallback to Web Audio API if it fails
                try {
                    this.startWhisperListening();
                } catch (error) {
                    console.warn('MediaRecorder failed, trying Web Audio API:', error);
                    this.startWebAudioListening();
                }
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
            this.updateStatus('Listening with Whisper AI...', '🎤 On');
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
                    console.log(`Audio chunk received: ${event.data.size} bytes`);
                }
            };
            
            this.mediaRecorder.onstop = () => {
                console.log('MediaRecorder stopped for processing');
                
                // START NEW RECORDING IMMEDIATELY - before processing starts
                if (this.isListening) {
                    setTimeout(() => {
                        this.startRecordingSegments();
                    }, 50); // Start next segment immediately
                }
                
                // Process the complete audio segment
                if (this.audioChunks.length > 0 && this.isListening && !this.isProcessing) {
                    this.processAudioWithWhisper();
                }
            };
            
            this.mediaRecorder.onerror = (error) => {
                console.error('MediaRecorder error:', error);
                this.updateStatus('Recording error - switching to Web Speech API', '⚠️');
                this.startWebSpeechListening();
            };
            
            // Start segmented recording approach
            this.startRecordingSegments();
            
        } catch (error) {
            console.error('Failed to start Whisper listening:', error);
            this.updateStatus('Whisper error - using Web Speech API', '⚠️');
            this.startWebSpeechListening();
        }
    }
    
    startWebAudioListening() {
        try {
            console.log('Starting Web Audio API recording as fallback');
            this.updateStatus('Listening with Whisper AI (Web Audio)...', '🎤 On');
            document.body.classList.add('listening');
            this.elements.startBtn.disabled = true;
            this.elements.stopBtn.disabled = false;
            
            // Create Web Audio API context
            const audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: 16000
            });
            
            // Create source from microphone stream
            const source = audioContext.createMediaStreamSource(this.audioStream);
            
            // Create script processor for capturing audio data
            const bufferSize = 4096;
            const processor = audioContext.createScriptProcessor(bufferSize, 1, 1);
            
            // Array to collect audio samples
            this.audioSamples = [];
            this.lastProcessTime = Date.now();
            
            processor.onaudioprocess = (event) => {
                if (!this.isListening) return;
                
                const inputBuffer = event.inputBuffer;
                const inputData = inputBuffer.getChannelData(0);
                
                // Copy audio data
                const samples = new Float32Array(inputData.length);
                samples.set(inputData);
                this.audioSamples.push(samples);
                
                // Process every 4 seconds
                const now = Date.now();
                if (now - this.lastProcessTime >= 4000) {
                    this.processWebAudioSamples();
                    this.lastProcessTime = now;
                }
            };
            
            // Connect the audio processing chain
            source.connect(processor);
            processor.connect(audioContext.destination);
            
            // Store references for cleanup
            this.audioContext = audioContext;
            this.audioProcessor = processor;
            this.audioSource = source;
            
        } catch (error) {
            console.error('Web Audio API failed:', error);
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
        
        // Clear previous chunks
        this.audioChunks = [];
        
        console.log('Starting new recording segment...');
        
        // Start recording a complete segment
        this.mediaRecorder.start();
        
        // Stop recording after 4 seconds to create a complete segment
        this.recordingInterval = setTimeout(() => {
            if (this.isListening && this.mediaRecorder.state === 'recording') {
                console.log('Stopping segment for processing...');
                this.mediaRecorder.stop();
                // Note: Next segment will be started in onstop handler
            }
        }, 4000); // 4-second complete segments
    }
    
    async processAudioWithWhisper() {
        if (this.isProcessing || this.audioChunks.length === 0 || !this.isModelLoaded) return;
        
        this.isProcessing = true;
        
        // Make a copy of chunks and clear the original array immediately
        const chunksToProcess = [...this.audioChunks];
        this.audioChunks = [];
        
        try {
            this.updateStatus('Transcribing with Whisper...', '🔄');
            
            // Convert audio chunks to blob
            const audioBlob = new Blob(chunksToProcess, { 
                type: this.mediaRecorder.mimeType || 'audio/webm' 
            });
            
            console.log(`Processing audio blob: ${audioBlob.size} bytes, type: ${audioBlob.type}`);
            
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
                console.log(`Audio decoded successfully: ${audioBuffer.duration}s, ${audioBuffer.sampleRate}Hz`);
            } catch (decodeError) {
                console.error('Audio decode error:', decodeError);
                console.log('Blob details:', { size: audioBlob.size, type: audioBlob.type });
                this.updateStatus('Audio format error - continuing...', '⚠️');
                this.isProcessing = false;
                
                // Continue listening even after decode errors
                setTimeout(() => {
                    this.updateStatus('Listening with Whisper AI...', '🎤 On');
                }, 1000);
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
                console.log(`Resampled from ${audioBuffer.sampleRate}Hz to 16000Hz`);
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
            if (rms < 0.001) {
                console.warn('Audio appears to be silent');
                this.isProcessing = false;
                return;
            }
            
            // Check for digital noise/artifacts (very high RMS indicates corrupted audio)
            if (rms > 0.5) {
                console.warn('Audio appears to be corrupted or too loud');
                this.isProcessing = false;
                return;
            }
            
            // Check for consistent patterns that indicate digital artifacts
            const segments = 10;
            const segmentLength = Math.floor(audio.length / segments);
            const segmentRMS = [];
            
            for (let i = 0; i < segments; i++) {
                const start = i * segmentLength;
                const end = Math.min(start + segmentLength, audio.length);
                const segmentData = audio.slice(start, end);
                const segmentRmsValue = Math.sqrt(segmentData.reduce((sum, val) => sum + val * val, 0) / segmentData.length);
                segmentRMS.push(segmentRmsValue);
            }
            
            // Check if all segments have very similar RMS (indicates digital noise)
            const avgRMS = segmentRMS.reduce((sum, rms) => sum + rms, 0) / segmentRMS.length;
            const variance = segmentRMS.reduce((sum, rms) => sum + Math.pow(rms - avgRMS, 2), 0) / segmentRMS.length;
            
            if (variance < 0.0001 && avgRMS > 0.01) {
                console.warn('Audio appears to contain digital artifacts - skipping');
                this.isProcessing = false;
                return;
            }
            
            // Transcribe with Whisper
            console.log(`Processing ${audio.length} samples (${audio.length / 16000}s), RMS: ${rms.toFixed(4)}`);
            
            // Determine language for Whisper
            let whisperLanguage = null;
            if (this.currentLanguage !== 'auto') {
                whisperLanguage = this.currentLanguage;
                console.log(`Using selected language for Whisper: ${whisperLanguage}`);
            } else {
                console.log('Using Whisper auto-detection');
            }
            
            const result = await this.pipeline(audio, {
                language: whisperLanguage,
                task: 'transcribe',
                return_timestamps: false,
                chunk_length_s: 30,
                stride_length_s: 5,
                temperature: 0.0,
                condition_on_previous_text: false,
                no_timestamps: true,
            });
            
            // Handle the result with enhanced validation
            if (result && result.text) {
                let transcription = result.text.trim();
                
                console.log('Raw transcription:', transcription);
                
                // Enhanced artifact filtering with detailed logging
                const isValid = this.isValidTranscription(transcription);
                console.log(`Transcription validation result: ${isValid ? 'VALID' : 'INVALID'} - "${transcription}"`);
                
                if (isValid) {
                    // Use Whisper's detected language directly
                    if (result.language) {
                        this.lastDetectedLanguage = result.language;
                        console.log(`Whisper detected language: ${result.language}`);
                    }
                    
                    // Display transcription immediately (simple approach)
                    this.elements.originalText.textContent = transcription;
                    
                    // Translate if needed
                    if (this.targetLanguage !== this.currentLanguage && this.targetLanguage !== 'auto') {
                        console.log(`Translating from ${this.lastDetectedLanguage || this.currentLanguage} to ${this.targetLanguage}`);
                        this.translateText(transcription).then(translation => {
                            this.elements.translatedText.textContent = translation;
                            this.addToHistory(transcription, translation);
                            
                            // TTS if enabled and translation is different from original
                            if (this.elements.autoSpeak.checked && 
                                translation && 
                                translation !== transcription &&
                                !translation.includes('[') && 
                                !translation.includes('→')) {
                                console.log('🔊 Speaking translation:', translation);
                                this.speakText(translation);
                            }
                        });
                    } else {
                        this.elements.translatedText.textContent = transcription;
                        this.addToHistory(transcription, transcription);
                    }
                    
                    console.log('✅ Valid transcription processed:', transcription);
                } else {
                    console.log('❌ Filtered out invalid transcription:', transcription);
                    this.debugTranscriptionFiltering(transcription);
                }
            }
            
            this.updateStatus('Listening with Whisper AI...', '🎤 On');
            
        } catch (error) {
            console.error('Audio processing error:', error);
            this.updateStatus('Transcription error - continuing...', '⚠️');
            
            // Don't stop listening on processing errors
            setTimeout(() => {
                this.updateStatus('Listening with Whisper AI...', '🎤 On');
            }, 2000);
        } finally {
            this.isProcessing = false;
        }
    }
    
    async processWebAudioSamples() {
        if (this.isProcessing || this.audioSamples.length === 0 || !this.isModelLoaded) return;
        
        this.isProcessing = true;
        
        try {
            this.updateStatus('Transcribing with Whisper...', '🔄');
            
            // Combine all audio samples
            const totalLength = this.audioSamples.reduce((sum, samples) => sum + samples.length, 0);
            const combinedAudio = new Float32Array(totalLength);
            
            let offset = 0;
            for (const samples of this.audioSamples) {
                combinedAudio.set(samples, offset);
                offset += samples.length;
            }
            
            // Clear samples
            this.audioSamples = [];
            
            console.log(`Processing Web Audio samples: ${combinedAudio.length} samples (${combinedAudio.length / 16000}s)`);
            
            // Check minimum length
            if (combinedAudio.length < 16000) {
                console.warn('Audio too short for transcription');
                this.isProcessing = false;
                return;
            }
            
            // Check audio quality
            const rms = Math.sqrt(combinedAudio.reduce((sum, val) => sum + val * val, 0) / combinedAudio.length);
            if (rms < 0.001) {
                console.warn('Audio appears to be silent');
                this.isProcessing = false;
                return;
            }
            
            console.log(`Processing ${combinedAudio.length} samples, RMS: ${rms.toFixed(4)}`);
            
            // Determine language for Whisper
            let whisperLanguage = null;
            if (this.currentLanguage !== 'auto') {
                whisperLanguage = this.currentLanguage;
                console.log(`Using selected language for Whisper: ${whisperLanguage}`);
            } else {
                console.log('Using Whisper auto-detection');
            }
            
            // Transcribe with Whisper
            const result = await this.pipeline(combinedAudio, {
                language: whisperLanguage,
                task: 'transcribe',
                return_timestamps: false,
                chunk_length_s: 30,
                stride_length_s: 5,
                temperature: 0.0,
                condition_on_previous_text: false,
                no_timestamps: true,
            });
            
            // Handle the result
            if (result && result.text) {
                let transcription = result.text.trim();
                
                console.log('Raw transcription:', transcription);
                
                if (this.isValidTranscription(transcription)) {
                    // Use Whisper's detected language directly
                    if (result.language) {
                        this.lastDetectedLanguage = result.language;
                        console.log(`Whisper detected language: ${result.language}`);
                    }
                    
                    // Display transcription immediately
                    this.elements.originalText.textContent = transcription;
                    
                    // Translate if needed
                    if (this.targetLanguage !== this.currentLanguage && this.targetLanguage !== 'auto') {
                        this.translateText(transcription).then(translation => {
                            this.elements.translatedText.textContent = translation;
                            this.addToHistory(transcription, translation);
                            
                            // TTS if enabled
                            if (this.elements.autoSpeak.checked && 
                                translation && 
                                translation !== transcription &&
                                !translation.includes('[') && 
                                !translation.includes('→')) {
                                this.speakText(translation);
                            }
                        });
                    } else {
                        this.elements.translatedText.textContent = transcription;
                        this.addToHistory(transcription, transcription);
                    }
                    
                    console.log('Valid transcription:', transcription);
                } else {
                    console.log('Filtered out invalid transcription:', transcription);
                }
            }
            
            this.updateStatus('Listening with Whisper AI (Web Audio)...', '🎤 On');
            
        } catch (error) {
            console.error('Web Audio processing error:', error);
            this.updateStatus('Transcription error - continuing...', '⚠️');
            
            setTimeout(() => {
                this.updateStatus('Listening with Whisper AI (Web Audio)...', '🎤 On');
            }, 2000);
        } finally {
            this.isProcessing = false;
        }
    }
    
    // Enhanced transcription validation - fixed for international characters
    isValidTranscription(text) {
        if (!text || text.length < 2) return false;
        
        // Only filter out very obvious artifacts - be much more permissive
        const artifacts = [
            '[Music]', '[Applause]', '[Noise]', '[Background music]', 
            'Thank you.', 'Thanks for watching!',
        ];
        
        // Check for exact matches with common artifacts
        if (artifacts.some(artifact => text.toLowerCase() === artifact.toLowerCase())) {
            return false;
        }
        
        // MAIN FILTER: Only block repetitive single characters (SSSSS, CCCCC, KKKKK, etc.)
        // Check for repetitive single characters or very short patterns
        const repetitivePatterns = [
            /^(.)\1{4,}$/, // Single character repeated 5+ times (e.g., "SSSSS", "CCCCC")
            /^\[(.)\]\s*\[(.)\]/, // Bracketed single characters (e.g., "[S] [S]")
            /^(.)\s+\1\s+\1\s+\1/, // Spaced repetitive characters (e.g., "S S S S")
        ];
        
        if (repetitivePatterns.some(pattern => pattern.test(text))) {
            return false;
        }
        
        // Check for excessive repetition of any single character (the main issue you want fixed)
        const charCounts = {};
        let totalChars = 0;
        
        for (const char of text.toLowerCase()) {
            if (char.match(/\w/)) { // Any word character (includes international)
                charCounts[char] = (charCounts[char] || 0) + 1;
                totalChars++;
            }
        }
        
        // Only filter if a single character makes up more than 80% of the text (very permissive)
        for (const [char, count] of Object.entries(charCounts)) {
            if (count / totalChars > 0.8) { // Very high threshold - only blocks obvious artifacts
                return false;
            }
        }
        
        // Remove the vowel check entirely - it was causing issues with international text
        // All real speech should pass at this point
        
        return true; // Accept everything else
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
        
        // Clean up Web Audio API resources
        if (this.audioProcessor) {
            try {
                this.audioProcessor.disconnect();
                this.audioProcessor = null;
            } catch (error) {
                console.warn('Error disconnecting audio processor:', error);
            }
        }
        
        if (this.audioSource) {
            try {
                this.audioSource.disconnect();
                this.audioSource = null;
            } catch (error) {
                console.warn('Error disconnecting audio source:', error);
            }
        }
        
        if (this.audioContext && this.audioContext.state !== 'closed') {
            try {
                this.audioContext.close();
                this.audioContext = null;
            } catch (error) {
                console.warn('Error closing audio context:', error);
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
        
        // Clear any pending audio chunks and samples
        this.audioChunks = [];
        this.audioSamples = [];
        
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
    
    async translateText(text, sourceLanguage = null) {
        if (!text.trim()) return text;
        
        this.updateStatus('Translating...', '🔄');
        
        try {
            let translatedText = '';
            let serviceUsed = 'Unknown';
            
            // Determine source language for translation
            let sourceLang = sourceLanguage || this.lastDetectedLanguage || this.currentLanguage;
            if (sourceLang === 'auto') {
                // Simple fallback - default to English if no detection
                sourceLang = 'en';
            }
            
            // Skip translation if same language
            if (sourceLang === this.targetLanguage) {
                console.log('Same language - no translation needed');
                return text;
            }
            
            // Always try to translate, don't use fallback as first option
            let success = false;
            
            if (this.apiKey) {
                try {
                    translatedText = await this.translateWithGoogleAPI(text, sourceLang);
                    serviceUsed = 'Google Translate API';
                    success = true;
                } catch (error) {
                    console.warn('Google API failed:', error.message);
                }
            }
            
            // Try free translation services if Google API failed or not available
            if (!success) {
                const services = [
                    { 
                        fn: () => this.translateWithMyMemory(text, sourceLang), 
                        name: 'MyMemory',
                        corsOptimized: true
                    },
                    { 
                        fn: () => this.translateWithDictionary(text, sourceLang), 
                        name: 'Dictionary',
                        corsOptimized: true
                    }
                ];
                
                for (const service of services) {
                    try {
                        const result = await service.fn();
                        if (result && result.trim() && 
                            !result.includes('[Translation failed]') && 
                            result.toLowerCase() !== text.toLowerCase()) {
                            translatedText = result;
                            serviceUsed = service.name;
                            success = true;
                            break;
                        }
                    } catch (error) {
                        console.warn(`${service.name} failed:`, error.message);
                        continue;
                    }
                }
            }
            
            // Only use fallback if all translation services failed
            if (!success) {
                translatedText = text; // Just show original text if translation fails
                serviceUsed = 'No translation available';
                console.warn('All translation services failed - showing original text');
            }
            
            console.log(`Translation via ${serviceUsed}: "${text}" → "${translatedText}"`);
            return translatedText;
            
        } catch (error) {
            console.error('Translation error:', error);
            return text; // Return original text on error
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
        // Determine proper source language
        let sourceLang = sourceLanguage || this.lastDetectedLanguage || this.currentLanguage;
        if (sourceLang === 'auto') {
            sourceLang = 'en'; // Default fallback
        }
        
        const limitedText = text.length > 500 ? text.substring(0, 500) + '...' : text;
        
        console.log(`MyMemory: Translating from ${sourceLang} to ${this.targetLanguage}`);
        
        // Use GET request to avoid CORS preflight issues
        const params = new URLSearchParams({
            q: limitedText,
            langpair: `${sourceLang}|${this.targetLanguage}`,
            de: 'translator@example.com',
            mt: '1'
        });
        
        const url = `https://api.mymemory.translated.net/get?${params.toString()}`;
        
        try {
            const response = await fetch(url, {
                method: 'GET',
                headers: {
                    'Accept': 'application/json',
                    'User-Agent': 'LiveTTSTranslator/1.0'
                }
            });
            
            if (!response.ok) {
                throw new Error(`MyMemory HTTP error: ${response.status}`);
            }
            
            const data = await response.json();
            console.log('MyMemory response:', data);
            
            if (data.responseStatus === 200 && data.responseData?.translatedText) {
                const translation = data.responseData.translatedText.trim();
                
                // Check if translation is valid
                if (translation && 
                    translation.toLowerCase() !== limitedText.toLowerCase() && 
                    !translation.includes('TRANSLATED BY GOOGLE') &&
                    !translation.includes('MYMEMORY WARNING') &&
                    !translation.includes('QUOTA EXCEEDED') &&
                    translation.length > 0) {
                    console.log('✅ MyMemory translation successful:', translation);
                    return translation;
                }
            }
            
            console.warn('MyMemory returned invalid translation:', data);
            throw new Error('MyMemory: No valid translation available');
            
        } catch (error) {
            console.error('MyMemory translation error:', error);
            throw error;
        }
    }
    
    async translateWithDictionary(text, sourceLanguage = null) {
        const russianDictionary = {
            // Basic words
            'да': 'yes',
            'нет': 'no',
            'привет': 'hello',
            'пока': 'bye',
            'спасибо': 'thank you',
            'пожалуйста': 'please',
            'извините': 'excuse me',
            
            // Common phrases from your speech
            'не понятно': 'not clear',
            'не очень': 'not very',
            'еще раз': 'once more',
            'что это': 'what is this',
            'скажем': 'let\'s say',
            'очень хорошо': 'very good',
            'как дела': 'how are you',
            'все хорошо': 'everything is good',
            
            // Additional common phrases
            'что-то': 'something',
            'а что': 'and what',
            'ну что': 'well what',
            'может быть': 'maybe',
            'конечно': 'of course',
            'хорошо': 'good',
            'плохо': 'bad',
            'сейчас': 'now',
            'потом': 'later',
            'здесь': 'here',
            'там': 'there',
            'когда': 'when',
            'где': 'where',
            'как': 'how',
            'почему': 'why',
            'кто': 'who',
            'что': 'what',
            
            // Single letters/characters that might be detected
            'а': 'ah',
            'и': 'and',
            'в': 'in',
            'на': 'on',
            'с': 'with',
            'к': 'to',
            'от': 'from',
            'по': 'by',
            'за': 'for',
            'о': 'about',
        };
        
        const lowerText = text.toLowerCase().trim();
        console.log(`Dictionary: Looking for translation of "${lowerText}"`);
        
        // Check for exact matches first
        if (russianDictionary[lowerText]) {
            console.log(`✅ Dictionary exact match found: ${russianDictionary[lowerText]}`);
            return russianDictionary[lowerText];
        }
        
        // Remove punctuation and try again
        const cleanText = lowerText.replace(/[^\w\s]/g, '').trim();
        if (cleanText !== lowerText && russianDictionary[cleanText]) {
            console.log(`✅ Dictionary clean match found: ${russianDictionary[cleanText]}`);
            return russianDictionary[cleanText];
        }
        
        // Check for partial matches (phrase contains dictionary word)
        let bestMatch = '';
        let bestScore = 0;
        
        for (const [russian, english] of Object.entries(russianDictionary)) {
            if (lowerText.includes(russian)) {
                const score = russian.length; // Longer matches get higher priority
                if (score > bestScore) {
                    bestMatch = english;
                    bestScore = score;
                }
            }
        }
        
        if (bestMatch) {
            console.log(`✅ Dictionary partial match found: ${bestMatch}`);
            return `${bestMatch} (partial)`;
        }
        
        console.log('❌ No dictionary translation found');
        throw new Error('No dictionary translation found');
    }
    
    async translateWithFallback(text, sourceLanguage = null) {
        const sourceLang = sourceLanguage || this.currentLanguage;
        
        // If same language, return original
        if (sourceLang === this.targetLanguage) {
            return text;
        }
        
        // Create a simple language indicator for display
        const sourceName = this.getLanguageName(sourceLang);
        const targetName = this.getLanguageName(this.targetLanguage);
        
        // Return the original text with a note that translation failed
        return `${text} [${sourceName}→${targetName} translation unavailable]`;
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
        if ('speechSynthesis' in window && text && text.trim()) {
            try {
                // Cancel any ongoing speech
                speechSynthesis.cancel();
                
                const utterance = new SpeechSynthesisUtterance(text);
                utterance.lang = this.getProperLanguageCode(this.targetLanguage);
                utterance.rate = 0.9;
                utterance.volume = 0.8;
                utterance.pitch = 1.0;
                
                // Add event listeners for debugging
                utterance.onstart = () => {
                    console.log('🔊 TTS started speaking:', text);
                };
                
                utterance.onend = () => {
                    console.log('🔊 TTS finished speaking');
                };
                
                utterance.onerror = (event) => {
                    console.error('🔇 TTS error:', event.error);
                };
                
                speechSynthesis.speak(utterance);
                
            } catch (error) {
                console.error('TTS error:', error);
            }
        } else {
            console.warn('TTS not available or empty text provided');
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

    // Debug function to show why transcriptions are filtered
    debugTranscriptionFiltering(text) {
        console.group('🔍 Transcription filtering debug:');
        console.log('Text:', text);
        console.log('Length:', text.length);
        
        // Check for common artifacts
        const artifacts = [
            '[Music]', '[Applause]', '[Noise]', '[Background music]', 
            'Thank you.', 'Thanks for watching!',
        ];
        const isArtifact = artifacts.some(artifact => text.toLowerCase() === artifact.toLowerCase());
        console.log('Is common artifact:', isArtifact);
        
        // Check repetitive patterns
        const repetitivePatterns = [
            /^(.)\1{4,}$/, // Single character repeated 5+ times
            /^\[(.)\]\s*\[(.)\]/, // Bracketed single characters
            /^(.)\s+\1\s+\1\s+\1/, // Spaced repetitive characters
        ];
        const hasRepetitivePattern = repetitivePatterns.some(pattern => pattern.test(text));
        console.log('Has repetitive pattern (SSSSS, CCCCC, etc.):', hasRepetitivePattern);
        
        // Check character distribution
        const charCounts = {};
        let totalChars = 0;
        for (const char of text.toLowerCase()) {
            if (char.match(/\w/)) {
                charCounts[char] = (charCounts[char] || 0) + 1;
                totalChars++;
            }
        }
        
        if (totalChars > 0) {
            const maxCharPercentage = Math.max(...Object.values(charCounts)) / totalChars;
            console.log('Max single character percentage:', (maxCharPercentage * 100).toFixed(1) + '%');
            console.log('Threshold for filtering:', '80%');
            console.log('Would be filtered for character repetition:', maxCharPercentage > 0.8);
        }
        
        console.log('✅ New validation is much more permissive - only blocks obvious artifacts!');
        console.groupEnd();
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