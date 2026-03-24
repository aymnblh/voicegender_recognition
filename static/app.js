// static/app.js

document.addEventListener('DOMContentLoaded', () => {
    const recordBtn = document.getElementById('record-btn');
    const statusText = document.getElementById('status-text');
    const statusDot = document.getElementById('status-dot');
    const waves = document.getElementById('waves');
    const instruction = document.getElementById('instruction');

    const resultSection = document.getElementById('result-section');
    const predictionLabel = document.getElementById('prediction-label');
    const genderIcon = document.getElementById('gender-icon');
    const confidenceValue = document.getElementById('confidence-value');
    const progressBarFill = document.getElementById('progress-bar-fill');
    const probMale = document.getElementById('prob-male');
    const probFemale = document.getElementById('prob-female');

    const resetBtn = document.getElementById('reset-btn');
    const errorBox = document.getElementById('error-box');
    const errorMessage = document.getElementById('error-message');

    let isRecording = false;
    let mediaRecorder = null;
    let audioChunks = [];

    // --- 1. Gestion des événements UI ---

    recordBtn.addEventListener('click', async () => {
        if (!isRecording) {
            startRecording();
        } else {
            stopRecording();
        }
    });

    resetBtn.addEventListener('click', () => {
        // Réinitialiser l'interface
        resultSection.classList.add('hidden');
        errorBox.classList.add('hidden');
        recordBtn.style.display = 'flex';
        instruction.style.display = 'block';
        instruction.textContent = "Appuyez pour parler";
        updateStatus('Prêt à écouter', 'ready');

        // Retirer les classes de couleur
        resultSection.classList.remove('is-male', 'is-female');
        progressBarFill.style.width = '0%';
    });


    // --- 2. Logique Audio (Web Audio API) ---

    async function startRecording() {
        try {
            // Demander l'accès au micro
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

            // Initialiser le MediaRecorder
            mediaRecorder = new MediaRecorder(stream);
            audioChunks = [];

            mediaRecorder.addEventListener('dataavailable', event => {
                if (event.data.size > 0) {
                    audioChunks.push(event.data);
                }
            });

            mediaRecorder.addEventListener('stop', () => {
                // Quand l'enregistrement est fini, on crée un Blob et on l'envoie
                const audioBlob = new Blob(audioChunks, { type: 'audio/webm' }); // ou audio/ogg selon navigateur

                // Fermer les pistes du micro
                stream.getTracks().forEach(track => track.stop());

                sendAudioToAPI(audioBlob);
            });

            // Lancer l'enregistrement
            mediaRecorder.start();
            isRecording = true;

            // Mettre à jour l'UI
            recordBtn.classList.add('recording');
            waves.classList.remove('hidden');
            instruction.textContent = "Appuyez pour arrêter l'enregistrement";
            updateStatus('Enregistrement en cours...', 'recording');
            errorBox.classList.add('hidden');
            resultSection.classList.add('hidden');

        } catch (err) {
            console.error("Erreur d'accès au microphone:", err);
            showError("Impossible d'accéder au microphone. Veuillez vérifier les permissions de votre navigateur.");
        }
    }

    function stopRecording() {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
            isRecording = false;

            // Mettre à jour l'UI
            recordBtn.classList.remove('recording');
            waves.classList.add('hidden');
            recordBtn.style.display = 'none'; // Cacher le bouton pendant l'analyse
            instruction.style.display = 'none';
            updateStatus('Analyse par l\'IA en cours...', 'processing');
        }
    }


    // --- 3. Communication avec le Backend (Flask API) ---

    async function sendAudioToAPI(audioBlob) {
        updateStatus('Conversion audio...', 'processing');

        try {
            // Conversion WebM -> WAV via le navigateur pour éviter FFmpeg côté serveur
            const wavBlob = await convertWebmToWav(audioBlob);

            const formData = new FormData();
            formData.append('audio', wavBlob, 'recording.wav');

            updateStatus('Analyse par l\'IA en cours...', 'processing');
            const response = await fetch('/api/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok && data.success) {
                displayResult(data);
            } else {
                showError(data.error || "Une erreur est survenue lors de l'analyse.");
                resetToReady();
            }

        } catch (error) {
            console.error("Erreur réseau/conversion:", error);
            showError("Impossible de contacter le serveur IA. Est-il lancé ?");
            resetToReady();
        }
    }

    // Fonction qui utilise l'AudioContext pour décoder le WebM et le ré-encoder en WAV
    async function convertWebmToWav(webmBlob) {
        const arrayBuffer = await webmBlob.arrayBuffer();
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);

        return audioBufferToWav(audioBuffer);
    }

    // Encode un AudioBuffer en WAV format 16-bit Mono (Idéal pour Librosa)
    function audioBufferToWav(buffer) {
        const numChannels = 1; // On force le mono
        const sampleRate = buffer.sampleRate;
        const format = 1; // PCM
        const bitDepth = 16;

        const channelData = buffer.getChannelData(0); // On prend la voie gauche
        const dataLength = channelData.length * (bitDepth / 8);
        const bufferWav = new ArrayBuffer(44 + dataLength);
        const view = new DataView(bufferWav);

        // Helper string writer
        const writeString = (view, offset, string) => {
            for (let i = 0; i < string.length; i++) {
                view.setUint8(offset + i, string.charCodeAt(i));
            }
        };

        writeString(view, 0, 'RIFF');
        view.setUint32(4, 36 + dataLength, true);
        writeString(view, 8, 'WAVE');
        writeString(view, 12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, format, true);
        view.setUint16(22, numChannels, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * numChannels * (bitDepth / 8), true);
        view.setUint16(32, numChannels * (bitDepth / 8), true);
        view.setUint16(34, bitDepth, true);
        writeString(view, 36, 'data');
        view.setUint32(40, dataLength, true);

        let offset = 44;
        for (let i = 0; i < channelData.length; i++) {
            let sample = Math.max(-1, Math.min(1, channelData[i]));
            sample = sample < 0 ? sample * 32768 : sample * 32767;
            view.setInt16(offset, sample, true);
            offset += 2;
        }

        return new Blob([view], { type: 'audio/wav' });
    }



    // --- 4. Affichage des résultats ---

    function displayResult(data) {
        updateStatus('Analyse terminée', 'ready');

        const isMale = data.prediction.toLowerCase() === 'male';

        // Mettre à jour les textes et icônes
        predictionLabel.textContent = isMale ? "Homme" : "Femme";
        genderIcon.className = isMale ? 'ph-duotone ph-gender-male' : 'ph-duotone ph-gender-female';

        confidenceValue.textContent = `${data.confidence}%`;

        probMale.textContent = `${data.details.male.toFixed(2)}%`;
        probFemale.textContent = `${data.details.women.toFixed(2)}%`;

        // Gérer les couleurs via des classes CSS
        resultSection.classList.remove('is-male', 'is-female');
        resultSection.classList.add(isMale ? 'is-male' : 'is-female');

        // Afficher la section résultat
        resultSection.classList.remove('hidden');

        // Animer la barre de progression (petit délai pour l'effet)
        setTimeout(() => {
            progressBarFill.style.width = `${data.confidence}%`;
        }, 100);
    }

    function updateStatus(text, dotClass) {
        statusText.textContent = text;
        statusDot.className = `dot ${dotClass}`;
    }

    function showError(msg) {
        errorMessage.textContent = msg;
        errorBox.classList.remove('hidden');
    }

    function resetToReady() {
        recordBtn.style.display = 'flex';
        instruction.style.display = 'block';
        instruction.textContent = "Appuyez pour réessayer";
        updateStatus('Prêt', 'ready');
    }
});
