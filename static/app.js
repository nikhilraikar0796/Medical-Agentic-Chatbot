
const $ = id => document.getElementById(id);

let history = [];
let imageData = null;
let recorder = null;
let audioChunks = [];
let cameraStream = null;
let lastAnswer = '';
let conversationId = null;
let activeAudioUrl = null;

// API headers — token is optional for local development.
const headers = () => {
    const h = {};

    const token = $('token')?.value?.trim();
    if (token) {
        h.Authorization = `Bearer ${token}`;
    }

    if (conversationId) {
        h['X-Conversation-ID'] = conversationId;
    }

    return h;
};

// Safely read JSON responses.
async function readResponse(response) {
    const contentType = response.headers.get('content-type') || '';

    if (contentType.includes('application/json')) {
        return await response.json();
    }

    const text = await response.text();
    return { detail: text || `Request failed (${response.status})` };
}

// Display a chat message.
function bubble(text, kind = 'bot', sources = []) {
    const el = document.createElement('article');
    el.className = kind;
    el.textContent = text;

    if (sources.length) {
        const sourceEl = document.createElement('div');
        sourceEl.className = 'sources';

        sourceEl.textContent =
            'Sources: ' +
            sources.slice(0, 8).map(source => {
                if (source.type === 'pdf') {
                    return `PDF page ${source.page}`;
                }

                return source.url || source.title || 'Source';
            }).join(' | ');

        el.appendChild(sourceEl);
    }

    $('messages').appendChild(el);
    $('messages').scrollTop = $('messages').scrollHeight;

    return el;
}

// Server health status.
async function check() {
    try {
        const response = await fetch('/api/health');
        const data = await readResponse(response);

        if (!response.ok) {
            throw new Error(data.detail || 'Health check failed');
        }

        $('status').textContent = data.configured
            ? '● Connected'
            : '⚠ Configure API keys';
    } catch {
        $('status').textContent = '● Server offline';
    }
}

check();

// Image upload.
function setImage(file) {
    if (!file) return;

    const allowedTypes = [
        'image/png',
        'image/jpeg',
        'image/webp'
    ];

    if (!allowedTypes.includes(file.type) || file.size > 5 * 1024 * 1024) {
        alert('Use PNG, JPEG, or WebP images under 5 MB.');
        return;
    }

    const reader = new FileReader();

    reader.onload = () => {
        imageData = reader.result;
        $('thumbnail').src = imageData;
        $('preview').hidden = false;
    };

    reader.onerror = () => {
        alert('Could not read the selected image.');
    };

    reader.readAsDataURL(file);
}

$('image').addEventListener('change', event => {
    setImage(event.target.files[0]);
});

$('remove').onclick = () => {
    imageData = null;
    $('preview').hidden = true;
    $('image').value = '';
};

// Chat submission.
$('chatForm').addEventListener('submit', async event => {
    event.preventDefault();

    const message = $('message').value.trim();
    if (!message) return;

    const prior = history.slice(-10);

    bubble(message, 'user');

    $('message').value = '';
    $('send').disabled = true;

    const pending = bubble('Thinking…');

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                ...headers(),
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message,
                history: prior,
                image_data: imageData
            })
        });

        const data = await readResponse(response);

        if (!response.ok) {
            throw new Error(
                data.detail || `Chat request failed (${response.status})`
            );
        }

        pending.textContent = data.answer || 'No answer was returned.';
        lastAnswer = data.answer || '';

        if (data.conversation_id) {
            conversationId = data.conversation_id;
        }

        if (data.sources?.length) {
            const sourceEl = document.createElement('div');
            sourceEl.className = 'sources';

            sourceEl.textContent =
                'Sources: ' +
                data.sources.slice(0, 8).map(source => {
                    if (source.type === 'pdf') {
                        return `PDF page ${source.page}`;
                    }

                    return source.url || source.title || 'Source';
                }).join(' | ');

            pending.appendChild(sourceEl);
        }

        history.push(
            { role: 'user', content: message },
            { role: 'assistant', content: lastAnswer }
        );

        // Keep the conversation history manageable.
        history = history.slice(-10);

        $('remove').click();
    } catch (error) {
        pending.textContent = `Error: ${error.message}`;
    } finally {
        $('send').disabled = false;
        $('messages').scrollTop = $('messages').scrollHeight;
    }
});

// Voice recording and transcription.
$('mic').onclick = async () => {
    // Stop an active recording.
    if (recorder && recorder.state === 'recording') {
        recorder.stop();
        $('mic').textContent = '🎤 Voice';
        return;
    }

    if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
        alert('Your browser does not support microphone recording.');
        return;
    }

    let stream;

    try {
        stream = await navigator.mediaDevices.getUserMedia({
            audio: true
        });

        audioChunks = [];

        // Choose a recording format supported by the browser.
        const preferredTypes = [
            'audio/webm;codecs=opus',
            'audio/webm',
            'audio/ogg;codecs=opus',
            'audio/ogg',
            'audio/mp4'
        ];

        let selectedType = '';

        if (typeof MediaRecorder.isTypeSupported === 'function') {
            selectedType = preferredTypes.find(type =>
                MediaRecorder.isTypeSupported(type)
            ) || '';
        }

        recorder = selectedType
            ? new MediaRecorder(stream, { mimeType: selectedType })
            : new MediaRecorder(stream);

        const currentRecorder = recorder;

        currentRecorder.ondataavailable = event => {
            if (event.data && event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        currentRecorder.onerror = event => {
            console.error('MediaRecorder error:', event.error);
        };

        currentRecorder.onstop = async () => {
            stream.getTracks().forEach(track => track.stop());

            $('mic').disabled = true;
            $('mic').textContent = 'Transcribing…';

            try {
                if (!audioChunks.length) {
                    throw new Error('No audio was recorded. Please try again.');
                }

                const mimeType =
                    currentRecorder.mimeType ||
                    audioChunks[0].type ||
                    'application/octet-stream';

                const extension = mimeType.includes('ogg')
                    ? 'ogg'
                    : mimeType.includes('mp4')
                        ? 'mp4'
                        : mimeType.includes('wav')
                            ? 'wav'
                            : 'webm';

                const audioBlob = new Blob(audioChunks, {
                    type: mimeType
                });

                if (audioBlob.size === 0) {
                    throw new Error('The recording is empty. Please try again.');
                }

                const formData = new FormData();

                formData.append(
                    'file',
                    audioBlob,
                    `recording.${extension}`
                );

                // Do not manually set Content-Type for FormData.
                const response = await fetch('/api/transcribe', {
                    method: 'POST',
                    headers: headers(),
                    body: formData
                });

                const data = await readResponse(response);

                if (!response.ok) {
                    throw new Error(
                        data.detail ||
                        `Transcription failed (${response.status})`
                    );
                }

                if (!data.text?.trim()) {
                    throw new Error('No speech detected. Please try again.');
                }

                $('message').value = data.text.trim();
                $('message').focus();
            } catch (error) {
                console.error('Transcription error:', error);
                alert(error.message || 'Could not transcribe the recording.');
            } finally {
                $('mic').disabled = false;
                $('mic').textContent = '🎤 Voice';
            }
        };

        currentRecorder.start();
        $('mic').textContent = '⏹ Stop recording';
    } catch (error) {
        stream?.getTracks().forEach(track => track.stop());

        $('mic').textContent = '🎤 Voice';

        alert(`Microphone unavailable: ${error.message}`);
    }
};

// Text-to-speech.
$('listen').onclick = async () => {
    if (!lastAnswer) {
        alert('Ask a question first.');
        return;
    }

    $('listen').disabled = true;

    try {
        const response = await fetch('/api/speak', {
            method: 'POST',
            headers: {
                ...headers(),
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                text: lastAnswer
            })
        });

        if (!response.ok) {
            const data = await readResponse(response);

            throw new Error(
                data.detail || `Speech generation failed (${response.status})`
            );
        }

        const audioBlob = await response.blob();

        if (activeAudioUrl) {
            URL.revokeObjectURL(activeAudioUrl);
        }

        activeAudioUrl = URL.createObjectURL(audioBlob);

        const audio = new Audio(activeAudioUrl);

        audio.onended = () => {
            if (activeAudioUrl) {
                URL.revokeObjectURL(activeAudioUrl);
                activeAudioUrl = null;
            }
        };

        audio.onerror = () => {
            if (activeAudioUrl) {
                URL.revokeObjectURL(activeAudioUrl);
                activeAudioUrl = null;
            }
        };

        await audio.play();
    } catch (error) {
        alert(error.message || 'Could not generate speech.');
    } finally {
        $('listen').disabled = false;
    }
};

// Camera access.
$('camera').onclick = async () => {
    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'environment' }
        });

        $('video').srcObject = cameraStream;
        $('video').hidden = false;
        $('cameraActions').hidden = false;

        await $('video').play().catch(() => {});
    } catch (error) {
        alert(`Camera unavailable: ${error.message}`);
    }
};

// Close camera and release the camera resource.
$('closeCamera').onclick = () => {
    cameraStream?.getTracks().forEach(track => track.stop());

    cameraStream = null;
    $('video').srcObject = null;
    $('video').hidden = true;
    $('cameraActions').hidden = true;
};

// Capture a camera image.
$('snap').onclick = () => {
    const video = $('video');
    const canvas = $('canvas');

    if (!video.videoWidth || !video.videoHeight) {
        alert('Camera is not ready. Please try again.');
        return;
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const context = canvas.getContext('2d');

    if (!context) {
        alert('Could not capture the camera image.');
        return;
    }

    context.drawImage(video, 0, 0);

    canvas.toBlob(blob => {
        if (!blob) {
            alert('Could not create the camera image.');
            return;
        }

        setImage(new File(
            [blob],
            'camera.jpg',
            { type: 'image/jpeg' }
        ));
    }, 'image/jpeg', 0.85);

    $('closeCamera').click();
};

// Release camera/audio resources if the page is closed.
window.addEventListener('beforeunload', () => {
    cameraStream?.getTracks().forEach(track => track.stop());

    if (activeAudioUrl) {
        URL.revokeObjectURL(activeAudioUrl);
    }
});
