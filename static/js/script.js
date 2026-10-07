// ==========================================
// НОВАЯ ЛОГИКА: CLIENT-SIDE CAMERA И WEBSOCKETS
// ==========================================

const socket = io();
const videoEl = document.getElementById('webcam');
const canvasEl = document.getElementById('canvas');
const ctx = canvasEl.getContext('2d');
const videoStreamImg = document.getElementById('video_stream');

let isStreaming = false;

// Запрашиваем доступ к веб-камере пользователя
navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } })
    .then((stream) => {
        videoEl.srcObject = stream;
        videoEl.play();
    })
    .catch((err) => {
        console.error("Ошибка доступа к камере: ", err);
        alert("Дайте разрешение на использование камеры в браузере!");
    });

videoEl.addEventListener('play', () => {
    isStreaming = true;
    canvasEl.width = videoEl.videoWidth;
    canvasEl.height = videoEl.videoHeight;
    // Начинаем отправлять кадры на сервер
    sendFrame();
});

// Функция отправки кадра на сервер (примерно 15-20 FPS)
function sendFrame() {
    if (!isStreaming) return;
    
    // Рисуем кадр из <video> на скрытый <canvas>
    ctx.drawImage(videoEl, 0, 0, canvasEl.width, canvasEl.height);
    
    // Сжимаем кадр в base64 (jpeg, качество 0.6 для экономии трафика)
    const frameData = canvasEl.toDataURL('image/jpeg', 0.6);
    
    // Отправляем на бэкенд ИИ
    socket.emit('process_frame', frameData);
    
    // Запускаем следующий кадр с небольшой задержкой (чтобы не убить сервер)
    setTimeout(sendFrame, 60); 
}

// Принимаем готовый кадр и метрики от сервера
socket.on('frame_result', (data) => {
    // Обновляем картинку (уже с отрисованными точками)
    videoStreamImg.src = data.image;
    
    const metrics = data.metrics;
    if(!metrics) return;

    // ОБНОВЛЕНИЕ МЕТРИК В ИНТЕРФЕЙСЕ (Точно как в твоем старом коде)
    valState.textContent = metrics.eye_state || "UNKNOWN";
    if(metrics.eye_state === 'OPEN') {
        valState.style.color = '#3fb950';
    } else if(metrics.eye_state === 'CLOSED' || metrics.eye_state === 'CLOSED (NODDING)' || metrics.eye_state === 'CLOSED (HEAD ROLL)') {
        valState.style.color = '#f85149'; 
    } else {
        valState.style.color = '#d29922';
    }

    valEar.textContent = `${metrics.ear.toFixed(3)} / ${metrics.base.toFixed(1)}`;
    valPerclos.textContent = `${metrics.p10.toFixed(1)}% / ${metrics.p60.toFixed(1)}%`;
    valDuration.textContent = metrics.duration.toFixed(1) + 's';

    valTrack.textContent = metrics.valid ? "РАСПОЗНАНО" : "ПОВОРОТ / ПОТЕРЯ";
    valTrack.style.color = metrics.valid ? "#3fb950" : "#f85149";

    if (metrics.history_len > 10) {
        valBpm.textContent = `${metrics.bpm} BPM`;
        if (metrics.bpm > 30 || (metrics.bpm < 4 && metrics.history_len > 30)) {
            valBpm.style.color = '#d29922'; 
        } else {
            valBpm.style.color = '#e3b341';
        }
    } else {
        valBpm.textContent = "Сбор данных...";
        valBpm.style.color = '#8b949e';
    }

    // ЛОГИКА ФАЗ
    if (!isDemoMode) {
        if (metrics.phase > 0) {
            activatePhase(metrics.phase, metrics.branch);
            countdownEl.textContent = metrics.branch === 'COOLDOWN' ? 'Охлаждение...' : 'РАБОТА ИИ';
        } else {
            if (currentActivePhase !== 0) {
                stopAll();
            }
        }
    }
});