import os
import urllib.request

MODEL_PATH = 'face_landmarker.task'
if not os.path.exists(MODEL_PATH):
    print("Скачивание модели Face Landmarker...")
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("Модель успешно скачана!")
import eventlet
eventlet.monkey_patch()
import cv2
import numpy as np
import math
import time
import base64
from collections import deque
import mediapipe as mp
import flask
from flask_socketio import SocketIO, emit

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

app = flask.Flask(__name__, static_folder='static', template_folder='templates')
# Инициализация WebSockets
socketio = SocketIO(app, cors_allowed_origins="*")

try:
    base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1
    )
    detector = vision.FaceLandmarker.create_from_options(options)
except Exception as e:
    print(f"ОШИБКА инициализации модели: {e}")
    exit()

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
FACE_3D_POINTS = [
    (0.0, 0.0, 0.0), (0.0, 330.0, -65.0),
    (-225.0, -170.0, -135.0), (225.0, -170.0, -135.0),
    (-150.0, 150.0, -125.0), (150.0, 150.0, -125.0)
]
FACE_INDICES = [1, 152, 33, 263, 61, 291]

YAW_MAX = 30.0
PITCH_MAX = 25.0
PITCH_DROP = 15.0  
T_BLINK = 0.3
ALPHA_EMA = 0.1
Q_MIN = 0.7

class DrowsinessPipeline:
    def __init__(self):
        self.ear_base = None
        self.calibration_samples = []
        self.calibration_frames_needed = 30
        self.history = deque()
        self.closure_active = False
        self.closure_start_time = 0.0
        self.current_closure_duration = 0.0
        
        self.active_branch = "NONE"      
        self.current_phase = 0           
        self.last_time = time.time()
        self.cooldown_timer = 0.0
        self.cooldown_start_phase = 0

    def calculate_ear(self, face_landmarks, eye_indices, img_w, img_h):
        pts = [(face_landmarks[i].x * img_w, face_landmarks[i].y * img_h) for i in eye_indices]
        v1 = math.dist(pts[1], pts[5])
        v2 = math.dist(pts[2], pts[4])
        h = math.dist(pts[0], pts[3])
        if h == 0: return 0.0, [(int(p[0]), int(p[1])) for p in pts]
        return (v1 + v2) / (2.0 * h), [(int(p[0]), int(p[1])) for p in pts]

    def get_true_head_pose(self, face_landmarks, img_w, img_h):
        image_points = np.array([(face_landmarks[i].x * img_w, face_landmarks[i].y * img_h) for i in FACE_INDICES], dtype="double")
        model_points = np.array(FACE_3D_POINTS, dtype="double")
        focal_length = img_w
        center = (img_w / 2, img_h / 2)
        camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double")
        dist_coeffs = np.zeros((4, 1))
        success, rvec, tvec = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        if not success: return 0.0, 0.0, (0,0), (0,0)
        rmat, _ = cv2.Rodrigues(rvec)
        proj_matrix = np.hstack((rmat, tvec))
        _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(proj_matrix)
        return float(euler[0][0]), float(euler[1][0]), (int(image_points[0][0]), int(image_points[0][1])), (0,0)

    def process_frame(self, detection_result, img_w, img_h):
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        if dt > 1.0: dt = 0.033

        eye_state = "UNKNOWN"
        closure_level_pct = 0.0
        ear_avg = 0.0
        pitch, yaw = 0.0, 0.0
        is_valid = False
        is_closed = False
        just_blinked = False
        pts_draw = []

        if detection_result and detection_result.face_landmarks:
            face_landmarks = detection_result.face_landmarks[0]
            pitch, yaw, _, _ = self.get_true_head_pose(face_landmarks, img_w, img_h)
            
            if abs(yaw) <= YAW_MAX and abs(pitch) <= PITCH_MAX:
                is_valid = True
                ear_l, pts_l = self.calculate_ear(face_landmarks, LEFT_EYE, img_w, img_h)
                ear_r, pts_r = self.calculate_ear(face_landmarks, RIGHT_EYE, img_w, img_h)
                ear_avg = round((ear_l + ear_r) / 2.0, 3)
                pts_draw = pts_l + pts_r

                if self.ear_base is None:
                    self.calibration_samples.append(ear_avg)
                    if len(self.calibration_samples) >= self.calibration_frames_needed:
                        self.ear_base = np.median(self.calibration_samples)
                    eye_state = "CALIBRATING"
                else:
                    cl = 1.0 - (ear_avg - (self.ear_base * 0.25)) / (self.ear_base - (self.ear_base * 0.25) + 1e-6)
                    closure_level_pct = np.clip(cl, 0.0, 1.0) * 100.0
                    if closure_level_pct < 50.0: eye_state = "OPEN"
                    elif closure_level_pct < 85.0: eye_state = "PARTIAL"
                    else: 
                        eye_state = "CLOSED"
                        is_closed = True
            else:
                if self.closure_active:
                    is_valid = True
                    is_closed = True
                    eye_state = "CLOSED (HEAD ROLL)"
                else:
                    eye_state = "UNKNOWN (HEAD TURN)"

            if abs(pitch) > PITCH_DROP and self.current_closure_duration >= 1.0:
                is_valid = True
                is_closed = True
                eye_state = "CLOSED (NODDING)"

        if is_closed:
            if not self.closure_active:
                self.closure_active = True
                self.closure_start_time = current_time
            self.current_closure_duration = current_time - self.closure_start_time
        else:
            if self.closure_active:
                if 0.05 < self.current_closure_duration < T_BLINK: just_blinked = True
                self.closure_active = False
            self.current_closure_duration = 0.0

        if self.ear_base is not None:
            self.history.append({'time': current_time, 'dt': dt, 'valid': is_valid, 'closed': is_closed, 'blink': just_blinked})

        while self.history and current_time - self.history[0]['time'] > 60.0: self.history.popleft()

        def calc_win(sec):
            v_t, c_t = 0.0, 0.0
            for item in reversed(self.history):
                if current_time - item['time'] > sec: break
                if item['valid']:
                    v_t += item['dt']
                    if item['closed']: c_t += item['dt']
            valid_ok = (v_t / sec) >= Q_MIN if sec > 0 else False
            return (c_t / v_t * 100.0) if valid_ok and v_t > 0 else 0.0, valid_ok

        p10, v10 = calc_win(10.0)
        p60, v60 = calc_win(60.0)

        history_duration = current_time - self.history[0]['time'] if self.history else 0
        blinks_count = sum(1 for item in self.history if item['blink'])
        bpm = int((blinks_count / history_duration) * 60) if history_duration > 10 else 0

        if self.ear_base is not None and is_valid and eye_state == "OPEN" and not (p10 > 40 or p60 > 35):
            self.ear_base = (1.0 - ALPHA_EMA) * self.ear_base + ALPHA_EMA * ear_avg

        if self.ear_base is not None and len(self.calibration_samples) >= self.calibration_frames_needed:
            p1_t = 2.0   
            p2_t = 4.5   
            p3_t = 6.5   
            p4_t = 8.5   

            new_phase = 0
            
            if self.current_closure_duration >= p4_t: new_phase = 4
            elif self.current_closure_duration >= p3_t: new_phase = 3
            elif self.current_closure_duration >= p2_t: new_phase = 2
            elif self.current_closure_duration >= p1_t: new_phase = 1

            if new_phase == 0:
                if v10 and p10 >= 45 and v60 and p60 >= 35: new_phase = 2 
                elif v10 and p10 >= 40: new_phase = 1 
                elif history_duration > 30 and v60 and (bpm < 4 or bpm > 30): new_phase = 1

            if new_phase < self.current_phase:
                if self.active_branch != "COOLDOWN":
                    self.active_branch = "COOLDOWN"
                    self.cooldown_timer = current_time
                    self.cooldown_start_phase = self.current_phase

                elapsed = current_time - self.cooldown_timer
                
                if self.cooldown_start_phase == 4:
                    if elapsed < 5.0: self.current_phase = 3
                    elif elapsed < 10.0: self.current_phase = 2
                    elif elapsed < 12.5: self.current_phase = 1
                    else: 
                        self.current_phase = 0
                        self.active_branch = "NONE"
                elif self.cooldown_start_phase == 3:
                    if elapsed < 5.0: self.current_phase = 2
                    elif elapsed < 7.5: self.current_phase = 1
                    else: 
                        self.current_phase = 0
                        self.active_branch = "NONE"
                elif self.cooldown_start_phase == 2:
                    if elapsed < 2.5: self.current_phase = 1
                    else: 
                        self.current_phase = 0
                        self.active_branch = "NONE"
                else: 
                    self.current_phase = 0
                    self.active_branch = "NONE"
            else:
                self.current_phase = new_phase
                if new_phase > 0:
                    self.active_branch = "BRANCH_1_CLOSURE" if self.current_closure_duration >= p1_t else "BRANCH_2_PERCLOS/BPM"
                else:
                    self.active_branch = "NONE"
        else:
            self.current_phase = 0
            self.active_branch = "NONE"

        status_text = f"NORMAL" if self.current_phase == 0 else f"PHASE {self.current_phase} ({self.active_branch})"

        return {
            'warning': status_text,
            'phase': self.current_phase,
            'branch': self.active_branch,
            'ear': ear_avg, 
            'base': self.ear_base if self.ear_base else 0.0,
            'p10': p10, 
            'p60': p60, 
            'closure': closure_level_pct, 
            'eye_state': eye_state,
            'duration': self.current_closure_duration,
            'bpm': bpm,
            'history_len': history_duration,
            'valid': is_valid,
            'pts': pts_draw
        }

pipeline = DrowsinessPipeline()

# === НОВЫЙ БЛОК: WEBSOCKETS ДЛЯ ОБРАБОТКИ КАДРОВ ===
@socketio.on('process_frame')
def handle_frame(data):
    # Клиент отправляет кадр в формате base64
    image_data = data.split(',')[1]
    decoded_data = base64.b64decode(image_data)
    np_data = np.frombuffer(decoded_data, np.uint8)
    frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

    if frame is None:
        return

    # Зеркалим для естественности (как в зеркале)
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    res = detector.detect(mp_image)
    
    latest_data = pipeline.process_frame(res, w, h)
    
    # Отрисовка ИИ-точек на кадре
    for pt in latest_data.get('pts', []):
        cv2.circle(frame, pt, 2, (0, 255, 0), -1)

    t_ms = int(time.time() * 1000)
    phase = latest_data.get('phase', 0)
    color = (0, 255, 0)

    if phase == 1:
        color = (0, 165, 255) if (t_ms % 600) > 300 else (0, 80, 120)
    elif phase == 2:
        color = (255, 140, 0) if (t_ms % 300) > 150 else (120, 60, 0)
    elif phase == 3:
        color = (0, 0, 255) if (t_ms % 150) > 75 else (0, 0, 100)
    elif phase >= 4:
        color = (138, 43, 226) if (t_ms % 80) > 40 else (0, 0, 0)
    
    cv2.putText(frame, f"STATUS: {latest_data.get('warning', 'NORMAL')}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Кодируем кадр обратно в base64 и отправляем браузеру вместе с метриками
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    encoded_frame = base64.b64encode(buffer).decode('utf-8')
    
    emit('frame_result', {
        'image': 'data:image/jpeg;base64,' + encoded_frame,
        'metrics': latest_data
    })

@app.route('/')
def index():
    return flask.render_template('index.html')

@app.route('/reset_calibration', methods=['POST'])
def reset_calibration():
    global pipeline
    pipeline = DrowsinessPipeline()
    return flask.jsonify({"status": "reset_success"})

@app.route('/reset_history', methods=['POST'])
def reset_history():
    global pipeline
    pipeline.history.clear()
    pipeline.current_closure_duration = 0.0
    pipeline.closure_active = False
    pipeline.current_phase = 0
    pipeline.active_branch = "NONE"
    pipeline.cooldown_timer = 0.0
    return flask.jsonify({"status": "history_cleared"})

if __name__ == '__main__':
    # Запуск через socketio
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
