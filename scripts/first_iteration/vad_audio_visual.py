import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import threading
import torch
import sounddevice as sd
import ssl
import os
import functools

# Исправление для новых версий PyTorch (2.6+), которые блокируют загрузку весов
original_load = torch.load
torch.load = functools.partial(original_load, weights_only=False)

from hsemotion.facial_emotions import HSEmotionRecognizer

# Решение проблемы с SSL
ssl._create_default_https_context = ssl._create_unverified_context

# --- КОНФИГУРАЦИЯ ---
MODEL_PATH = 'raw_models/face_landmarker.task'
SAMPLING_RATE = 16000
CHUNK_SIZE = 512 

LIP_INNER_UPPER = 13
LIP_INNER_LOWER = 14
FACE_TOP = 10
FACE_BOTTOM = 152

VISUAL_THRESHOLD = 0.005
HISTORY_LENGTH = 7
UTTERANCE_SILENCE_LIMIT = 15

# Глобальные переменные
audio_speaking = False
audio_confidence = 0.0
audio_history = []
model_vad = None
running = True

# Инициализация HSEmotion (загрузится при старте)
# 'mobilenet_7' — быстрая и точная модель для 7 базовых эмоций
fer = HSEmotionRecognizer(model_name='enet_b0_8_best_afew', device='cpu')

# --- АУДИО ЛОГИКА ---
def audio_callback(indata, frames, time_info, status):
    global audio_speaking, audio_confidence, model_vad, audio_history, running
    if not running: raise sd.CallbackAbort()
    if status or model_vad is None: return

    audio_float32 = indata.flatten().astype(np.float32)
    tensor_input = torch.from_numpy(audio_float32)
    
    with torch.no_grad():
        confidence = model_vad(tensor_input, SAMPLING_RATE).item()
        audio_confidence = confidence
        is_spk = confidence > 0.4
        audio_history.append(is_spk)
        if len(audio_history) > HISTORY_LENGTH: audio_history.pop(0)
        audio_speaking = sum(audio_history) >= (HISTORY_LENGTH // 2 + 1)

def audio_thread_function():
    global model_vad
    try:
        model_vad, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', onnx=False)
        with sd.InputStream(samplerate=SAMPLING_RATE, channels=1, callback=audio_callback, 
                            blocksize=CHUNK_SIZE, dtype='float32'):
            while running: time.sleep(0.1)
    except Exception as e: print(f"Audio Error: {e}")

def calculate_distance(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def get_face_bbox(landmarks, w, h):
    """Вычисляет bounding box лица с небольшим отступом."""
    x_coords = [p.x for p in landmarks]
    y_coords = [p.y for p in landmarks]
    
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    # Добавляем 10% отступа
    padding_x = (x_max - x_min) * 0.1
    padding_y = (y_max - y_min) * 0.1
    
    start_x = max(0, int((x_min - padding_x) * w))
    start_y = max(0, int((y_min - padding_y) * h))
    end_x = min(w, int((x_max + padding_x) * w))
    end_y = min(h, int((y_max + padding_y) * h))
    
    return start_x, start_y, end_x, end_y

def process_utterance(frames_blendshapes, best_face_frame):
    """Определяет общую экспрессивность и конкретную эмоцию."""
    if not frames_blendshapes:
        return "NEUTRAL", 0.0, "None"
    
    relevant_indices = [9, 10, 11, 12, 13, 25, 26, 44, 45, 46, 47]
    all_scores = [max([fs[i].score for i in relevant_indices]) for fs in frames_blendshapes]
    avg_expression = np.mean(all_scores)
    
    is_neutral = avg_expression <= 0.15
    binary_status = "NEUTRAL" if is_neutral else "NOT NEUTRAL"
    
    detailed_emotion = "Neutral"
    if not is_neutral and best_face_frame is not None:
        # Используем модель HSEmotion для детальной классификации
        try:
            # Модель ожидает RGB изображение
            rgb_face = cv2.cvtColor(best_face_frame, cv2.COLOR_BGR2RGB)
            detailed_emotion, _ = fer.predict_emotions(rgb_face, logits=False)
        except Exception as e:
            print(f"HSEmotion Error: {e}")
            detailed_emotion = "Error"
            
    return binary_status, avg_expression, detailed_emotion

def main():
    global running
    threading.Thread(target=audio_thread_function, daemon=True).start()
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found")
        return

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        num_faces=1,
        running_mode=vision.RunningMode.VIDEO)

    visual_history = []
    is_recording_utterance = False
    utterance_frames_bs = []
    best_face_frame = None
    max_expression_in_utterance = 0.0
    silence_counter = 0
    
    last_binary = "NONE"
    last_detailed = "None"
    
    cap = None
    for idx in [0, 1]:
        c = cv2.VideoCapture(idx)
        if c.isOpened():
            cap = c; break
    if cap is None: return

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened() and running:
            success, frame = cap.read()
            if not success: break
            h, w, _ = frame.shape

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            result = landmarker.detect_for_video(mp_image, int(time.time() * 1000))
            
            video_speaking = False
            current_bs = None
            current_face_crop = None
            current_expr_score = 0.0

            if result.face_landmarks:
                face_landmarks = result.face_landmarks[0]
                # Вырезаем лицо для HSEmotion
                x1, y1, x2, y2 = get_face_bbox(face_landmarks, w, h)
                if x2 > x1 and y2 > y1:
                    current_face_crop = frame[y1:y2, x1:x2].copy()

                # VAD
                p_up, p_low = face_landmarks[LIP_INNER_UPPER], face_landmarks[LIP_INNER_LOWER]
                face_h = calculate_distance(face_landmarks[FACE_TOP], face_landmarks[FACE_BOTTOM])
                norm_dist = calculate_distance(p_up, p_low) / face_h
                
                video_speaking = norm_dist > VISUAL_THRESHOLD
                visual_history.append(video_speaking)
                if len(visual_history) > HISTORY_LENGTH: visual_history.pop(0)
                video_speaking = sum(visual_history) >= (HISTORY_LENGTH // 2 + 1)
                
                if result.face_blendshapes:
                    current_bs = result.face_blendshapes[0]
                    # Считаем экспрессию для выбора лучшего кадра
                    rel_idx = [9, 10, 11, 12, 13, 25, 26, 44, 45, 46, 47]
                    current_expr_score = max([current_bs[i].score for i in rel_idx])

            # --- УПРАВЛЕНИЕ ОТРЕЗКАМИ ---
            final_speaking = audio_speaking or video_speaking
            
            if final_speaking:
                if not is_recording_utterance:
                    is_recording_utterance = True
                    utterance_frames_bs = []
                    best_face_frame = None
                    max_expression_in_utterance = 0.0
                    print("--- Start ---")
                
                silence_counter = 0
                if current_bs:
                    utterance_frames_bs.append(current_bs)
                    # Если текущий кадр более выразительный, запоминаем его лицо
                    if current_expr_score > max_expression_in_utterance:
                        max_expression_in_utterance = current_expr_score
                        best_face_frame = current_face_crop
            
            elif is_recording_utterance:
                silence_counter += 1
                if silence_counter > UTTERANCE_SILENCE_LIMIT:
                    is_recording_utterance = False
                    last_binary, score, last_detailed = process_utterance(utterance_frames_bs, best_face_frame)
                    print(f"--- End: {last_binary} | {last_detailed} ({score:.2f}) ---")

            # --- UI ---
            # Статусы в реальном времениa_color = (0, 255, 0) if audio_speaking else (0, 0, 255)
            a_color = (0, 255, 0) if audio_speaking else (0, 0, 255)
            cv2.putText(frame, f"AUDIO: {'SPEAKING' if audio_speaking else 'SILENT'} ({audio_confidence:.2f})", 
                        (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, a_color, 2)
            
            v_color = (0, 255, 0) if video_speaking else (0, 0, 255)
            cv2.putText(frame, f"VISUAL: {'SPEAKING' if video_speaking else 'SILENT'} ({norm_dist:.4f})", 
                        (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, v_color, 2)
            cv2.putText(frame, f"SPEECH: {'YES' if final_speaking else 'NO'}", (30, 100), 1, 1.5, (255,255,255), 2)
            
            b_color = (0, 165, 255) if last_binary == "NOT NEUTRAL" else (0, 255, 0)
            cv2.putText(frame, f"TYPE: {last_binary}", (30, 130), 1, 1.5, b_color, 2)
            
            if last_binary == "NOT NEUTRAL":
                cv2.putText(frame, f"EMOTION: {last_detailed.upper()}", (30, 150), 1, 1.5, (0, 255, 255), 2)
            
            if is_recording_utterance:
                cv2.putText(frame, "RECORDING...", (30, 180), 1, 1.2, (0, 255, 255), 2)

            cv2.imshow('Multimodal Emotion Analysis (HSEmotion)', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                running = False; break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
