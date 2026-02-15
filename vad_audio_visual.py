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


ssl._create_default_https_context = ssl._create_unverified_context

MODEL_PATH = 'raw_models/face_landmarker.task'

VISUAL_THRESHOLD = 0.005
HISTORY_LENGTH = 7

SAMPLING_RATE = 16000
CHUNK_SIZE = 512 

LIP_INNER_UPPER = 13
LIP_INNER_LOWER = 14
FACE_TOP = 10
FACE_BOTTOM = 152
    

audio_speaking = False
audio_confidence = 0.0
audio_history = []
model = None
running = True

def audio_callback(indata, frames, time_info, status):
    global audio_speaking, audio_confidence, model, audio_history, running
    if not running:
        raise sd.CallbackAbort()
    if status:
        return
    
    if model is None:
        return

    audio_float32 = indata.flatten().astype(np.float32)
    tensor_input = torch.from_numpy(audio_float32)
    
    with torch.no_grad():
        confidence = model(tensor_input, SAMPLING_RATE).item()
        audio_confidence = confidence
        audio_speaking = confidence > 0.4

        cur_history = audio_history
        cur_history.append(audio_speaking)
        if len(cur_history) > HISTORY_LENGTH: cur_history.pop(0)
        audio_speaking = sum(cur_history) >= (HISTORY_LENGTH // 2 + 1)
        audio_history = cur_history

def audio_thread_function():
    global model
    print("Загрузка модели Silero VAD...")
    try:
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                      model='silero_vad',
                                      force_reload=False,
                                      onnx=False)
        print("Модель Silero VAD успешно загружена.")
        
        with sd.InputStream(samplerate=SAMPLING_RATE, 
                            channels=1, 
                            callback=audio_callback, 
                            blocksize=CHUNK_SIZE, 
                            dtype='float32'):
            print("Микрофон активирован.")
            while running:
                time.sleep(0.1)
    except Exception as e:
        print(f"Ошибка в аудио-потоке: {e}")

def calculate_distance(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def main():
    threading.Thread(target=audio_thread_function, daemon=True).start()

    if not os.path.exists(MODEL_PATH):
        print(f"Ошибка: Файл {MODEL_PATH} не найден!")
        return

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
        running_mode=vision.RunningMode.VIDEO)

    visual_history = []

    cap = None
    for idx in [0, 1]:
        temp_cap = cv2.VideoCapture(idx)
        if temp_cap.isOpened():
            cap = temp_cap
            print(f"Используется камера с индексом {idx}")
            break
    
    if cap is None:
        print("Ошибка: Не удалось открыть ни одну камеру.")
        return

    print("Запуск комбинированного VAD...")
    
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            timestamp_ms = int(time.time() * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            
            video_speaking = False
            norm_lip_dist = 0.0
            
            if result.face_landmarks:
                for face_landmarks in result.face_landmarks:
                    p_upper = face_landmarks[LIP_INNER_UPPER]
                    p_lower = face_landmarks[LIP_INNER_LOWER]
                    p_face_top = face_landmarks[FACE_TOP]
                    p_face_bottom = face_landmarks[FACE_BOTTOM]
                    
                    lip_dist = calculate_distance(p_upper, p_lower)
                    face_size = calculate_distance(p_face_top, p_face_bottom)
                    norm_lip_dist = lip_dist / face_size
                    
                    is_speaking = norm_lip_dist > VISUAL_THRESHOLD
                    visual_history.append(is_speaking)
                    if len(visual_history) > HISTORY_LENGTH: visual_history.pop(0)
                    video_speaking = sum(visual_history) >= (HISTORY_LENGTH // 2 + 1)
                    
                    # Отрисовка точек
                    h, w, _ = frame.shape
                    cv2.circle(frame, (int(p_upper.x * w), int(p_upper.y * h)), 2, (255, 255, 255), -1)
                    cv2.circle(frame, (int(p_lower.x * w), int(p_lower.y * h)), 2, (255, 255, 255), -1)

            # Статусы
            a_color = (0, 255, 0) if audio_speaking else (0, 0, 255)
            cv2.putText(frame, f"AUDIO: {'SPEAKING' if audio_speaking else 'SILENT'} ({audio_confidence:.2f})", 
                        (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, a_color, 2)
            
            v_color = (0, 255, 0) if video_speaking else (0, 0, 255)
            cv2.putText(frame, f"VISUAL: {'SPEAKING' if video_speaking else 'SILENT'} ({norm_lip_dist:.4f})", 
                        (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, v_color, 2)
            
            final_speaking = audio_speaking and video_speaking
            f_color = (0, 255, 255) if final_speaking else (255, 255, 255)
            cv2.putText(frame, f"FINAL: {'SPEAKING' if final_speaking else 'SILENT'}", 
                        (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, f_color, 3)

            cv2.imshow('Multimodal VAD', frame)
            if cv2.waitKey(1) & 0xFF == 27: 
                global running
                running = False
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
