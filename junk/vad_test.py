import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time


MODEL_PATH = 'raw_models/face_landmarker.task'

SPEAKING_THRESHOLD = 0.005
HISTORY_LENGTH = 3

LIP_INNER_UPPER = 13
LIP_INNER_LOWER = 14
FACE_TOP = 10
FACE_BOTTOM = 152


def calculate_distance(p1, p2):
    """Вычисляет Евклидово расстояние между двумя точками."""
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def main():
    
    # Конфигурация MediaPipe Tasks
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
        running_mode=vision.RunningMode.VIDEO)
    speaking_history = []

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
        print("Camera 1 opened successfully")
    else:
        print("Camera 0 opened successfully")

    print("Запуск VAD...")
    
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            
            timestamp_ms = int(time.time() * 1000)
            
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            
            is_speaking = False
            
            if result.face_landmarks:
                for face_landmarks in result.face_landmarks:
                    p_upper = face_landmarks[LIP_INNER_UPPER]
                    p_lower = face_landmarks[LIP_INNER_LOWER]
                    p_face_top = face_landmarks[FACE_TOP]
                    p_face_bottom = face_landmarks[FACE_BOTTOM]
                    
                    lip_dist = calculate_distance(p_upper, p_lower)
                    face_size = calculate_distance(p_face_top, p_face_bottom)
                    norm_lip_dist = lip_dist / face_size
                    
                    if norm_lip_dist > SPEAKING_THRESHOLD:
                        is_speaking = True
                    
                    speaking_history.append(is_speaking)
                    if len(speaking_history) > HISTORY_LENGTH:
                        speaking_history.pop(0)
                    
                    stable_speaking = sum(speaking_history) >= (HISTORY_LENGTH // 2 + 1)
                    
                    # Визуализация
                    h, w, _ = frame.shape
                    color = (0, 255, 0) if stable_speaking else (0, 0, 255)
                    status = "SPEAKING" if stable_speaking else "SILENT"
                    
                    cv2.circle(frame, (int(p_upper.x * w), int(p_upper.y * h)), 2, (255, 255, 255), -1)
                    cv2.circle(frame, (int(p_lower.x * w), int(p_lower.y * h)), 2, (255, 255, 255), -1)
                    
                    cv2.putText(frame, f"{status} ({norm_lip_dist:.4f})", (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            cv2.imshow('MediaPipe Tasks VAD', frame)
            
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
