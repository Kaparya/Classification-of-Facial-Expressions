import cv2

def list_cameras():
    """
    Проверяет первые 10 индексов камер и выводит информацию о доступных.
    """
    index = 0
    available_cameras = []
    
    print("Поиск доступных камер...")
    print("-" * 30)
    
    while index < 10:
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            # Получаем параметры камеры
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            backend = cap.getBackendName()
            
            print(f"Камера [{index}]: ДОСТУПНА | Разрешение: {int(width)}x{int(height)} | Backend: {backend}")
            available_cameras.append(index)
            cap.release()
        else:
            # Некоторые индексы могут быть пропущены, но мы проверяем до 10
            pass
        index += 1
    
    print("-" * 30)
    if not available_cameras:
        print("Камеры не найдены.")
    else:
        print(f"Всего найдено камер: {len(available_cameras)}")
        print(f"Индексы: {available_cameras}")

if __name__ == "__main__":
    list_cameras()
