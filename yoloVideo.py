import cv2
import numpy as np

# Загрузка YOLO
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")

# Загрузка классов из файла coco.names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Открытие видеофайла
cap = cv2.VideoCapture("car.mp4")  # Замени "video_file.mp4" на путь к твоему видео

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break  # Прерывание цикла, если кадры закончились

    height, width, channels = frame.shape

    # Препроцессинг кадра для YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Переменные для хранения данных о детектированных объектах
    class_ids = []
    confidences = []
    boxes = []

    # Интерпретация результатов
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Порог уверенности
                # Расчёт координат рамки
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Верхний левый угол рамки
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Сохранение данных о детектированном объекте
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Применение non-maxima suppression (NMS)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])  # Получение имени класса
            confidence = confidences[i]
            color = colors[class_ids[i]]

            # Отображение рамки и имени класса
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), font, 2, color, 2)

    # Показ текущего кадра с результатами
    cv2.imshow("Video", frame)

    # Прерывание по нажатию клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Закрытие окна и завершение видеопотока
cap.release()
cv2.destroyAllWindows()
