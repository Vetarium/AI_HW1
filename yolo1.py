import cv2
import numpy as np


net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")


with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

#images for detection
img = cv2.imread("image1.jpeg")
#img = cv2.imread("image2.jpg")
#img = cv2.imread("car.jpg")
height, width, channels = img.shape

# YOLO штуки с процессингом
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)


class_ids = []
confidences = []
boxes = []

# резы детекта
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:  #больше чем 05
            # рамка для обьекта
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            # Данные о чем задетектилось
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Применение non-maxima suppression (NMS) для фильтрации пересекающихся рамок
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(len(classes), 3))

if len(indexes) > 0:
    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])  #имя класса
        confidence = confidences[i]
        color = colors[class_ids[i]]

        # рамка и название обьекта
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, f"{label} {confidence:.2f}", (x, y - 10), font, 2, color, 2)

# Показ резов
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
