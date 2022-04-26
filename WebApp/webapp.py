from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import os
from tensorflow.keras.models import Sequential, Model, load_model
from itertools import combinations
import math

app = Flask(__name__)
faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
faceMaskModel = load_model("FaceMaskModel.model")

MODEL_PATH = "yolo"
labelsPath = os.path.sep.join([MODEL_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")
weightsPath = os.path.sep.join([MODEL_PATH, "yolov3.weights"])
configPath = os.path.sep.join([MODEL_PATH, "yolov3.cfg"])
net = cv2.dnn.readNet(weightsPath, configPath)
classes = []
with open(labelsPath, "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))


def MaskDetector(img):
    preds = faceMaskModel.predict(np.expand_dims(img, axis=0))[0]
    if round(preds[0]) == 0:
        return 1
    else:
        return 0


def calculateDistance(x1, y1, x2, y2):
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist


def detectSocialDistance(img):
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    center_points = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and class_id == 0:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h, center_x, center_y])
                center_points.append([center_x, center_y])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    combination_points = list(combinations(center_points, 2))

    result = "Not Found"
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h, box_center_x, box_center_y = boxes[i]
            cv2.circle(img, (center_x, center_y), 3, (0, 0, 255), cv2.FILLED)

            for points in combination_points:
                center_x, center_y = points[0]
                prev_center_x, prev_center_y = points[1]
                euclidean_distance = calculateDistance(center_x, center_y, prev_center_x, prev_center_y)

                width_of_3_tiles = 335
                if width_of_3_tiles > euclidean_distance > 150:
                    if box_center_x == center_x or box_center_y == center_y:
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        result = "No Social Distance"
                if euclidean_distance > width_of_3_tiles and euclidean_distance > 150:
                    if box_center_x == center_x or box_center_y == center_y:
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        result = "Social Distance is there."
    return result


def prediction(path):
    testImage = cv2.imread(path)
    print("Image loaded")
    grayscaled_img = cv2.cvtColor(testImage, cv2.COLOR_BGR2GRAY)
    face_coordinates = faceDetector.detectMultiScale(grayscaled_img)
    for (x, y, w, h) in face_coordinates:
        image = testImage[y:y + h, x:x + w]
        image = cv2.resize(image, (224, 224))
        result = MaskDetector(image)
        if result == 1:
            mask = "Mask On"
        else:
            mask = "No Mask"
    img = cv2.resize(testImage, (700, 700))
    distance = detectSocialDistance(img)
    return mask, distance


@app.route("/", methods=["POST", "GET"])
def predict():
    if request.method == 'GET':
        return render_template('index.html')
    elif request.method == "POST":
        image = request.files.get('imgup')
        image.save('./' + secure_filename(image.filename))
        name, score = prediction(image.filename)
        kwargs = {'name': name, 'score': score}
        return render_template('index2.html', **kwargs)


if __name__ == '__main__':
    app.run(host="0.0.0.0")
