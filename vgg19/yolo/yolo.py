import cv2
import numpy as np

# Load YOLO model

cam = cv2.VideoCapture(0)

net = cv2.dnn.readNet("models/yolov3-tiny.weights", "configs/yolov3-tiny.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

while True:
    # Load an image
    ret, img = cam.read()
    if not ret:
        print("unable to record frame")
    
    height, width, channels = img.shape

    # Prepare the image for YOLO
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    # Run the forward pass
    outs = net.forward(output_layers)

    # Processing the output
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:] # center x, center y, width,  height, object confidence score, class confidence scores...
            class_id = np.argmax(scores)
            class_confidence = scores[class_id]
            object_confidence = detection[4]
            if object_confidence > 0.5:
                # Get the coordinates for the bounding box
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(class_confidence))
                class_ids.append(class_id)

    # Apply non-max suppression to remove overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw the bounding boxes
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)  # Green box
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.imshow("Camera Feed", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # Show the image
cam.release()
cv2.destroyAllWindows()