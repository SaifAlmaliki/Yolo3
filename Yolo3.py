import cv2
import argparse
import numpy as np

# parser: to run our code from CMD ==> (pthon yolo3.py --image dog.jpg --config yolov3.cfg --weights yolov3.weights --classes labels_v3.txt)
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--image',    required=True, help='path to input image')
parser.add_argument('-c', '--config',   required=True, help='path to congig file')
parser.add_argument('-w', '--weights',  required=True, help='path to yolov3.weight file')
parser.add_argument('-cl', '--classes', required=True, help='path to labels_v3 file')

args = parser.parse_args()


# loop through all layers and return the output layer
def get_output_layers(net):
    layerNames = net.getLayerNames()
    outputLayers = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return outputLayers

# drow Rectangle around the object
def draw_rectangle(myImage, cid, confidence, x1, y1, x_plus_width, y_plus_height):
    label = str(classes[cid])
    color = COLORS[cid]   # each object will surrounded with different color accoring to labels file
    x2 = x_plus_width
    y2 = y_plus_height
    cv2.rectangle(myImage, (x1,y1), (x2,y2), color, 2)
    cv2.putText(myImage, label, (x1-10, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

myImage = cv2.imread(args.image)
width  = myImage.shape[1]
height = myImage.shape[0]
scale = 0.00392

# Load our classes from (_v3.txt) file and read all classes
classes = None
with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]


COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# Load weihts and construct graph
net = cv2.dnn.readNet(args.weights, args.config)

# Create a 4D blob from a Frame
# (416, 416): network size
blob = cv2.dnn.blobFromImage(myImage, scale,  (416,416) , (0,0,0), True, crop=False)
net.setInput(blob)
myOutput = net.forward(get_output_layers(net))

class_ids = []
confidences = []
boxes = []

conf_threshold = 0.5
nms_threshold = 0.4

for out in myOutput:
    for detection in out:
        scores     = detection[5:]
        cid        = np.argmax(scores)
        confidence = scores[cid]
        if confidence > 0.5:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w        = int(detection[2] * width)
            h        = int(detection[3] * height)
            
            x        = center_x - w / 2
            y        = center_y - h / 2
            
            class_ids.append(cid)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])


indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
for i in indices:
    i = i[0]
    box = boxes[i]
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    draw_rectangle(myImage, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))


cv2.imshow('Object Detection', myImage)
cv2.waitKey()
cv2.imwrite("Detction-Result.jpg", myImage)
if(cv2.waitKey() >= 0):
    cv2.destroyAllWindows()