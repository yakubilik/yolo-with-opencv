import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob, os


model = cv2.dnn.readNetFromDarknet("yolov4-obj.cfg","yolov4-obj_last.weights")
layers = model.getLayerNames()
unconnect = model.getUnconnectedOutLayers()
unconnect = unconnect-1

output_layers = []
for i in unconnect:
    output_layers.append(layers[int(i)])


classFile = 'obj.names'
classNames=[]
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
print(classNames)

vid = cv2.VideoCapture(0)
while(True):
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    ret, frame = vid.read()

    img_width = frame.shape[1]
    img_height = frame.shape[0]


    img_blob = cv2.dnn.blobFromImage(frame,1/255,(416,416),swapRB=True)

    model.setInput(img_blob)
    detection_layers = model.forward(output_layers)

    ids_list = []
    boxes_list = []
    confidences_list = []

    for detection_layer in detection_layers:
        for object_detection in detection_layer:
            scores = object_detection[5:]
            predicted_id = np.argmax(scores)
            confidence =scores[predicted_id]


            if confidence > 0.10:

                label = classNames[predicted_id]
                bounding_box = object_detection[0:4] * np.array([img_width,img_height,img_width,img_height])
                (box_center_x, box_center_y ,box_width ,box_height) = bounding_box.astype("int")
                start_x = int(box_center_x- (box_width/2))
                start_y = int(box_center_y - (box_height/2))

                ids_list.append(predicted_id)
                confidences_list.append(float(confidence))
                boxes_list.append([start_x,start_y,int(box_width),int(box_height)])

    max_ids = cv2.dnn.NMSBoxes(boxes_list,confidences_list,0.5,0.4)


    for max_id in max_ids:
        max_class_id=max_id
        box = boxes_list[max_class_id]

        start_x = box[0]
        start_y = box[1]
        box_width = box[2]
        box_height= box[3]

        predicted_id = ids_list[max_class_id]
        label = classNames[predicted_id]
        print(classNames[predicted_id])
        confidence=confidences_list[max_class_id]

        end_x = start_x + box_width
        end_y = start_y+box_height


        cv2.rectangle(frame,(start_x,start_y),(end_x,end_y),(255, 0, 0),2)

        cv2.putText(frame,label,(start_x,start_y-20),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),1,1)


    cv2.imshow('frame', frame)
