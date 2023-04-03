from ultralytics import YOLO 
import cv2
import math
import base64
import numpy as np

def detect(Base64String):  # function for inference
    Base64code =Base64String.encode()  # convert incoming string 
    nparr = np.frombuffer(base64.b64decode(Base64code), np.uint8) # create np array by using the specified buffer. 
    im1 = cv2.imdecode(nparr,cv2.IMREAD_COLOR) # no save to disk, read image data from a memory cache and convert it into image format
    model = YOLO("box_vs_bag/best.pt")  # weight file of the model we trained
    classNames = ["bag","box"]  # names of detectable classes
    results = model.predict(source=im1)  # perform the detection 
    data={"objectTypes":[],"confidences":[],"coordinates":[]}  # dictionary to be filled for multi-detection
    for r in results:  # this block is used for iterating over all detections, each detection will be processed individually
        boxes = r.boxes  
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # cordinates are taken
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # coordinates are converted to integar format
            w, h = x2 - x1, y2 - y1  # width and height values ​​are calculated
            conf = math.ceil((box.conf[0] * 100)) / 100  # confidence is found
            cls = int(box.cls[0])  # since we have two classes of object types it will take one of the values ​​0-1
            data["objectTypes"] += [classNames[cls]] # every detection will be saved and returned in single json as a result
            data["confidences"] += [conf]
            data["coordinates"] += [[x1, y1, x2, y2]]
    return data
