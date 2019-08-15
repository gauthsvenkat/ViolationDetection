import keras
import sys

import os
import numpy as np
import time
import argparse
import cv2

from sort import Sort

#Add project to PATH
if __name__ == "__main__" and __package__ is None:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
        import keras_retinanet.bin  # noqa: F401
        __package__ = "keras_retinanet.bin"

from .. import models
from ..utils.image import read_image_bgr, preprocess_image, resize_image
from ..utils.visualization import draw_box, draw_caption
from ..utils.colors import label_color
from ..utils.custom_utils import get_session, set_colors, get_labels, intersect, draw_counter, count_write


keras.backend.tensorflow_backend.set_session(get_session())

#This function initializes trackers for each vehicle class
def get_trackers(class_ids):
        trackers = {}
        for i in classes:
                trackers[i] = Sort()
        return trackers

line = [] #This list stores a tuple of x,y coordinates. len(line) should always be even.

#This function gets called when a mouse event is encountered. Stores the x,y coordinates of a left click into a tuple.
def get_coordinate(event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
                cv2.circle(first_frame, (x,y), 10, (0,0,255), 5)
                line.append((x,y))

#This function displays the first frame to get all the lines in the video.
def show_frame_and_get_line(first_frame):
        while True:
                cv2.imshow('Frame', first_frame)
                if cv2.waitKey(20) & 0xFF == ord('c'):
                        cv2.destroyWindow('Frame')
                        break

def is_red(frame, pixel_location, thresh=170):
    return frame[pixel_location[1], pixel_location[0], 2] > thresh

parser = argparse.ArgumentParser(description='Count vehicles on videos')
parser.add_argument('input_path', type=str, help='Full input path to video')
parser.add_argument('-o','--output_path', type=str, help='Full output path to predictions')
parser.add_argument('-m', '--model_path', type=str, default='snapshots/inference_model.h5', help='Full path to trained model')
parser.add_argument('-c', '--class_path', type=str, default='data/classes.csv', help='Path to the classes csv file')
parser.add_argument('-b', '--backbone', type=str, default='resnet152', help='Backbone name')
parser.add_argument('--count_save', type=str, default='output/count.csv')
parser.add_argument('--violation_save_location', type=str, default='output/violators/', help='Folder to store violations')
parser.add_argument('--count_overall', action='store_true', help='specific count or overall count')
parser.add_argument('--conf', type=float, default=0.9, help='Filter out predictions lesser than this confidence')
parser.add_argument('--min_side', type=int, default=800)
parser.add_argument('--max_side', type=int, default=1333)
parser.add_argument('--bad_classes', nargs='+', help='List of classes (according to data/classes.csv) that are violators')
parser.add_argument('-v', '--verbose', action='store_true', help='Modify verbosity (Will display additional information in the terminal)')

args=parser.parse_args()

#Opens a window to display the first frame and resizes it to min_side and max_side
cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Frame', args.max_side, args.min_side)

cv2.setMouseCallback('Frame', get_coordinate) #Callback function to record mouse events

cap = cv2.VideoCapture(args.input_path) #Initialize videocapture objects
fps = cap.get(cv2.CAP_PROP_FPS) #Get FPS of the video
ret, first_frame = cap.read() #Read first frame for getting lines
model = models.load_model(args.model_path, backbone_name=args.backbone) #Load the inference model
classes = get_labels(args.class_path) #Dictionary where key is the class name and value is the class_id
class_color = set_colors(len(classes)) #Set some random colors for each class
writer = None #Initialize VideoWriter object
trackers = get_trackers(classes) #Initialize all the trackers
memory = {} #Dictionary to store objects detected in previous frames where key is the class+object_id and value is the 4 bbox coordinates
counter = {} #A dict that contains the unique counts of all the classes for all the lines
frame_count = 1 #Initialize frame count
args.bad_classes = [2, 10] if args.bad_classes is None else args.bad_classes #Violator classes defaults to two wheeler w/o helmet and triples

show_frame_and_get_line(first_frame) #Get coordinates of lines and signal location if given

signal_pixel_location = line.pop() if len(line)%2 else None #Last element of line is the signal pixel location

#Initialize the counter for all the lines
for i in range(len(line)//2):
        counter['line'+str(i)] = [0] * len(classes)

#Open the predictions window and resize it
cv2.namedWindow('Predictions', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Predictions', args.max_side,args.min_side)

#If output_path is not None, then initialize VideoWriter object to write predictions to MP4
if args.output_path:
        fourcc = cv2.VideoWriter_fourcc(*"MP4V")
        writer = cv2.VideoWriter(args.output_path, fourcc, fps, (first_frame.shape[1], first_frame.shape[0]), True)

#Keep looping while ret is True
while ret:

        start = time.time()
        ret, frame = cap.read()
        draw = frame.copy()
        frame_tensor = frame.copy()
        previous = memory.copy()
        memory = {}
        red_signal_status = is_red(frame, signal_pixel_location) if signal_pixel_location else False #If signal location is given then check if it is red else False

        frame_tensor = preprocess_image(frame_tensor) #Convert frame to tensor
        frame_tensor, scale = resize_image(frame_tensor, min_side=args.min_side, max_side=args.max_side)

        boxes, scores, labels = model.predict_on_batch(np.expand_dims(frame_tensor, axis=0)) #Get the predictions from the model

        #Convert the predictions to less cryptic shit and weed out predictions lower than args.conf
        boxes, scores, labels = boxes[0][scores[0]>args.conf], np.expand_dims(scores[0][scores[0]>args.conf], axis=1), labels[0][scores[0]>args.conf]
        boxes /= scale #Adjust to scale

        dets = np.append(boxes,scores, axis=1) #dets is a (n, 5) array where first 4 columns are bbox coordinates and 5th column is confidence

        [cv2.line(draw, line[x], line[x+1], (0,255,255), 5) for x in range(0, len(line), 2)] #Draw all the lines that were recorded in the beginning

        for class_id in classes: #For each class 
                class_dets = dets[labels==class_id] #Only consider a particular class
        
                if class_dets.size != 0:
                        tracks = trackers[class_id].update(class_dets) #Update trackers if no detections
                else :
                        tracks = [] #Else nevermind

                for box in tracks: #For each box in tracks
                        memory[classes[class_id]+str(box[4])] = box[0:4] #Store the bbox in memory (to compare in the future)

                        if classes[class_id]+str(box[4]) in previous: #Check if this object was in the previous frame

                                prev_box = previous[classes[class_id]+str(box[4])] #Get the bbox coordinates of the object in the previous frame
                                (x2, y2) = (int(prev_box[0]), int(prev_box[1]))
                                (w2, h2) = (int(prev_box[2]), int(prev_box[3]))
                                p0 = (int(box[0] + (box[2]-box[0])/2), int(box[1] + (box[3]-box[1])/2)) #Mid point of object in the previous frame
                                p1 = (int(x2 + (w2-x2)/2), int(y2 + (h2-y2)/2)) #Mid point of object in the current frame
                                cv2.line(draw, p0, p1, (255,255,255), 3) #Draw a trajectory line of the object 

                                #Check if the object intersects with any of the lines

                                for i in range(0, len(line), 2):
                                        if intersect(p0, p1, line[i], line[i+1]):
                                                counter['line'+str(i//2)][class_id]+=1
                                                if class_id in args.bad_classes or red_signal_status: #If violation classes (Triples, withouthelment and signal skippers)
                                                        b = box.astype(int)
                                                        roi = frame[b[1]:b[3], b[0]:b[2]]
                                                        cv2.imwrite(args.violation_save_location+str(frame_count)+'.jpg', roi) #Get a snap of the violators

                        b = box.astype(int) #Convert bbox to int
                        draw_box(draw, b, class_color[class_id])
                        
                        caption = "{}".format(classes[class_id])
                        draw_caption(draw, b, caption, class_color[class_id])

                        draw_counter(draw, counter, classes, class_color, args.count_overall)

        if writer:
                writer.write(draw)

        cv2.imshow('Predictions', draw)
        if cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows()
                cap.release()
                
                if writer:
                        writer.release()

                count_write(counter, classes, args.count_save) #Write the counts to a csv file

                break

        frame_count+=1
        
        if args.verbose:
                print("processing time: ", time.time() - start, "red signal status: ", red_signal_status)
