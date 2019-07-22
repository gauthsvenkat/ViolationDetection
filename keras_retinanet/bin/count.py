import keras
import sys

import cv2
import os
import numpy as np
import time
import csv
import argparse
from sort import Sort

import random

import tensorflow as tf


if __name__ == "__main__" and __package__ is None:
	sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
	import keras_retinanet.bin  # noqa: F401
	__package__ = "keras_retinanet.bin"

from .. import models
from ..utils.image import read_image_bgr, preprocess_image, resize_image
from ..utils.visualization import draw_box, draw_caption
from ..utils.colors import label_color

def get_session():
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	return tf.Session(config=config)
keras.backend.tensorflow_backend.set_session(get_session())

def set_colors(length, seed=2506):
	random.seed(seed)
	return [(random.randint(0,255), random.randint(0,255), random.randint(0,255)) for i in range(length)]

def get_labels(path):
	labels = {}

	with open(path,'r') as f:
		data=csv.reader(f)
		for row in data:
			labels[int(row[1])] = row[0]
	f.close()

	return labels

def get_trackers(class_ids):
	trackers = {}
	for i in classes:
		trackers[i] = Sort()
	return trackers

def intersect(A,B,C,D):
	return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A,B,C):
	return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

line = []
def get_coordinate(event,x,y,flags,param):
	if event == cv2.EVENT_LBUTTONDOWN:
		cv2.circle(first_frame, (x,y), 10, (0,0,255), 5)
		line.append((x,y))

def show_frame_and_get_line(first_frame):
	while True:
		cv2.imshow('Frame', first_frame)
		if cv2.waitKey(20) & 0xFF == ord('c'):
			cv2.destroyWindow('Frame')
			break

def draw_counter(img, counter, classes, overall=False):

	if overall:
		text = "{}:{}".format('Vehicles', sum(counter))
		cv2.putText(img, text, (100, 1400), 0, 3, (0, 0, 0), 4)
		cv2.putText(img, text, (100, 1400), 0, 3, (255, 255, 255), 2)

	else:
		for i, c in enumerate(counter):
			text = "{}:{}".format(classes[i], counter[i])
			cv2.putText(img, text, (100, 1400 - 100*i), 0, 3, (0, 0, 0), 4)
			cv2.putText(img, text, (100, 1400 - 100*i), 0, 3, (255, 255, 255), 2)


parser = argparse.ArgumentParser(description='Count vehicles on videos')
parser.add_argument('input_path', type=str, help='Full input path to video')
parser.add_argument('-o','--output_path', type=str, help='Full output path to predictions')
parser.add_argument('-m', '--model_path', type=str, default='snapshots/inference_model.h5', help='Full path to trained model')
parser.add_argument('-c', '--class_path', type=str, default='data/classes.csv', help='Path to the classes csv file')
parser.add_argument('-b', '--backbone', type=str, default='resnet152', help='Backbone name')
parser.add_argument('--violation_save_location', type=str, default='violators/', help='Folder to store violations')
parser.add_argument('--count_overall', action='store_true', help='specific count or overall count')
parser.add_argument('--conf', type=float, default=0.9, help='Filter out predictions lesser than this confidence')
parser.add_argument('--min_side', type=int, default=800)
parser.add_argument('--max_side', type=int, default=1333)

args=parser.parse_args()

cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Frame', args.max_side, args.min_side)
cv2.setMouseCallback('Frame', get_coordinate)

cap = cv2.VideoCapture(args.input_path)
fps = cap.get(cv2.CAP_PROP_FPS)
ret, first_frame = cap.read()
model = models.load_model(args.model_path, backbone_name=args.backbone)
classes = get_labels(args[class_path])
class_color = set_colors(len(classes))
writer = None
trackers = get_trackers(classes)
memory = {}
counter = [0] * len(classes)
frame_count = 1
show_frame_and_get_line(first_frame)

cv2.namedWindow('Predictions', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Predictions', args.max_side,args.min_side)

if args.output_path:
	fourcc = cv2.VideoWriter_fourcc(*"MP4V")
	writer = cv2.VideoWriter(args.output_path, fourcc, fps, (first_frame.shape[1], first_frame.shape[0]), True)

while ret:

	ret, frame = cap.read()
	draw = frame.copy()
	frame_tensor = frame.copy()
	previous = memory.copy()
	memory = {}

	frame_tensor = preprocess_image(frame_tensor)
	frame_tensor, scale = resize_image(frame_tensor, min_side=args.min_side, max_side=args.max_side)

	start = time.time()
	boxes, scores, labels = model.predict_on_batch(np.expand_dims(frame_tensor, axis=0))
	print("processing time: ", time.time() - start)


	boxes, scores, labels = boxes[0][scores[0]>args.conf], np.expand_dims(scores[0][scores[0]>args.conf], axis=1), labels[0][scores[0]>args.conf]
	boxes /= scale

	dets = np.append(boxes,scores, axis=1)

	cv2.line(draw, line[0], line[1], (0,255,255), 5)

	for class_id in classes:
		class_dets = dets[labels==class_id]
	
		if class_dets.size != 0:
			tracks = trackers[class_id].update(class_dets)
		else :
			tracks = []

		for box in tracks:
			memory[classes[class_id]+str(box[4])] = box[0:4]

			if classes[class_id]+str(box[4]) in previous:

				prev_box = previous[classes[class_id]+str(box[4])]
				(x2, y2) = (int(prev_box[0]), int(prev_box[1]))
				(w2, h2) = (int(prev_box[2]), int(prev_box[3]))
				p0 = (int(box[0] + (box[2]-box[0])/2), int(box[1] + (box[3]-box[1])/2))
				p1 = (int(x2 + (w2-x2)/2), int(y2 + (h2-y2)/2))
				cv2.line(draw, p0, p1, (255,255,255), 3)

				if intersect(p0, p1, line[0], line[1]):
					if class_id == 2:
						b = box.astype(int)
						roi = frame[b[1]:b[3], b[0]:b[2]]
						cv2.imwrite(args.violation_save_location+str(frame_count)+'.jpg', roi)

					counter[class_id]+=1

			b = box.astype(int)
			draw_box(draw, b, color=class_color[class_id])
			
			caption = "{}".format(classes[class_id])
			draw_caption(draw, b, caption)

			draw_counter(draw, counter, classes, args.count_overall)

	if writer:
		writer.write(draw)

	cv2.imshow('Predictions', draw)
	if cv2.waitKey(10) == ord('q'):
		cv2.destroyAllWindows()
		cap.release()
		
		if writer:
			writer.release()

		break

	frame_count+=1
		