import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time
import csv
import argparse

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

def get_session():
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	return tf.Session(config=config)
keras.backend.tensorflow_backend.set_session(get_session())



def get_labels(path):
	labels = {}

	with open(path,'r') as f:
		data=csv.reader(f)
		for row in data:
			labels[int(row[1])] = row[0]
	f.close()

	return labels

parser = argparse.ArgumentParser(description='Detect vehicles on videos')
parser.add_argument('-i','--input_path', type=str, help='Full input path to video')
parser.add_argument('-o','--output_path', type=str, help='Full output path to predictions')
parser.add_argument('-m', '--model_path', type=str, default='inference_model.h5', help='Full path to trained model')
parser.add_argument('-b', '--backbone', type=str, default='resnet50', help='Backbone name')
parser.add_argument('--min_side', type=int, default=800)
parser.add_argument('--max_side', type=int, default=1333)

args=parser.parse_args()

cap = cv2.VideoCapture(args.input_path)
ret, frame = cap.read()
model = models.load_model(args.model_path, backbone_name=args.backbone)
labels_to_names = get_labels('data/classes.csv')
writer = None

cv2.namedWindow('Predictions', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Predictions', args.max_side,args.min_side)

if args.output_path:
	fourcc = cv2.VideoWriter_fourcc(*"MP4V")
	writer = cv2.VideoWriter(args.output_path, fourcc, 30, (frame.shape[1], frame.shape[0]), True)

while ret:

	ret, frame = cap.read()

	draw = frame.copy()

	frame = preprocess_image(frame)
	frame, scale = resize_image(frame, min_side=args.min_side, max_side=args.max_side)

	start = time.time()
	boxes, scores, labels = model.predict_on_batch(np.expand_dims(frame, axis=0))
	print("processing time: ", time.time() - start)

	boxes /= scale

	for box, score, label in zip(boxes[0], scores[0], labels[0]):
		# scores are sorted so we can break
		if score < 0.5:
			break
			
		color = label_color(label)
		
		b = box.astype(int)
		draw_box(draw, b, color=color)
		
		caption = "{} {:.3f}".format(labels_to_names[label], score)
		draw_caption(draw, b, caption)

	if writer:
		writer.write(draw)

	cv2.imshow('Predictions', draw)
	if cv2.waitKey(25) == ord('q'):
		cv2.destroyAllWindows()
		cap.release()
		
		if writer:
			writer.release()

		break
		