import random
import csv
import cv2
import tensorflow as tf

def get_session():
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	return tf.Session(config=config)

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

def ccw(A,B,C):
	return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def intersect(A,B,C,D):
	return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def draw_counter(img, counter_dict, classes, class_color, overall=False):

	x_orig = 100

	for line_count, (line, counter) in enumerate(counter_dict.items()):
		y_orig = 1400

		if overall:
			text = "{}:{}".format('Vehicles', sum(counter))
			cv2.putText(img, text, (x_orig, y_orig), 0, 1, (255, 255, 255), 2)
			x_orig+=500

		else:
			for i, c in enumerate(counter):
				text = "{}:{}".format(classes[i], counter[i])
				cv2.putText(img, text, (x_orig, y_orig), 0, 1, class_color[i], 2)
				y_orig-=50
			cv2.putText(img, line, (x_orig, y_orig), 0, 1, class_color[i], 2)
			x_orig+=500

def count_write(counter_dict, classes, location):
	with open(location, 'w') as f:
		wr = csv.writer(f)
		for line, counter in counter_dict.items():
			wr.writerow([line])
			for i, count in enumerate(counter):
				wr.writerow([classes[i], count])
			wr.writerow(['Overall', sum(counter)])
			[wr.writerow(['']) for _ in range(2)]
		f.close
