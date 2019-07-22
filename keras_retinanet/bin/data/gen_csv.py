#Script to generate the train.csv and valid.csv for training

import csv
from glob import glob
import os

tcast = lambda x: int(round(float(x)))
anno2img = lambda anno: os.path.join(anno.replace('annotations', 'images').replace('.csv', '')) #To convert anno location to image location (full path)
class_name = lambda name: name.lower().replace(' ', '')

def driver(mode=None):

	anno_path = os.path.join(mode,'annotations','*.csv')
	annos = glob(anno_path)

	w = open(mode+'.csv','w', newline='')
	writer = csv.writer(w)

	for anno in annos:

		if os.path.getsize(anno) == 0:
			writer.writerow([anno2img(anno), '','','','','']) #If annotation file is empty, no object
			continue

		r = open(anno)
		data = csv.reader(r)
		next(data)

		
		for row in data:
			try:
				to_write = [anno2img(anno), #Image name (full path)
							tcast(row[3]), #x
							tcast(row[4]), #y
							tcast(float(row[3])+float(row[5])), #x+w
							tcast(float(row[4])+float(row[6])), #y+h
							class_name(row[0])]
				writer.writerow(to_write)
			except ValueError:
				print('Cannot convert ', row)

		r.close()
	w.close()

for mode in ['train', 'valid']:
	driver(mode)