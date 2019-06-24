import json
import csv
from random import Random
import argparse
from shutil import copyfile

f = open('dataset.json', 'r')
annos = json.load(f)

split = 0.9

Random(4).shuffle(annos)

train_annos = annos[:int(len(annos)*split)]
valid_annos = annos[int(len(annos)*split):]

for anno in train_annos:

	wfile = open('train/annotations/'+anno['image'].replace(' ', '')+'.csv', 'w', newline='')
	writer = csv.writer(wfile, quotechar="'")
	copyfile('pic_folder/'+anno['image'], 'train/images/'+anno['image'].replace(' ', ''))

	if anno['tags']:
		to_write = ['name','color','type','x','y','w','h']
		writer.writerow(to_write)

		for tag in anno['tags']:
			to_write = ['\"'+str(tag['name'])+'\"',
						'\"'+str(tag['color'])+'\"',
						'\"'+str(tag['type'])+'\"',
						'\"'+str(tag['pos']['x'])+'\"',
						'\"'+str(tag['pos']['y'])+'\"',
						'\"'+str(tag['pos']['w'])+'\"',
						'\"'+str(tag['pos']['h'])+'\"']
			writer.writerow(to_write)
	wfile.close()

for anno in valid_annos:

	wfile = open('valid/annotations/'+anno['image'].replace(' ', '')+'.csv', 'w', newline='')
	writer = csv.writer(wfile, quotechar="'")
	copyfile('pic_folder/'+anno['image'], 'valid/images/'+anno['image'].replace(' ', ''))

	if anno['tags']:
		to_write = ['name','color','type','x','y','w','h']
		writer.writerow(to_write)

		for tag in anno['tags']:
			to_write = ['\"'+str(tag['name'])+'\"',
						'\"'+str(tag['color'])+'\"',
						'\"'+str(tag['type'])+'\"',
						'\"'+str(tag['pos']['x'])+'\"',
						'\"'+str(tag['pos']['y'])+'\"',
						'\"'+str(tag['pos']['w'])+'\"',
						'\"'+str(tag['pos']['h'])+'\"']
			writer.writerow(to_write)
	wfile.close()

