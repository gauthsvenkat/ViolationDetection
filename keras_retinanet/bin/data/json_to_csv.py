import json
import csv
from random import Random
import argparse
import os, shutil
from glob import glob

parser = argparse.ArgumentParser(description='Convert json file to csv. (Make sure you run this code in the directory you want to convert to)')
parser.add_argument('json_file', type=str, help='JSON file to convert')
parser.add_argument('pic_folder', type=str, help='Folder containing all the pictures (that are in JSON)')
parser.add_argument('--seed', type=int, default=4 ,help='Seed for random shuffle')
parser.add_argument('--proportion', type=float, default=0.9, help='Proportion split for train and valid')

args = parser.parse_args()


def driver(annos, split=None, src=None):

	assert os.path.exists(src), src + " Path does not exists"

	for anno in annos:

		wfile = open(split+'/annotations/'+anno['image'].replace(' ', '')+'.csv', 'w', newline='')
		writer = csv.writer(wfile, quotechar="'")
		shutil.copyfile(src+'/'+anno['image'], split+'/images/'+anno['image'].replace(' ', ''))

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


f = open(args.json_file, 'r')
annos = json.load(f)

Random(args.seed).shuffle(annos)

anno_split = {
		'train':annos[:int(len(annos)*args.proportion)],
		'valid':annos[int(len(annos)*args.proportion):] 
		}

for path in ['train', 'valid']:
	if os.path.exists(path+'/annotations/'):
		files = glob(path+'/annotations/*.csv')
		[os.remove(f) for f in files]
	else:
		os.makedirs(path+'/annotations/')

	if os.path.exists(path+'/images'):
		files = glob(path+'/images/*.jpg')
		[os.remove(f) for f in files]
	else:
		os.makedirs(path+'/images/')

	driver(anno_split[path], split=path, src=args.pic_folder)










