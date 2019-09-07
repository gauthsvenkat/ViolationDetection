import json
import ntpath
import xml.etree.ElementTree as ET
import csv
from random import Random
import argparse
import os, shutil
from glob import glob
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Convert xml files to csv. (Make sure you run this code in the directory you want to convert to)')
parser.add_argument('xml_folder', type=str, help='xml files to convert')
parser.add_argument('pic_folder', type=str, help='Folder containing all the pictures')
parser.add_argument('--seed', type=int, default=4 ,help='Seed for random shuffle')
parser.add_argument('--proportion', type=float, default=0.9, help='Proportion split for train and valid')

args = parser.parse_args()

def driver(annos, split=None, xml_src=None, img_src=None):

    assert os.path.exists(img_src), img_src + " Path does not exists"

    for anno in tqdm(annos):
        tree = ET.parse(os.path.join(xml_src,anno))
        root = tree.getroot()
        
        img_file = anno.replace('.xml', '.jpg')
        csv_file = anno.replace('.xml', '.csv')

        wfile = open(split+'/annotations/'+csv_file, 'w', newline='')
        writer = csv.writer(wfile, quotechar="'")
        try:
            shutil.copyfile(os.path.join(img_src,img_file), split+'/images/'+img_file)
            to_write = ['name','color','type','x','y','w','h']
            writer.writerow(to_write)

            for member in root.findall('object'):
                to_write = ['\"'+member[0].text+'\"',
                            '\"'+'unspecified'+'\"',
                            '\"'+'2dbbox'+'\"',
                            '\"'+member[4][0].text+'\"', #x
                            '\"'+member[4][1].text+'\"', #y
                            '\"'+str(int(member[4][2].text)-int(member[4][0].text))+'\"', #w
                            '\"'+str(int(member[4][3].text)-int(member[4][1].text))+'\"'] #h
                writer.writerow(to_write)
            wfile.close()
        except:
            print(anno, " Not copied")

annos = [ntpath.basename(f) for f in glob(os.path.join(args.xml_folder, '*.xml'))]

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

    driver(anno_split[path], split=path, xml_src=args.xml_folder, img_src=args.pic_folder)










