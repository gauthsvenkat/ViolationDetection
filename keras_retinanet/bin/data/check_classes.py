import ntpath
import xml.etree.ElementTree as ET
import argparse
import os, shutil
from glob import glob

parser = argparse.ArgumentParser(description='Check classes in xml')
parser.add_argument('xml_folder', type=str, help='xml files to check classes')
parser.add_argument('legal_classes', type=str, help='available classes')

args = parser.parse_args()

classes = open(args.legal_classes, 'r').read().splitlines()

annos = [ntpath.basename(f) for f in glob(os.path.join(args.xml_folder, '*.xml'))]

for anno in annos:
	tree = ET.parse(os.path.join(args.xml_folder, anno))
	root = tree.getroot()

	for member in root.findall('object'):
		if member[0].text not in classes:
			print("Unknown class \"{}\" in {}".format(member[0].text, anno))
