#Script to write all the available classes to a csv file

import csv
from glob import glob
import os

get_cname = lambda name: name.lower().replace(' ', '') #Throwaway function to convert name to all lower without space

path = os.path.join('train', 'annotations', '*.csv')

files = glob(path) #Get all the annotation files in path
class_name = []

for file in files:

    if os.path.getsize(file) == 0: #Ignore if annotation file is empty
        continue

    with open(file) as f:
        data = csv.reader(f)
        next(data)
        for row in data:
            if get_cname(row[0]) not in class_name: #Append new class name to csv, else ignore
                class_name.append(get_cname(row[0]))
        f.close()

with open('classes.csv', 'w', newline='') as f: #Write all the class names to a csv file

    writer = csv.writer(f)
    for i, name in enumerate(class_name):
        writer.writerow([name, i])

f.close()