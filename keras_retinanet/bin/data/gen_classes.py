import csv
from glob import glob
import os

get_cname = lambda name: name.lower().replace(' ', '') if name.lower() != 'scooter'.lower() else 'bike'
path = os.path.join('train', 'annotations', '*.csv')

files = glob(path)
class_name = []

#import ipdb; ipdb.set_trace()

for file in files:

    if os.path.getsize(file) == 0:
        continue

    with open(file) as f:
        data = csv.reader(f)
        next(data)
        for row in data:
            if get_cname(row[0]) not in class_name:
                class_name.append(get_cname(row[0]))
        f.close()

with open('classes.csv', 'w', newline='') as f:

    writer = csv.writer(f)
    for i, name in enumerate(class_name):
        writer.writerow([name, i])

f.close()