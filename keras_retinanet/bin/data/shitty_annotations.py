import csv
from glob import glob
import os

tcast = lambda x: float(str(x))

def driver(path=None):
    
    paths=glob(path+'/annotations/*.csv')
    
    for path in paths:

        if os.path.getsize(path) == 0:
            continue

        with open(path) as f:
            data = csv.reader(f)
            next(data)
            for row in data:
                if row[3] == row[4] == row[5] == row[6]:
                    print("cursed ", path)
                    f.close()
                    os.remove(path)
                    os.remove(path.replace('annotations', 'images').replace('.csv', ''))
                    break

            f.close()

for path in ['train', 'valid']:
    driver(path)