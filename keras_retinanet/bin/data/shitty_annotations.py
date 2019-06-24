import csv
from glob import glob
import os

paths=glob('valid/annotations/*.csv')
tcast = lambda x: int(float(x))

for path in paths:

    if os.path.getsize(path) == 0:
        continue

    with open(path) as f:
        data = csv.reader(f)
        next(data)
        for row in data:
            if float(row[3]) == float(row[4]) == float(row[5]) == float(row[6]):
                print("cursed ", path)
                f.close()
                os.remove(path)
                break


        f.close()

paths=glob('train/annotations/*.csv')
tcast = lambda x: int(float(x))

for path in paths:

    if os.path.getsize(path) == 0:
        continue

    with open(path) as f:
        data = csv.reader(f)
        next(data)
        for row in data:
            if float(row[3]) == float(row[4]) == float(row[5]) == float(row[6]):
                print("cursed ", path)
                f.close()
                os.remove(path)
                break


        f.close()