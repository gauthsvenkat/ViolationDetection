import csv
from glob import glob
import os

paths=glob('annotations/*.csv')
tcast = lambda x: int(float(x))

for path in paths:
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