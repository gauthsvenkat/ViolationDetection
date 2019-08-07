python json_to_csv.py $1 $2 --proportion 0.98
python shitty_annotations.py
python gen_classes.py
python gen_csv.py

