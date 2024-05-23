import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
import multiprocessing
from pathlib import Path

rename = {
    'Bear': ['Bear', 'Brown bear'],
    'Domestic_cat': ['Cat'],
    'Frisbee': ['Flying disc'],
    'Giant_panda': ['Panda'],
    'Red_panda': ['Red panda'],
    'Sofa': ['Couch'],
    'Turtle': ['Turtle', 'Sea turtle'],
    }

need = [i.capitalize() for i in ['airplane', 'antelope', 'ball', 'bear', 'bicycle', 'bird', 'bus', 'car', 'cattle', 'dog', 'domestic_cat', 'elephant', 'fox', 'frisbee', 'giant_panda', 'hamster', 'horse', 'lion', 'lizard', 'monkey', 'motorcycle', 'person', 'rabbit', 'red_panda', 'sheep', 'skateboard', 'snake', 'sofa', 'squirrel', 'tiger', 'train', 'turtle', 'watercraft', 'whale', 'zebra']]

max_per_class = 3000
min_quality = 100

df = pd.read_csv('~/fiftyone/open-images-v6/train/metadata/classes.csv', names=['LabelName', 'DisplayName'])
metadata = dict(zip(df.LabelName, df.DisplayName))

images_root_dir = '~/fiftyone/open-images-v6/train/data/'
anno = pd.read_csv('~/fiftyone/open-images-v6/train/labels/detections.csv')
DEST_DIR = "./vidvrd_queries"

reverse_rename = {
    'Cat': 'Domestic_cat',
    'Flying disc': 'Frisbee',
    'Panda': 'Giant_panda',
    'Red panda': 'Red_panda',
    'Couch': 'Sofa',
}

def func(row_num):
    row = anno.iloc[row_num[0]]

    image_name = row.ImageID + '.jpg'
    image_path = os.path.join(images_root_dir, image_name)
    try:
        object_name = metadata[row.LabelName]

        if object_name in ['Turtle', 'Sea turtle']:
            object_name = 'Turtle'
        elif object_name in ['Bear', 'Brown bear']:
            object_name = 'Bear'
        if object_name in reverse_rename:
            object_name = reverse_rename[object_name]

        if (os.path.exists(f'{DEST_DIR}/{object_name}') and len(os.listdir(f'{DEST_DIR}/{object_name}')) > max_per_class) or (object_name not in need):
            return

        dest_image_path = os.path.join(DEST_DIR, object_name, image_name)
        Path(f"{DEST_DIR}/{object_name}").mkdir(parents=True, exist_ok=True)

        image = Image.open(image_path).convert('RGB')
        w, h = image.size

        bbox = (row.XMin * w, row.YMin * h, row.XMax * w, row.YMax * h)
        cropped_image = image.crop(bbox)

        if cropped_image.size[0] * cropped_image.size[1] < min_quality:
            return
        
        cropped_image.save(dest_image_path)

    except Exception as e:
        # print(e)
        return

with multiprocessing.Pool(processes = 8) as p:
    max_ = len(anno)
    params = []
    for i in range(max_):
        params.append([i])
    with tqdm(total=max_) as pbar:
        for _, _ in tqdm(enumerate(p.imap_unordered(func, params))):
            pbar.update()
