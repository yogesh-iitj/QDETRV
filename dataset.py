import cv2
import pandas as pd
import torch
import numpy as np
# import utils
from utils import utils
import random

class VidOR1s():
    def __init__(self, root_dir, df, transform=None):
        self.root_dir = root_dir
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.df.iloc[idx]
        ndf = self.df[(self.df['imageid'] == row.imageid) & (self.df['classid'] == row.classid)]
        class_q = row.classid.split('_')[0]
        image_q = cv2.imread(f'{self.root_dir}/classes/images/{class_q}_{random.randint(0,7)}.jpg', cv2.IMREAD_COLOR)
        image_q = cv2.cvtColor(image_q, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_q /= 255.0
        image_t = cv2.imread(f'{self.root_dir}/src/3264/{row.imageid}.jpg', cv2.IMREAD_COLOR)
        image_t = cv2.cvtColor(image_t, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_t /= 255.0
        # image_q = Image.open(f'{self.root_dir}/classes/images/{row.classid}.jpg')
        # image_t = Image.open(f'{self.root_dir}/src/3264/{row.imageid}.jpg')
        
        # width, height = image_t.size
        height, width, _ = image_t.shape

        # conveerting boxes into coco format
        boxes = ndf[['lx', 'ty', 'rx', 'by']].values
        area = []
        for box in boxes:
            area.append(utils.calculate_bbox_area(box, height, width))

        #converting boxes into coco format
        for j, box in enumerate(boxes):
            x_c = 0.5*(box[0] + box[2])
            y_c = 0.5*(box[1] + box[3])
            w = box[2] - box[0]
            h = box[3] - box[1]
            boxes[j] = [x_c, y_c, w, h]

        labels =  np.zeros(len(boxes), dtype=np.int32)

        if self.transform:
            sample = {
                'image': image_q,
                'image_t': image_t,
                'bboxes': boxes,
                'labels': labels
            }
            sample = self.transform(**sample)
            image_t = sample['image_t']
            image_q = sample['image']
            boxes = sample['bboxes']
            labels = sample['labels']
        
        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target['image_id'] = torch.tensor([idx])
        target['area'] = torch.as_tensor(area, dtype=torch.float32)

        return image_q, image_t, target, width, height, row.classid, row.imageid
    

class VidORnf():
    def __init__(self, root_dir, df, transform=None):
        self.root_dir = root_dir
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.df.loc[idx]
        # ndf = self.df[(self.df['imageid'] == row.imageid) & (self.df['classid'] == row.class_name)]
        image_q_name = random.choice(row['classid'][2:-2].replace("'", "").replace(" ", "").split(','))
        image_q = cv2.imread(f'{self.root_dir}/images/{image_q_name}.jpg', cv2.IMREAD_COLOR)
        image_q = cv2.cvtColor(image_q, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_q /= 255.0
        
        image_t = cv2.imread(f'{self.root_dir}/3264/{row.imageid}.jpg', cv2.IMREAD_COLOR)
        image_t = cv2.cvtColor(image_t, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_t /= 255.0
        # image_q = Image.open(f'{self.root_dir}/classes/images/{row.classid}.jpg')
        # image_t = Image.open(f'{self.root_dir}/src/3264/{row.imageid}.jpg')
        
        # width, height = image_t.size
        height, width, _ = image_t.shape

        # conveerting boxes into coco format
        # boxes = ndf[['lx', 'ty', 'rx', 'by']].values
        n_box = []
        for box in self.df['boxes'][idx][2:-2].split(","):
            box = box.replace("\"", "").replace("'", "").replace(" ", "").split('_')
            b = [float(val) for val in box]
            # print(b)
            n_box.append(b)
        # n_box = np.array(n_box)


        area = []
        for box in n_box:
            area.append(utils.calculate_bbox_area(box, height, width))

        #converting boxes into coco format
        for j, box in enumerate(n_box):
            x_c = 0.5*(box[0] + box[2])
            y_c = 0.5*(box[1] + box[3])
            w = box[2] - box[0]
            h = box[3] - box[1]
            n_box[j] = [x_c, y_c, w, h]

        labels =  np.zeros(len(n_box), dtype=np.int32)

        if self.transform:
            sample = {
                'image': image_q,
                'image_t': image_t,
                'bboxes': n_box,
                'labels': labels
            }
            sample = self.transform(**sample)
            image_t = sample['image_t']
            image_q = sample['image']
            n_box = sample['bboxes']
            labels = sample['labels']
        
        target = {}
        target["boxes"] = torch.as_tensor(n_box, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target['image_id'] = torch.tensor([idx])
        target['area'] = torch.as_tensor(area, dtype=torch.float32)

        return image_q, image_t, target, width, height, row.class_name, row.imageid