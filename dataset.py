import cv2
import pandas as pd
import torch
import numpy as np
# import utils
from utils import utils
import random
import os
from torchvision import transforms

def custom_collate_fn(batch):
    images_q, images_t, context_images, targets, widths, heights, class_names, image_ids = zip(*batch)
    
    # Stack images and context images
    images_q = torch.stack(images_q)
    images_t = torch.stack(images_t)
    context_images = torch.stack(context_images)
    
    # Collate targets
    max_num_boxes = max(len(t['boxes']) for t in targets)
    collated_targets = []
    
    for t in targets:
        num_boxes = len(t['boxes'])
        padded_boxes = torch.zeros((max_num_boxes, 4), dtype=torch.float32)
        padded_labels = torch.zeros((max_num_boxes,), dtype=torch.int64)
        padded_area = torch.zeros((max_num_boxes,), dtype=torch.float32)
        
        if num_boxes > 0:
            padded_boxes[:num_boxes] = t['boxes']
            padded_labels[:num_boxes] = t['labels']
            padded_area[:num_boxes] = t['area']
        
        collated_targets.append({
            "boxes": padded_boxes,
            "labels": padded_labels,
            "image_id": t['image_id'],
            "area": padded_area
        })
    
    # Convert lists to tensors
    widths = torch.tensor(widths)
    heights = torch.tensor(heights)
    
    return images_q, images_t, context_images, collated_targets, widths, heights, class_names, image_ids

class VidOR():
    def __init__(self, root_dir, df, transform=None, image_size=(224, 224)):
        self.root_dir = root_dir
        self.df = df
        self.transform = transform
        self.image_size = image_size
        self.resize_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.df.loc[idx]
        image_q_name = random.choice(row['classid'][2:-2].replace("'", "").replace(" ", "").split(','))
        image_q = cv2.imread(f'{self.root_dir}/images/{image_q_name}.jpg', cv2.IMREAD_COLOR)
        image_q = cv2.cvtColor(image_q, cv2.COLOR_BGR2RGB)

        video_name, frame_number = row.imageid.split('_')
        frame_number = int(frame_number)

        # Read target image (current frame)
        image_t = cv2.imread(f'{self.root_dir}/vidor/{row.imageid}.jpg', cv2.IMREAD_COLOR)
        image_t = cv2.cvtColor(image_t, cv2.COLOR_BGR2RGB)

        # Read context frames
        context_images = []
        for offset in range(-2, 3):  # -2, -1, 0, +1, +2
            context_frame_number = frame_number + offset
            context_image_path = f'{self.root_dir}/vidor/{video_name}_{context_frame_number}.jpg'
            if os.path.exists(context_image_path):
                context_image = cv2.imread(context_image_path, cv2.IMREAD_COLOR)
                context_image = cv2.cvtColor(context_image, cv2.COLOR_BGR2RGB)
            else:
                context_image = np.zeros_like(image_t)  # Zero context image for edge cases

            context_images.append(context_image)

        height, width = self.image_size

        n_box = []
        for box in self.df['boxes'][idx][2:-2].split(","):
            box = box.replace("\"", "").replace("'", "").replace(" ", "").split('_')
            b = [float(val) for val in box]
            n_box.append(b)

        area = []
        for box in n_box:
            area.append(utils.calculate_bbox_area(box, height, width))

        for j, box in enumerate(n_box):
            x_c = 0.5 * (box[0] + box[2])
            y_c = 0.5 * (box[1] + box[3])
            w = box[2] - box[0]
            h = box[3] - box[1]
            n_box[j] = [x_c, y_c, w, h]

        labels = np.zeros(len(n_box), dtype=np.int32)

        if self.transform:
            sample = {
                'image': image_q,
                'image_t': image_t,
                'bboxes': n_box,
                'labels': labels
            }
            sample = self.transform(**sample)
            image_q = sample['image']
            image_t = sample['image_t']
            n_box = sample['bboxes']
            labels = sample['labels']

        # Resize images
        image_q = self.resize_transform(image_q).float()
        image_t = self.resize_transform(image_t).float()
        context_images = torch.stack([self.resize_transform(img).float() for img in context_images], dim=0)

        target = {}
        target["boxes"] = torch.as_tensor(n_box, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target['image_id'] = torch.tensor([idx])
        target['area'] = torch.as_tensor(area, dtype=torch.float32)

        return image_q, image_t, context_images, target, width, height, row.class_name, row.imageid




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
        image_q_name = random.choice(row['classid'][2:-2].replace("'", "").replace(" ", "").split(','))
        image_q = cv2.imread(f'{self.root_dir}/images/{image_q_name}.jpg', cv2.IMREAD_COLOR)
        image_q = cv2.cvtColor(image_q, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_q /= 255.0
        
        image_t = cv2.imread(f'{self.root_dir}/vidor/{row.imageid}.jpg', cv2.IMREAD_COLOR)
        image_t = cv2.cvtColor(image_t, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_t /= 255.0

        height, width, _ = image_t.shape

        n_box = []
        for box in self.df['boxes'][idx][2:-2].split(","):
            box = box.replace("\"", "").replace("'", "").replace(" ", "").split('_')
            b = [float(val) for val in box]

            n_box.append(b)


        area = []
        for box in n_box:
            area.append(utils.calculate_bbox_area(box, height, width))


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
    


class data_pre():
    def __init__(self, root_dir="/data1/yogesh/dataset/imagenet/ILSVRC2012_image_train", transform=utils.get_transform()):
        self.transform = transform if transform is not None else utils.get_transform()
        self.root_dir = root_dir
        self.all_files = os.listdir(self.root_dir)
        self.image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
        self.image_files = [file for file in self.all_files if any(file.lower().endswith(ext) for ext in self.image_extensions)]
        self.df = pd.DataFrame(self.image_files, columns=['Image_Name'])

    def __len__(self):
        return len(self.df)
    
    def random_patch(self, image_t, width, height):
        patch_scale = random.uniform(0.2, 0.6)
        patch_width = int(patch_scale * width)
        patch_height = int(patch_scale * height)

        x = random.randint(0, width - patch_width)
        y = random.randint(0, height - patch_height)

        image_q = image_t[y:y+patch_height, x:x+patch_width]

        n_box = [[x, y, x + patch_width, y + patch_height]]

        # Resize images and update bounding box
        image_t = cv2.resize(image_t, (512, 512))
        image_q = cv2.resize(image_q, (512, 512))

        n_box = [[int(x * 512 / width), int(y * 512 / height), int((x + patch_width) * 512 / width), int((y + patch_height) * 512 / height)]]

        return image_q, image_t, n_box
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        row = self.df.loc[idx]
        image_t = cv2.imread(f'{self.root_dir}/{row.Image_Name}', cv2.IMREAD_COLOR)
        image_t = cv2.cvtColor(image_t, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_t /= 255.0
        height, width, _ = image_t.shape

        # Generate random patch (image_q) and its bounding box
        image_q, image_t, n_box = self.random_patch(image_t, width, height)

        area = []
        for box in n_box:
            area.append(utils.calculate_bbox_area(box, height, width))

        #converting boxes into coco format
        for j, box in enumerate(n_box):
            x1, y1, x2, y2 = box
            x_center = (x1 + x2) / 2 / width
            y_center = (y1 + y2) / 2 / height
            w = (x2 - x1) / width
            h = (y2 - y1) / height
            n_box[j] = [x_center, y_center, w, h]

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

        return image_q, image_t, target, width, height