import cv2
import pandas as pd
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import os
import errno
import random
import yaml
import os.path as op
from pathlib import Path
import json
from collections import OrderedDict
from omegaconf import OmegaConf



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_transform():
    return A.Compose([
        A.Resize(512, 512, p=1.0),
        ToTensorV2(p=1.0)
    ], bbox_params=A.BboxParams(format='coco', label_fields=['labels']),
       additional_targets={'image_t': 'image'}
    )

def get_train_transforms():
    return A.Compose([
        # A.ToGray(p=0.01),
        # A.HorizontalFlip(p=0.5),
        # A.VerticalFlip(p=0.5),
        A.Resize(height=512, width=512, p=1),
        # A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(p=1.0)],
        p=1.0,
        bbox_params=A.BboxParams(format='coco', min_area=0, min_visibility=0, label_fields=['labels']),
        additional_targets={'image_t': 'image'}
    )

def get_valid_transforms():
    return A.Compose([A.Resize(height=512, width=512, p=1.0),
                      A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                      ToTensorV2(p=1.0)], 
                      p=1.0, 
                      bbox_params=A.BboxParams(format='coco',min_area=0, min_visibility=0,label_fields=['labels']),
                      additional_targets={'image_t': 'image'}
                      )



def normalize_bbox(bbox, rows, cols):
    """Normalize coordinates of a bounding box. Divide x-coordinates by image width and y-coordinates
    by image height.
    """
    x_min, y_min, x_max, y_max = bbox[:4]
    normalized_bbox = [x_min / cols, y_min / rows, x_max / cols, y_max / rows]
    return normalized_bbox + list(bbox[4:])



def denormalize_bbox(bbox, rows, cols):
    """Denormalize coordinates of a bounding box. Multiply x-coordinates by image width and y-coordinates
    by image height. This is an inverse operation for :func:`~albumentations.augmentations.bbox.normalize_bbox`.
    """
    x_min, y_min, x_max, y_max = bbox[:4]
    denormalized_bbox = [x_min * cols, y_min * rows, x_max * cols, y_max * rows]
    return denormalized_bbox + list(bbox[4:])



def normalize_bboxes(bboxes, rows, cols):
    """Normalize a list of bounding boxes."""
    return [normalize_bbox(bbox, rows, cols) for bbox in bboxes]



def denormalize_bboxes(bboxes, rows, cols):
    """Denormalize a list of bounding boxes."""
    return [denormalize_bbox(bbox, rows, cols) for bbox in bboxes]



def calculate_bbox_area(bbox, rows, cols):
    """Calculate the area of a bounding box in pixels."""
    bbox = denormalize_bbox(bbox, rows, cols)
    x_min, y_min, x_max, y_max = bbox[:4]
    area = (x_max - x_min) * (y_max - y_min)
    return area


def mkdir(path):
    # if it is the current folder, skip.
    if path == '':
        return
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def load_config_file(file_path):
    with open(file_path, 'r') as fp:
        return OmegaConf.load(fp)


def set_seed(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def load_from_yaml_file(yaml_file):
    with open(yaml_file, 'r') as fp:
        return yaml.load(fp, Loader=yaml.FullLoader)


def find_file_path_in_yaml(fname, root):
    if fname is not None:
        if op.isfile(fname):
            return fname
        elif op.isfile(op.join(root, fname)):
            return op.join(root, fname)
        else:
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), op.join(root, fname)
            )


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)