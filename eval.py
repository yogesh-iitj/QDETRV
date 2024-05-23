import os
import config
import dataset
import engine_qdetrv
import utils.utils as utils
from model import QDETRvPre
from torch.optim.lr_scheduler import ReduceLROnPlateau

import sys
sys.path.append('./detr/')
from detr.models.matcher import HungarianMatcher
from detr.models.detr import SetCriterion
from sklearn.model_selection import train_test_split
from utils.logger import setup_logger
from utils.utils import set_seed, mkdir, load_config_file

import torch
from torch import nn
import pandas as pd

import wandb
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def collate_fn(batch):
    return tuple(zip(*batch))

def run():
    # creating dataloaders
    torch.cuda.empty_cache()

    df = pd.read_csv(config.csv_path)
    
    root_dir = config.root_dir
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    train_dataset = dataset.VidOR(root_dir=root_dir, df=train_df.reset_index(drop=True), transform=utils.get_train_transforms())
    val_dataset = dataset.VidOR(root_dir=root_dir, df=val_df.reset_index(drop=True), transform=utils.get_val_transforms())

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = QDETRv(config.num_classes)
    model = nn.DataParallel(model)

    matcher = HungarianMatcher()
    weight_dict = {'loss_ce': 1, 'loss_bbox': 1 , 'loss_giou': 1}
    losses = ['labels', 'boxes', 'cardinality']
    
    criterion = SetCriterion(config.num_classes-1, matcher, weight_dict, eos_coef=config.null_class_coef, losses=losses)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
   
    model.to(device)
    criterion.to(device)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", config.EPOCHS)
    logger.info("  Batch size = %d", config.BATCH_SIZE)

    best_loss = float('inf')

    for epoch in range(config.EPOCHS):
        train_loss = engine_qdetrv.train_fn(train_loader, model, criterion, optimizer, device, epoch)
        valid_loss = engine_qdetrv.eval_fn(val_loader, model, criterion, device)

        print('|EPOCH {}| TRAIN_LOSS {}| VALID_LOSS {}|'.format(epoch + 1, train_loss.avg, valid_loss.avg))
        logger.info('|EPOCH {}| TRAIN_LOSS {}| VALID_LOSS {}|'.format(epoch + 1, train_loss.avg, valid_loss.avg))

        # Save current model checkpoint
        torch.save(model.state_dict(), os.path.join(config.checkpoint_path, 'best_fine.pth'))

        scheduler.step(valid_loss.avg)

def evaluate():
    df = pd.read_csv(config.csv_path)
    
    root_dir = config.root_dir
    test_dataset = dataset.VidOR(root_dir=root_dir, df=df.reset_index(drop=True), transform=utils.get_val_transforms())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = QDETRv(config.num_classes)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(os.path.join(config.checkpoint_path, 'best_fine.pth')))
    model.to(device)
    model.eval()

    results = []

    with torch.no_grad():
        for images, targets in test_loader:
            images = list(image.to(device) for image in images)
            outputs = model(images)

            for output, target in zip(outputs, targets):
                pred_boxes = output['boxes'].cpu().numpy()
                gt_boxes = target['boxes'].cpu().numpy()
                image_id = target['image_id']
                category_id = target['category_id']

                for pred_box, gt_box in zip(pred_boxes, gt_boxes):
                    results.append({
                        "image_id": image_id,
                        "category_id": category_id,
                        "pred_box": pred_box,
                        "gt_box": gt_box
                    })

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(config.output_path, 'test_results.csv'), index=False)

    # Compute mAP
    coco_gt = COCO(config.annotations_path)
    coco_dt = coco_gt.loadRes(os.path.join(config.output_path, 'test_results.csv'))

    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    print("mAP@0.5:", coco_eval.stats[1])  # mAP@0.5
    print("Category-wise mAP:")
    for i, catId in enumerate(coco_gt.getCatIds()):
        cat_name = coco_gt.loadCats(catId)[0]["name"]
        print(f"Category {cat_name}: mAP {coco_eval.stats[1 + i * 12]}")

def main():
    global logger

    mkdir(path=config.path_check)
    mkdir(path=config.path_logs)
    # logger = setup_logger(config.path_logs, config.path_logs, 0, filename="training_logs.txt")

    # logger.info("Training started")
    
    # run()

    # logger.info("Training completed")

    logger.info("Evaluation started")
    
    evaluate()

    logger.info("Evaluation completed")

if __name__ == "__main__":
    main()
