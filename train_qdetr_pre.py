import config_pre as config
import dataset
import engine_pre as engine
import utils.utils as utils
from model import QGDETRP
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


def collate_fn(batch):
    return tuple(zip(*batch))

def run():
    # creating dataloaders
    torch.cuda.empty_cache()


    train_dataset =  dataset.data_pre()

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size= config.BATCH_SIZE, shuffle=True, num_workers= 1, collate_fn=collate_fn)
    # device = torch.device('cuda:2')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = QGDETRP(config.num_classes)
    model = nn.DataParallel(model)

    matcher = HungarianMatcher()
    weight_dict = weight_dict = {'loss_ce': 1, 'loss_bbox': 1 , 'loss_giou': 1}
    losses = ['labels', 'boxes', 'cardinality']
    
    criterion = SetCriterion(config.num_classes-1, matcher, weight_dict, eos_coef = config.null_class_coef, losses=losses)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
   
    model.to(device)
    criterion.to(device)


    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", config.EPOCHS)


    for epoch in range(config.EPOCHS):
        train_loss = engine_qdetrv.train_fn(train_loader, model, criterion, optimizer, device, epoch)
        valid_loss = engine_qdetrv.eval_fn(val_loader, model, criterion, device)

        print('|EPOCH {}| TRAIN_LOSS {}| VALID_LOSS {}|'.format(epoch + 1, train_loss.avg, valid_loss.avg))
        logger.info('|EPOCH {}| TRAIN_LOSS {}| VALID_LOSS {}|'.format(epoch + 1, train_loss.avg, valid_loss.avg))

        # Save current model checkpoint
        torch.save(model.state_dict(), os.path.join(config.checkpoint_path, 'best_fine.pth'))



def main():


    global logger
    mkdir(path=config.path_check)
    mkdir(path=config.path_logs)
    logger = setup_logger(config.path_logs, config.path_logs, 0, filename="training_logs.txt")

    logger.info(f"Training started")
    
    run()

    logger.info(f"Training completed")

if __name__ == "__main__":

    
    main()

