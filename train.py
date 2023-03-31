import config
import dataset
import engine
import utils.utils as utils
from model import QGDETR
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

wandb.init(project='one-shot-vid-det_f_data', entity='yogeshiitj')

# config = dict(
#     epochs=5,
#     classes=10,
#     kernels=[16, 32],
#     batch_size=128,
#     learning_rate=0.005,
#     dataset="MNIST",
#     architecture="CNN")

def collate_fn(batch):
    return tuple(zip(*batch))

def run():
    # creating dataloaders
    torch.cuda.empty_cache()
    # df = pd.read_csv(config.data_df)
    # train_df, test_df = df.loc[df['split'] == 'train'], df.loc[df['split'] == 'val-new-cl']

    df = pd.read_csv('/data1/saswats/baseline/os2d/baselines/CoAE/data_new_open/grozi/classes/grozi_listed.csv')
    train_df, test_df = df.loc[df['split'] == "['train']"].reset_index(drop=True), df.loc[df['split'] == "['val-new-cl']"].reset_index(drop=True) 
    # val_df = train_df.sample(frac=0.2, random_state=42)
    # train_df = train_df[:1000]
    train, val = train_test_split(train_df, test_size=0.2, random_state=42)

    root_dir = '/data1/yogesh/one-shot-det-vid/dataset/final_split'
    train_dataset =  dataset.VidORnf(root_dir = root_dir, df = train.reset_index(drop=True), transform=utils.get_train_transforms())
    val_dataset = dataset.VidORnf(root_dir = root_dir, df = val.reset_index(drop=True), transform=utils.get_valid_transforms())
    test_dataset = dataset.VidORnf(root_dir = root_dir, df = test_df, transform=None)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size= config.BATCH_SIZE, shuffle=True, num_workers= config.NUM_W, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size= config.BATCH_SIZE, shuffle=False, num_workers= config.NUM_W, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size= config.BATCH_SIZE, shuffle=False, num_workers= config.NUM_W, collate_fn=collate_fn)

    # device = torch.device('cuda:2')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = QGDETR(config.num_classes)
    model = nn.DataParallel(model)
    # model.load_state_dict(torch.load('/data1/yogesh/one-shot-det-vid/qdetr/checkpoint/QGdetrF_best_no_aug.pth'))

    matcher = HungarianMatcher()
    weight_dict = weight_dict = {'loss_ce': 1, 'loss_bbox': 1 , 'loss_giou': 1}
    losses = ['labels', 'boxes', 'cardinality']
    
    criterion = SetCriterion(config.num_classes-1, matcher, weight_dict, eos_coef = config.null_class_coef, losses=losses)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
   
    model.to(device)
    criterion.to(device)
    wandb.watch(model, criterion, log="all")
    best_loss = 10**6
    # file=open('/data1/yogesh/one-shot-det-vid/qdetr/logs/logs.txt', 'a')

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", config.EPOCHS)
    logger.info("  Number of GPUs = %d", config.n_gpu)


    for epoch in range(config.EPOCHS):

        train_loss = engine.train_fn(train_loader, model, criterion, optimizer, device, epoch)
        valid_loss = engine.eval_fn(val_loader, model, criterion, device)
        # scheduler.step(valid_loss.avg)
        # print(f"Epoch={epoch}, Train Loss={train_loss}, Val Loss={valid_loss}")

        print('|EPOCH {}| TRAIN_LOSS {}| VALID_LOSS {}|'.format(epoch+1,train_loss.avg,valid_loss.avg))
        wandb.log({'epoch': epoch+1, 'train loss': train_loss.avg, 'val loss': valid_loss.avg })
        logger.info('|EPOCH {}| TRAIN_LOSS {}| VALID_LOSS {}|'.format(epoch+1,train_loss.avg,valid_loss.avg))

        # file.write(f'|EPOCH {epoch+1}| TRAIN_LOSS {train_loss.avg}| VALID_LOSS {valid_loss.avg}\n')
        torch.save(model.state_dict(), f'/data1/yogesh/one-shot-det-vid/qdetr/checkpoint/QGdetr_new_data_best_{epoch+1}f_data.pth')
        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            print('Best model found at Epoch {}........Saving Model'.format(epoch+1))
            # file.write(f'Best model found at Epoch {epoch+1}........Saving Model\n')
            logger.info(f'Best model found at Epoch {epoch+1}........Saving Model')
            
            # torch.save(model.state_dict(), f'/data1/yogesh/one-shot-det-vid/qdetr/checkpoint/QGdetrF_best_no_aug.pth')
        scheduler.step(valid_loss.avg)

def main():


    global logger
    # creating directories for saving checkpoints and logs
    mkdir(path=config.path_check)
    mkdir(path=config.path_logs)
    logger = setup_logger(config.path_logs, config.path_logs, 0, filename="training_logs.txt")

    logger.info(f"Training started")
    
    run()

    logger.info(f"Training completed")

if __name__ == "__main__":

    
    main()

