
import config
import dataset
import engine
import utils
from model import QGDETR

import sys
sys.path.append('./detr/')
from detr.models.matcher import HungarianMatcher
from detr.models.detr import SetCriterion

import torch
import pandas as pd


import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def collate_fn(batch):
    return tuple(zip(*batch))

# def run():
#     # creating dataloaders
#     df = pd.read_csv('/data1/saswats/baseline/os2d/baselines/CoAE/data/grozi/classes/grozi.csv')
#     train_df, test_df = df.loc[df['split'] == 'train'], df.loc[df['split'] == 'val-new-cl']
#     val_df = train_df.sample(frac=0.2, random_state=42)

#     root_dir = '/data1/saswats/baseline/os2d/baselines/CoAE/data/grozi/'
#     train_datset =  dataset.VidOR1s(root_dir = root_dir, df = train_df, transform=utils.get_train_transforms())
#     val_dataset = dataset.VidOR1s(root_dir = root_dir, df = val_df, transform=utils.get_valid_transforms())
#     # test_dataset = dataset.VidOR1s(root_dir = root_dir, df = test_df, transform=None)

#     train_loader = torch.utils.data.DataLoader(train_datset, batch_size= config.BATCH_SIZE, shuffle=True, num_workers= config.NUM_W, collate_fn=collate_fn)
#     val_loader = torch.utils.data.DataLoader(val_dataset, batch_size= config.BATCH_SIZE, shuffle=False, num_workers= config.NUM_W, collate_fn=collate_fn)
#     # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size= config.BATCH_SIZE, shuffle=False, num_workers= config.NUM_W, collate_fn=collate_fn)

#     device = torch.device('cuda:2')

#     model = QGDETR(config.num_classes)
    
#     matcher = HungarianMatcher()
#     weight_dict = weight_dict = {'loss_ce': 1, 'loss_bbox': 1 , 'loss_giou': 1}
#     losses = ['labels', 'boxes', 'cardinality']
    
#     criterion = SetCriterion(config.num_classes-1, matcher, weight_dict, eos_coef = config.null_class_coef, losses=losses)
#     optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR)

#     model.to(device)
#     criterion.to(device)

#     best_loss = 10**6
#     for epoch in range(config.EPOCHS):

#         train_loss = engine.train_fn(train_loader, model, criterion, optimizer, device, epoch)
#         valid_loss = engine.eval_fn(val_loader, model, criterion, device)

#         # print(f"Epoch={epoch}, Train Loss={train_loss}, Val Loss={valid_loss}")

#         print('|EPOCH {}| TRAIN_LOSS {}| VALID_LOSS {}|'.format(epoch+1,train_loss.avg,valid_loss.avg))
        
#         if valid_loss.avg < best_loss:
#             best_loss = valid_loss.avg
#             print('Best model found at Epoch {}........Saving Model'.format(epoch+1))
#             torch.save(model.state_dict(), f'/data1/yogesh/one-shot-det-vid/qdetr/checkpoint/QGdetrF_best_{epoch+1}.pth')


def load_train_objs():
    # train_set = MyTrainDataset(2048)  # load your dataset
    df = pd.read_csv('/data1/saswats/baseline/os2d/baselines/CoAE/data/grozi/classes/grozi.csv')
    train_df, test_df = df.loc[df['split'] == 'train'], df.loc[df['split'] == 'val-new-cl']
    val_df = train_df.sample(frac=0.2, random_state=42)

    root_dir = '/data1/saswats/baseline/os2d/baselines/CoAE/data/grozi/'
    train_datset =  dataset.VidOR1s(root_dir = root_dir, df = train_df, transform=utils.get_train_transforms())
    val_dataset = dataset.VidOR1s(root_dir = root_dir, df = val_df, transform=utils.get_valid_transforms())
    # test_dataset = dataset.VidOR1s(root_dir = root_dir, df = test_df, transform=None)

    # train_loader = torch.utils.data.DataLoader(train_datset, batch_size= config.BATCH_SIZE, shuffle=True, num_workers= config.NUM_W, collate_fn=collate_fn)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size= config.BATCH_SIZE, shuffle=False, num_workers= config.NUM_W, collate_fn=collate_fn)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size= config.BATCH_SIZE, shuffle=False, num_workers= config.NUM_W, collate_fn=collate_fn)


    model = QGDETR(config.num_classes)  # load your model
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return train_set, model, optimizer



def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
    ddp_setup(rank, world_size)
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, rank, save_every)
    trainer.train(total_epochs)
    destroy_process_group()



if __name__ == "__main__":
    run()
    import argparse
    # parser = argparse.ArgumentParser(description='Training')
    # parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    # parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    # parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    # args = parser.parse_args()

    world_size = torch.cuda.device_count()
    mp.spawn(main, world_size, save_every, config.EPOCHS, config.BATCH_SIZE, nprocs=world_size)

