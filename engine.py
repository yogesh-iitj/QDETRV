import config
import utils.utils as utils
from tqdm import tqdm
import torch

def train_fn(data_loader, model, criterion, optimizer, device, epoch, scheduler=None):
    model.train()
    criterion.train()

    summary_loss = utils.AverageMeter()

    tk0 = tqdm(data_loader, total=len(data_loader))

    for step, (image_q, image_t, targets, width, height, classid, imageid) in enumerate(tk0):

        images_q = list(image_q.to(device) for image_q in image_q)
        images_t = list(image_t.to(device) for image_t in image_t)
        # targets = [{k: v.to(device) for k, v in t.items()} for t in target]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # targets = [{k: v for k, v in t.items()} for t in targets]
        # targets = [{k: v.to(device) for k, v in targets.items()}]
        # print(len(image_q))
        # print(image_q[0].shape)
        images_q = torch.stack(images_q).to(device)
        images_t = torch.stack(images_t).to(device)
        # print(images_q.shape)
        # print(images_t.shape)
        output = model(images_q, images_t)
        loss_dict = criterion(output, targets)

        weight_dict = criterion.weight_dict

        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        summary_loss.update(losses.item(), config.BATCH_SIZE)
        tk0.set_postfix(loss=summary_loss.avg)

    return summary_loss



def eval_fn(data_loader, model, criterion, device):
    model.eval()
    criterion.eval()

    summary_loss = utils.AverageMeter()

    with torch.no_grad():

        tk0 = tqdm(data_loader, total=len(data_loader))
        for step, (image_q, image_t, targets, width, height, classid, imageid) in enumerate(tk0):

            images_q = list(image_q.to(device) for image_q in image_q)
            images_t = list(image_t.to(device) for image_t in image_t)
            # targets = [{k: v.to(device) for k, v in t.items()} for t in target]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            # targets = [{k: v for k, v in t.items()} for t in targets]
            # targets = [{k: v.to(device) for k, v in targets.items()}]
            # print(len(image_q))
            # print(image_q[0].shape)
            images_q = torch.stack(images_q).to(device)
            images_t = torch.stack(images_t).to(device)
            # print(images_q.shape)
            # print(images_t.shape)
            output = model(images_q, images_t)
            loss_dict = criterion(output, targets)

            weight_dict = criterion.weight_dict

            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            summary_loss.update(losses.item(), config.BATCH_SIZE)
            tk0.set_postfix(loss=summary_loss.avg)
        
        return summary_loss

