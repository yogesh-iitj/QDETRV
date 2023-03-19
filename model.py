
import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.models as models

resnet = models.resnet50(pretrained=True)

for param in resnet.parameters():
    param.requires_grad = False

#model
class QGDETR(nn.Module):
    """
    Differences from DETR:
    * learned positional encoding (instead of sine)
    * positional encoding is passed at input (instead of attention)
    * fc bbox predictor (instead of MLP)
    """
    def __init__(self, num_classes, hidden_dim=256, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()
        # create ResNet-50 backbone
        self.backbone = resnet50()
        del self.backbone.fc

        # create conversion layer
        self.conv = nn.Conv2d(2048*2, hidden_dim, 1)

        # create a default PyTorch transformer
        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers)

        # prediction heads, one extra class for predicting non-empty slots
        # note that in baseline DETR linear_bbox layer is 3-layer MLP
        self.linear_class = nn.Linear(hidden_dim, num_classes)
        self.linear_bbox = nn.Linear(hidden_dim, 4)

        # output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))

        # spatial positional encodings
        # note that in baseline DETR we use sine positional encodings
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, query, inputs):
        # propagate inputs through ResNet-50 up to avg-pool layer

        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        y = self.backbone.conv1(query)
        y = self.backbone.bn1(y)
        y = self.backbone.relu(y)
        y = self.backbone.maxpool(y)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        y = self.backbone.layer1(y)
        y = self.backbone.layer2(y)
        y = self.backbone.layer3(y)
        y = self.backbone.layer4(y)

        # convert from 2048 to 256 feature planes for the transformer
        x_c = torch.cat((y,x), 1)
        h = self.conv(x_c)
        # hq = self.conv(y)
        # print("Shape after 1*1 conv", h.shape)
        
        

        # construct positional encodings
        H, W = h.shape[-2:]
        # Hq, Wq = hq.shape[-2:]
        #print(H,W)

        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)

        # print(h.shape)
        R, _ , _, _ = h.shape
        obj_query = self.query_pos.unsqueeze(1).repeat(1, R, 1)
        # print("Shape fixed pos end:", pos.shape)
        # propagate through the transformer
        # print("input shape at encoder:", (pos + 0.1 * h.flatten(2).permute(2, 0, 1)).shape)
        # print("shape of learnable object query at decoder: ", obj_query.shape)
        h = self.transformer(pos + 0.1 * h.flatten(2).permute(2, 0, 1),
                             obj_query).transpose(0, 1)
        

        return {'pred_logits': self.linear_class(h), 
                'pred_boxes': self.linear_bbox(h).sigmoid()}


class QGDETRD(nn.Module):
    """
    Differences from DETR:
    * learned positional encoding (instead of sine)
    * positional encoding is passed at input (instead of attention)
    * fc bbox predictor (instead of MLP)
    """
    def __init__(self, num_classes, hidden_dim=256, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()
        # create ResNet-50 backbone
        self.backbone = resnet50()
        del self.backbone.fc

        # create conversion layer
        self.conv = nn.Conv2d(2048*2, hidden_dim, 1)

        # create a default PyTorch transformer
        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers)

        # prediction heads, one extra class for predicting non-empty slots
        # note that in baseline DETR linear_bbox layer is 3-layer MLP
        self.linear_class = nn.Linear(hidden_dim, num_classes)
        self.linear_bbox = nn.Linear(hidden_dim, 4)

        # output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))
        self.query_onj = nn.Linear(2048, 100)
        # spatial positional encodings
        # note that in baseline DETR we use sine positional encodings
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, query, inputs):
        # propagate inputs through ResNet-50 up to avg-pool layer

        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        y = self.backbone.conv1(query)
        y = self.backbone.bn1(y)
        y = self.backbone.relu(y)
        y = self.backbone.maxpool(y)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        y = self.backbone.layer1(y)
        y = self.backbone.layer2(y)
        y = self.backbone.layer3(y)
        y = self.backbone.layer4(y)

        # convert from 2048 to 256 feature planes for the transformer
        x_c = torch.cat((y,x), 1)
        h = self.conv(x_c)
        # hq = self.conv(y)
        # print("Shape after 1*1 conv", h.shape)
        
        

        # construct positional encodings
        H, W = h.shape[-2:]
        # Hq, Wq = hq.shape[-2:]
        #print(H,W)

        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)

        # print(h.shape)
        R, _ , _, _ = h.shape
        obj_query = self.query_pos.unsqueeze(1).repeat(1, R, 1)
        # print("Shape fixed pos end:", pos.shape)
        # propagate through the transformer
        # print("input shape at encoder:", (pos + 0.1 * h.flatten(2).permute(2, 0, 1)).shape)
        # print("shape of learnable object query at decoder: ", obj_query.shape)
        h = self.transformer(pos + 0.1 * h.flatten(2).permute(2, 0, 1),
                             obj_query).transpose(0, 1)
        

        return {'pred_logits': self.linear_class(h), 
                'pred_boxes': self.linear_bbox(h).sigmoid()}