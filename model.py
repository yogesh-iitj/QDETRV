
import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.models as models
import config
#model
class QDETRv(nn.Module):
    def __init__(self, num_classes, hidden_dim=256, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6, context_window = 5):
        super().__init__()
        # Create ResNet-50 backbone
        self.backbone = models.resnet50(pretrained=True)
        for param in self.backbone.parameters():
            param.requires_grad = False

        del self.backbone.fc
        self.context_window = context_window

        # Convolution layers to project features to the required dimensions
        self.conv_q = nn.Conv2d(2048, hidden_dim, 1)
        self.conv_k = nn.Conv2d(2048*self.context_window, hidden_dim, 1)
        self.conv_v = nn.Conv2d(2048, hidden_dim, 1)

        # Cross-attention layer
        self.cross_attention = nn.MultiheadAttention(hidden_dim, nheads, batch_first=True)

        # Transformer
        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers)

        # Prediction heads
        self.linear_class = nn.Linear(hidden_dim, num_classes)
        self.linear_bbox = nn.Linear(hidden_dim, 4)

        # Positional encodings
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, query, inputs, context):
        # Process input and query through the backbone
        x = self.process_through_backbone(inputs)  # Value
        y = self.process_through_backbone(query)   # Query
       
        # Process context images and convert them to keys
        context_features = []
        for img in context:
            feature = self.process_through_backbone(img)
            context_features.append(feature)
        context_features = torch.stack(context_features, dim=0) # Aggregate context features
        context_features = context_features.view(context_features.size(0), -1, context_features.size(3), context_features.size(4))  # Reshape to [batch_size, 5*2048, 7, 7]
  
        # Project x, y, and context_features to the dimensions required for attention
        q = self.conv_q(y).flatten(2).permute(0, 2, 1)  # Query
        k = self.conv_k(context_features).flatten(2).permute(0, 2, 1)  # Key
        v = self.conv_v(x).flatten(2).permute(0, 2, 1)  # Value
        


        # Apply cross-attention
        h, _ = self.cross_attention(q, k, v)  # Output has the shape of the value
   

        # Transformer processing
        H, W = x.shape[-2:]
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)

        R, _, _ = h.shape
        obj_query = self.query_pos.unsqueeze(1).repeat(1, R, 1)

        h_transformed = self.transformer(pos + 0.1 * h.permute(1, 0, 2),
                             obj_query).transpose(0, 1)
        
        return {
            'pred_logits': self.linear_class(h_transformed), 
            'pred_boxes': self.linear_bbox(h_transformed).sigmoid(),
            'feature': h_transformed,
            'cnn_feature': self.conv_q(y)
        }

    def process_through_backbone(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        return x

class QDETRvPre(nn.Module):
    def __init__(self, num_classes, hidden_dim=256, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6, context_window = 5):
        super().__init__()
        # Create ResNet-50 backbone
        self.backbone = models.resnet50(pretrained=True)
        for param in self.backbone.parameters():
            param.requires_grad = False

        del self.backbone.fc
        self.context_window = context_window

        # Convolution layers to project features to the required dimensions
        self.conv_q = nn.Conv2d(2048, hidden_dim, 1)
        self.conv_k = nn.Conv2d(2048*self.context_window, hidden_dim, 1)
        self.conv_v = nn.Conv2d(2048, hidden_dim, 1)

        # Cross-attention layer
        self.cross_attention = nn.MultiheadAttention(hidden_dim, nheads, batch_first=True)

        # Transformer
        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers)

        # Prediction heads
        self.linear_class = nn.Linear(hidden_dim, num_classes)
        self.linear_bbox = nn.Linear(hidden_dim, 4)

        # Positional encodings
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, query, inputs, context, memory):
        # Process input and query through the backbone
        x = self.process_through_backbone(inputs)  # Value
        y = self.process_through_backbone(query)   # Query
       
        # Process context images and convert them to keys
        context_features = []
        for img in context:
            feature = self.process_through_backbone(img)
            context_features.append(feature)
        context_features = torch.stack(context_features, dim=0) # Aggregate context features
        context_features = context_features.view(context_features.size(0), -1, context_features.size(3), context_features.size(4))  # Reshape to [batch_size, 5*2048, 7, 7]
  
        # Project x, y, and context_features to the dimensions required for attention
        q = self.conv_q(y).flatten(2).permute(0, 2, 1)  # Query
        k = self.conv_k(context_features).flatten(2).permute(0, 2, 1)  # Key
        v = self.conv_v(x).flatten(2).permute(0, 2, 1)  # Value
        

        # Apply cross-attention
        h, _ = self.cross_attention(q, k, v)  # Output has the shape of the value
   
        # Transformer processing
        H, W = x.shape[-2:]
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)

        R, _, _ = h.shape
        if memory == None
            obj_query = self.query_pos.unsqueeze(1).repeat(1, R, 1)
        else 
            obj_query = memory

        h_transformed = self.transformer(pos + 0.1 * h.permute(1, 0, 2),
                             obj_query).transpose(0, 1)
        
        return {
            'pred_logits': self.linear_class(h_transformed), 
            'pred_boxes': self.linear_bbox(h_transformed).sigmoid(),
            'feature': h_transformed,
            'cnn_feature': self.conv_q(y)
        }

    def process_through_backbone(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        return x


class QDETR(nn.Module):

    def __init__(self, num_classes, hidden_dim=256, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()
        # create ResNet-50 backbone
        self.backbone = models.resnet50(pretrained=True)
        for param in self.backbone.parameters():
             param.requires_grad = False

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

        # construct positional encodings
        H, W = h.shape[-2:]

        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)

        # print(h.shape)
        R, _ , _, _ = h.shape
        obj_query = self.query_pos.unsqueeze(1).repeat(1, R, 1)

        h = self.transformer(pos + 0.1 * h.flatten(2).permute(2, 0, 1),
                             obj_query).transpose(0, 1)
        

        return {'pred_logits': self.linear_class(h), 
                'pred_boxes': self.linear_bbox(h).sigmoid()}



class QDETRD(nn.Module):

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

        # construct positional encodings
        H, W = h.shape[-2:]

        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)


        R, _ , _, _ = h.shape
        obj_query = self.query_pos.unsqueeze(1).repeat(1, R, 1)

        h = self.transformer(pos + 0.1 * h.flatten(2).permute(2, 0, 1),
                             obj_query).transpose(0, 1)
        

        return {'pred_logits': self.linear_class(h), 
                'pred_boxes': self.linear_bbox(h).sigmoid()}