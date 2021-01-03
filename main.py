import torch
from models.vision_transformer import VisionTransformer

test = torch.rand(10, 3, 32, 32)
vision_transformer = VisionTransformer()
res = vision_transformer(test)
print(res.shape)