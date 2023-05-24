import torch
import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self, img_encoder, text_encoder, projection) -> None:
        super().__init__()
        self.img_encoder = img_encoder
        self.text_encoder = text_encoder
        self.projection = projection

    def forward(self, img_1, img_2, prompt):
        img_1_embedding = self.norm(self.img_encoder(img_1))
        img_2_embedding = self.norm(self.img_encoder(img_2))
        prompt_embedding = self.norm(self.text_encoder(prompt))
        return self.projection(img_1_embedding), self.projection(img_2_embedding), self.projection(prompt_embedding)

    def norm(self, vec: torch.Tensor):
        return vec / vec.norm(dim=1, keepdim=True)

class Projection(nn.Module):
    def __init__(self, num_hidden=512) -> None:
        super().__init__()
        self.linear1 = nn.Linear(num_hidden, num_hidden, dtype=torch.float16)
        self.linear2 = nn.Linear(num_hidden, num_hidden, dtype=torch.float16)
        self.activation = F.relu

    def forward(self, embedding):
        return self.linear2(self.activation(self.linear1(embedding)))