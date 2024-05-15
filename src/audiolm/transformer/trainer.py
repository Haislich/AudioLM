
import torch
import torch.nn as nn
from AbsoluteTransformer import TransformerDecoderOnly

def train(model, optimizer, train_dataloader, criterion, epochs): 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()
    for epoch in range(epochs):
        pass
