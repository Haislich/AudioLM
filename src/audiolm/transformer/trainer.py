
import torch
import torch.nn as nn
import tqdm
from math import ceil

from AbsoluteTransformer import TransformerDecoderOnly
from semantic_acoustic_modeling.W2VHuBERT_Quantizier import W2VHuBERT_Quantizier
from data_preparation import AudioDataLoader
#from data_preparation import AudioDataLoader

class Trainer(nn.Module):
    def __init__(self, model:TransformerDecoderOnly, optimizer, train_dataloader:AudioDataLoader, criterion, quantizier: W2VHuBERT_Quantizier, epochs: int = 10):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model
        self.model.to(self.device)
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.criterion = criterion
        self.quantizier = quantizier
        self.epochs = epochs
        self.loss_function = nn.CrossEntropyLoss()

    def train(self): 
        self.model.train()

        for epoch in range(self.modelepochs):
            epoch_loss = 0
            for batch in tqdm(self.train_dataloader, desc=f"Epoch: {epoch+1}/{epochs}", total=ceil(len(self.train_dataloader)//self.train_dataloader.batch_size)):
                batch = batch.to(self.device)
                with torch.no_grad():
                    semantic_token_batch = self.quantizier.forward(batch)
                input = semantic_token_batch[:, :-1].to(self.device)
                target = semantic_token_batch[:, 1:].to(self.device)
                

                self.optimizer.zero_grad()
                causal_mask = self.model.generate_causal_mask(seq_len=input.size(1)).to(self.device)
                output = self.model(input, tgt_mask=causal_mask)
                output = output.reshape(-1, output.size(-1))
                target = target.reshape(-1)

                loss = self.loss_function(output, target)
                loss.backward()
                self.optimizer.step()



