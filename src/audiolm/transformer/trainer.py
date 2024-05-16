
import torch
import torch.nn as nn
import tqdm
from math import ceil
import os

from AbsoluteTransformer import TransformerDecoderOnly
from semantic_acoustic_modeling.W2VHuBERT_Quantizier import W2VHuBERT_Quantizier
from data_preparation import AudioDataLoader
#from data_preparation import AudioDataLoader

class Trainer(nn.Module):
    """
    Trainer class for training a Transformer model.

    Args:
        model (TransformerDecoderOnly): The Transformer model to be trained.
        optimizer: The optimizer used for training.
        train_dataloader (AudioDataLoader): The dataloader for the training data.
        val_dataloader (AudioDataLoader): The dataloader for the validation data.
        scheduler: The scheduler for adjusting the learning rate during training.
        intervals (int): The number of batches between each logging of training loss.
        save_path: The path to save the checkpoints and the trained model.
        quantizier (W2VHuBERT_Quantizier): The quantizier for converting audio data to semantic tokens.
        early_stopping_range (int, optional): The number of epochs to wait for improvement in validation loss before early stopping. Defaults to 5.
        epochs (int, optional): The number of training epochs. Defaults to 10.
    """
    
    def __init__(self, 
                 model:TransformerDecoderOnly, 
                 optimizer, 
                 train_dataloader:AudioDataLoader, 
                 val_dataloader: AudioDataLoader,
                 test_dataloader: AudioDataLoader,
                 scheduler, 
                 intervals: int,
                 save_path,
                 quantizier: W2VHuBERT_Quantizier, 
                 early_stopping_range:int = 5,
                 epochs: int = 10):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.model.to(self.device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optimizer
        self.intervals = intervals
        self.quantizier = quantizier
        self.scheduler = scheduler
        self.epochs = epochs
        self.save_path = save_path
        self.best_val_loss = float("inf")
        self.early_stopping_range = early_stopping_range
        self.early_stop_counter = 0
        self.loss_function = nn.CrossEntropyLoss()

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)


    def train(self):
        """
        Train the Transformer model.
        """
        self.model.train()

        for epoch in range(self.epochs):
            epoch_loss = 0
            for batch_idx, batch in enumerate(tqdm(self.train_dataloader, desc=f"Epoch: {epoch+1}/{epochs}", total=ceil(len(self.train_dataloader)/self.train_dataloader.batch_size))):
                batch = batch.to(self.device)
                with torch.no_grad():
                    #get the semantic tokens from the audio data
                    semantic_token_batch = self.quantizier.forward(batch)
                
                #teacher forcing, shift the input by one position in order to predict the next token
                input = semantic_token_batch[:, :-1].to(self.device)
                target = semantic_token_batch[:, 1:].to(self.device)
                
                self.optimizer.zero_grad()
                #generate causal mask to prevent the model from attending to future tokens
                causal_mask = self.model.generate_causal_mask(seq_len=input.size(1)).to(self.device)

                #forward pass
                #we pass the input to the DecoderOnly and the casual mask, so at time-i
                #the model can only attend to the tokens from time 0 to i and predict the token at time i+1
                #example:
                # semantic_token_batch = [t1, t2, t3, t4] 
                #input = [t1, t2, t3], target = [t2, t3, t4], causal_mask = [[1, 0, 0], [1, 1, 0], [1, 1, 1]]
                #iteration 1: input = [t1, t2, t3] mask = [1, 0, 0] -> predict t2
                #iteration 2: input = [t1, t2, t3] mask = [1, 1, 0] -> predict t3  
                #iteration 3: input = [t1, t2, t3] mask = [1, 1, 1] -> predict t4
                 
                output = self.model(input, tgt_mask=causal_mask)
                output = output.reshape(-1, output.size(-1))
                target = target.reshape(-1)

                loss = self.loss_function(output, target)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

                if self.scheduler: 
                    self.scheduler.step()
                
                if batch_idx % self.intervals == 0:
                    print(f"Epoch: {epoch+1}/{self.epochs} | Batch: {batch_idx}/{len(self.train_dataloader)} | Loss: {loss.item(): .6f}")
                
            avg_epoch_loss = epoch_loss / len(self.train_dataloader)
            print(f"Epoch: {epoch+1}/{self.epochs} | Loss: {avg_epoch_loss: .4f}")

            self.save_checkpoint(epoch)

            validation_loss = self.evaluate()
            if validation_loss < self.best_val_loss:
                self.best_val_loss = validation_loss
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
            
            if self.early_stop_counter >= self.early_stopping_range:
                print(f"Early Stopping at epoch: {epoch+1}")
                break
            
    
    def evaluate(self):
        """
        Evaluate the Transformer model on the validation data.

        Returns:
            float: The average validation loss.
        """
        self.model().eval()
        validation_loss = 0
        with torch.no_grad:
            for batch in enumerate(tqdm(self.val_dataloader, total=ceil(len(self.val_dataloader)/self.val_dataloader.batch_size))):
                batch = batch.to(self.device)
                semantic_token_batch = self.quantizier.forward(batch)
                
                #teacher forcing, shift the input by one position in order to predict the next token
                input = semantic_token_batch[:, :-1].to(self.device) 
                target = semantic_token_batch[:, 1:].to(self.device)
                
                #generate causal mask to prevent the model from attending to future tokens
                causal_mask = self.model.generate_causal_mask(seq_len=input.size(1)).to(self.device)

                #forward pass
                output = self.model(input, tgt_mask=causal_mask)
                output = output.reshape(-1, output.size(-1))
                target = target.reshape(-1)

                loss = self.loss_function(output, target)
                validation_loss += loss.item()
        
        avg_val_loss = validation_loss / len(self.val_dataloader)
        print(f"Validation Loss: {avg_val_loss: .4f}")

        return avg_val_loss


    
    def test(self):
        """
        Test the Transformer model on the test data.

        Args:
            test_dataloader (AudioDataLoader): The dataloader for the test data.
        """
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch in enumerate(tqdm(self.test_dataloader, total=ceil(len(self.test_dataloader)/self.test_dataloader.batch_size))):
                batch = batch.to(self.device)
                semantic_token_batch = self.quantizier.forward(batch)

                #teacher forcing, shift the input by one position in order to predict the next token
                input = semantic_token_batch[:, :-1].to(self.device)
                target = semantic_token_batch[:, 1:].to(self.device)

                #generate causal mask to prevent the model from attending to future tokens
                causal_mask = self.model.generate_causal_mask(seq_len=input.size(1)).to(self.device)

                output = self.model(input, tgt_mask=causal_mask)
                output = output.reshape(-1, output.size(-1))
                target = target.reshape(-1) 

                loss = self.loss_function(output, target)
                test_loss += loss.item()
        
        avg_test_loss = test_loss / len(self.test_dataloader)
        print(f"Test Loss: {avg_test_loss: .4f}")

        return avg_test_loss


    def save_checkpoint(self, epoch):
        """
        Save a checkpoint of the model and optimizer.

        Args:
            epoch (int): The current epoch.
        """
        checkpoint_path = os.path.join(self.save_path, f"model_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'early_stop_counter': self.early_stop_counter
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        

    def load_checkpoint(self, checkpoint_path):
        """
        Load a checkpoint of the model and optimizer.

        Args:
            checkpoint_path (str): The path to the checkpoint file.

        Returns:
            int: The epoch from which training will resume.
        """
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])   
        self.best_val_loss = checkpoint['best_val_loss']
        self.early_stop_counter = checkpoint['early_stop_counter']
        print(f"Checkpoint loaded: {checkpoint_path}, starting from epoch: {checkpoint['epoch']+1}")
        return checkpoint['epoch']+1



    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved: {path}")



