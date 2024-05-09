import torch
import torch.nn.functional as F
import torchaudio
from torch import nn
import math

"""
Resources:
https://towardsdatascience.com/transformers-explained-visually-part-3-multi-head-attention-deep-dive-1c1ff1024853 
https://www.kaggle.com/code/arunmohan003/transformer-from-scratch-using-pytorch
https://www.datacamp.com/tutorial/building-a-transformer-with-py-torch 
"""



class MultiHeadAttention(nn.Module): 
    def __init__(self,
                 embed_dim, 
                 num_heads
                 ):
        super(MultiHeadAttention, self).__init__()
        #assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.embed_dim = embed_dim #Dimensione dell'embedding
        self.num_heads = num_heads #Numero di attention heads
        self.single_head_dim = int(self.embed_dim / self.num_heads) #Dimensione di ogni attention head

        #Definizione dei layer di trasformazione. 
        #Ogni Linear layer riceverà in input il tensore di embedding, coi propri pesi restituiranno la matrice della query, della chiave e del valore
        self.W_q = nn.Linear(self.single_head_dim, self.single_head_dim) #[batch_size, seq_length, embed_dim] -> [batch_size, seq_length, single_head_dim]
        self.W_k = nn.Linear(self.single_head_dim, self.single_head_dim) 
        self.W_v = nn.Linear(self.single_head_dim, self.single_head_dim) 
        self.W_o = nn.Linear(self.num_heads * self.single_head_dim, self.embed_dim) 


    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Qui calcoliamo l'attention score incapsulando il prodotto scalare tra query e chiavi all'interno di una softmax 

        Returns:
            torch.Tensor: L'attention score (Una matrice di dimensione indicante l'importanza di ogni valore rispetto alla query)
        """

        #Calcoliamo il prodotto scalare tra query e chiavi
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.single_head_dim)

        #Applichiamo la maschera se presente
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9) #Sostituiamo i valori 0 con -inf per evitare che vengano considerati, 
                                                                   #è una pratica comune per le attention mask

        #Applichiamo la softmax
        attn_scores = F.softmax(attn_scores, dim=-1)

        #Applichiamo l'attention score ai valori
        attn_values = torch.matmul(attn_scores, V)

        return attn_values 
    
    def split_heads(self, x):
        """
        Dividiamo il tensore in num_heads parti, una per ogni attention head.
        Il metodo View ridimensiona il tensore senza cambiarne il contenuto.
        In questo caso stiamo splittando la terza dimensione embed_dim aggiungendo una quarta dimensione al tensore, 
        passando così da: [batch_size, seq_length, embed_dim] -> [batch_size, seq_length, num_heads, single_head_dim]
        facendo si che ogni attention head possa operare su una singola parte del tensore.
        """
        batch_size, seq_length, _ = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.single_head_dim)


    def combine_heads(self, x, seq_len_query):
        """
        Operazione inversa a split_heads per unire i risultati delle n_attention heads appena calcolati
        """
        batch_size, *_ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len_query, self.single_head_dim * self.num_heads)


    def forward(self, Q, K, V, mask=None):
        """
        Calcola l'attention multi-head splittando prima il calcolo tra le n_heads e poi riunendo i risultati
        """
        seq_len_query = Q.size(1)

        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))



        #Trasponiamo Q, K e V per il calcolo del prodotto scalare

        Q = Q.transpose(1, 2) #[batch_size, num_heads, seq_length, single_head_dim]
        K = K.transpose(1, 2) 
        V = V.transpose(1, 2)

        #Calcoliamo l'attention score
        attn_values = self.scaled_dot_product_attention(Q, K, V, mask)

        #Uniamo le n_heads
        attn_values = self.W_o(self.combine_heads(attn_values, seq_len_query))

        return attn_values


#a = MultiHeadAttention(512, 8)




# class TransformerDecoder(nn.Module):
#     def __init__(self, 
#                  num_layers=12,
#                  num_heads=16, 
#                  embed_dim=None, 
#                  ffn_dim=None, 
#                  dropout=None,
#                  ):
#         super().__init__()

