import torch
import torch.nn.functional as F
import torchaudio
from torch import nn
import math

# def relative_to_absolute(q):
#         """
#         Converts the dimension that is specified from the axis
#         from relative distances (with length 2*tokens-1) to absolute distance (length tokens)
#         Input: [bs, heads, length, 2*length - 1]
#         Output: [bs, heads, length, length]
#         """
#         b, h, l, _, device, dtype = *q.shape, q.device, q.dtype
#         dd = {'device': device, 'dtype': dtype}
#         col_pad = torch.zeros((b, h, l, 1), **dd)
#         x = torch.cat((q, col_pad), dim=3)  # zero pad 2l-1 to 2l
#         flat_x = x.reshape(b, h, l * (x.shape[3]))
#         flat_pad = torch.zeros((b, h, l - 1), **dd)
#         flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)
#         final_x = flat_x_padded.reshape(b, h, l + 1, 2 * l - 1)
#         final_x = final_x[:, :, :l, (l - 1):]
#         return final_x

# def rel_pos_emb_1d(q, rel_emb, shared_heads):
#         """
#         Same functionality as RelPosEmb1D

#         Args:
#             q: a 4d tensor of shape [batch, heads, tokens, dim]
#             rel_emb: a 2D or 3D tensor
#             of shape [ 2*tokens-1 , dim] or [ heads, 2*tokens-1 , dim]
#         """
#         if shared_heads:
#             emb = torch.einsum('b h t d, r d -> b h t r', q, rel_emb)
#         else:
#             emb = torch.einsum('b h t d, h r d -> b h t r', q, rel_emb)
#         return relative_to_absolute(emb)


# class RelativePositionalEmbedding(nn.Module):
   
#    def __init__(self, tokens, dim_head, heads=None):
#        super().__init__()
#        scale = dim_head ** -0.5
#        self.shared_heads = heads if heads is not None else True
#        if self.shared_heads:
#            self.rel_pos_emb = nn.Parameter(torch.randn(2 * tokens - 1, dim_head) * scale)
#        else:
#            self.rel_pos_emb = nn.Parameter(torch.randn(heads, 2 * tokens - 1, dim_head) * scale)

#    def forward(self, q):
#        return rel_pos_emb_1d(q, self.rel_pos_emb, self.shared_heads)


class RelativePosition(nn.Module):

    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units)).to(self.device)
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = torch.LongTensor(final_mat).to(self.device)
        embeddings = self.embeddings_table[final_mat].cuda(self.device)

        return embeddings
   


"""
Resources:
https://towardsdatascience.com/transformers-explained-visually-part-3-multi-head-attention-deep-dive-1c1ff1024853 
https://www.kaggle.com/code/arunmohan003/transformer-from-scratch-using-pytorch
https://www.datacamp.com/tutorial/building-a-transformer-with-py-torch 
"""



class MultiHeadAttention(nn.Module): 
    
    def __init__(self,
                 embed_dim=100, 
                 num_heads=2,
                 sequence_length=100,
                 attn_dropout_prob: float = 0.1,
                 embed_dropout_prob: float = 0.1,
                 device = "cuda" if torch.cuda.is_available() else "cpu"
                 ):
        super(MultiHeadAttention, self).__init__()

        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads"

        self.single_head_dim = embed_dim // num_heads #Dimensione di ogni attention head
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        _tril_reshape = torch.tril(torch.ones(sequence_length, sequence_length)).view(
            1, 1, sequence_length, sequence_length
        )
        self.register_buffer("casual_mask", _tril_reshape)
        self.max_relative_position = 2

        self.relative_position_k = RelativePosition(self.single_head_dim, self.max_relative_position)
        self.relative_position_v = RelativePosition(self.single_head_dim, self.max_relative_position)


        #Definizione dei layer di trasformazione. 
        #Ogni Linear layer riceverà in input il tensore di embedding, coi propri pesi restituiranno la matrice della query, della chiave e del valore
        self.W_q = nn.Linear(embed_dim, embed_dim) #[batch_size, seq_length, embed_dim]
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)

        self.attn_dropout = nn.Dropout(attn_dropout_prob)
        self.embed_dropout = nn.Dropout(embed_dropout_prob)

        self.scale = torch.sqrt(torch.FloatTensor([self.single_head_dim])).to(device)
   

    def split_heads(self, x):
        """
        Dividiamo il tensore in num_heads parti, una per ogni attention head.
        Il metodo View ridimensiona il tensore senza cambiarne il contenuto.
        In questo caso stiamo splittando la terza dimensione embed_dim aggiungendo una quarta dimensione al tensore, 
        passando così da: [batch_size, seq_length, embed_dim] -> [batch_size, seq_length, num_heads, single_head_dim]
        facendo si che ogni attention head possa operare su una singola parte del tensore.
        """
        batch_size, _, _ = x.size()
        return x.view(batch_size, -1, self.num_heads, self.embed_dim).permute(0, 2, 1, 3)


    def combine_heads(self, abs_rel_attn_score, batch_size):
        """
        Operazione inversa a split_heads per unire i risultati delle n_attention heads appena calcolati
        """
        #[batch size, n heads, query len, head dim] -> [batch size, query len, n heads, head dim]
        abs_rel_attn_score = abs_rel_attn_score.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.embed_dim)
        return abs_rel_attn_score


    def scaled_dot_product_attention(self, Q, K, V, seq_len):
        """
        Qui calcoliamo l'attention score incapsulando il prodotto scalare tra query e chiavi all'interno di una softmax 

        Returns:
            torch.Tensor: L'attention score (Una matrice di dimensione indicante l'importanza di ogni valore rispetto alla query)
        """

        len_k = K.shape[1]
        len_q = Q.shape[1]
        len_v = V.shape[1]
        batch_size = Q.shape[0]

        Q1_r = self.split_heads(Q)
        K1_r = self.split_heads(K)
        attn_values1 = torch.matmul(Q1_r, K1_r.permute(0, 1, 3, 2)) / self.scale

        Q2_r = Q.permute(1, 0, 2).contiguous().view(len_q, batch_size*self.num_heads, self.embed_dim)
        K2_r = self.relative_position_k(len_q, len_k)
        attn_values2 = torch.matmul(Q2_r, K2_r.transpose(1, 2)).transpose(0,1)
        attn_values2 = attn_values2.contiguous().view(batch_size, self.num_heads, len_q, len_k) / self.scale
        attn_score = attn_values1 + attn_values2

        attn_score = attn_score.masked_fill(self.casual_mask[:, :, :seq_len, :seq_len] == 0, -1e9) #Sostituiamo i valori -inf per evitare che vengano considerati dalle attention, 
                                                                                                            #è una pratica comune per le attention mask

        attn_score = F.softmax(attn_score, dim=-1) #Applichiamo la softmax per ottenere l'attention score
        attn_score = self.attn_dropout(attn_score) #Applichiamo il dropout

        #attn_score = [batch size, n heads, query len, key len]
        V_r = self.split_heads(V)
        w1 = torch.matmul(attn_score, V_r)
        V2_r = self.relative_position_v(len_q, len_v)
        w2 = attn_score.permute(2, 0, 1, 3).contiguous().view(len_q, batch_size*self.num_heads, len_k)
        w2 = torch.matmul(w2, V2_r)
        w2 = w2.transpose(0, 1).contiguous().view(batch_size, self.num_heads, len_q, self.single_head_dim)

        return w1 + w2


    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        Calcola l'attention multi-head splittando prima il calcolo tra le n_heads e poi riunendo i risultati
        """

        Q = self.W_q(x) #[batch_size, query_len, embed_dim]
        K = self.W_k(x)
        V = self.W_v(x)
        batch_size = Q.shape[0]

        #Calcoliamo l'attention score
        abs_rel_attn_score = self.scaled_dot_product_attention(Q, K, V, x.shape[1])
        
        #Uniamo le n_heads
        abs_rel_attn_score = self.combine_heads(abs_rel_attn_score, batch_size)
        
        output = self.W_o(abs_rel_attn_score)

        return output


class FeedForward(nn.Module):
    def __init__(self, embed_dim=100, embed_dropout_prob=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(embed_dropout_prob),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


class CasualAttentionBlock(nn.Module):
    def __init__(self, embed_dim=100):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.multi_head_self_attention = MultiHeadAttention()
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.ff_net = FeedForward()

    def forward(self, x: torch.Tensor):
        x = x + self.multi_head_self_attention(self.layer_norm_1(x))
        x = x + self.ff_net(self.layer_norm_2(x))
        return x


class RelativeTransformer(nn.Module):
    def __init__(self, num_layers=12, embed_dim=100):
        super().__init__()
        
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.token_embed_table = "TODO"
        self.pos_embed_table = "TODO"
        self.layers = nn.ModuleList([CasualAttentionBlock(embed_dim) for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, self.token_embed_table.shape[0])

        self.apply(self.init_weights)
        self.sequence_length = "TODO"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def init_weights(self, module):
        pass

#     def forward(self, x: torch.Tensor):
#         for layer in self.layers:
#             x = layer(x)
#         return x

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

