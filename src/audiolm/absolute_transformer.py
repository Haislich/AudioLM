import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from transformers import GPT2LMHeadModel

from audiolm.costants import DEBUG


class TransformerDecoderOnly(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        num_heads=16,
        layers=12,
        feedforward_dim=4096,
        attn_dropout_prob=0.1,
    ):
        super(TransformerDecoderOnly, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.layers = layers
        self.feedforward_dim = feedforward_dim
        self.attn_dropout_prob = attn_dropout_prob
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.embedding_table = nn.Embedding(vocab_size, embed_dim)

        self.positional_encoding = PositionalEncoding(embed_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=attn_dropout_prob,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=layers,
        )

        self.dim_model = embed_dim
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, tgt, memory=None, tgt_mask=None, tgt_key_padding_mask=None):

        print(f"tgt:{tgt.shape},  {math.sqrt(self.dim_model)}, {tgt.dtype}")
        tgt = self.embedding_table(tgt) * math.sqrt(self.dim_model)
        tgt = self.positional_encoding(tgt)

        if memory is None:
            memory = tgt
        print(f"tgt shape: {tgt.shape}" if DEBUG else "")
        # print(
        #     f"tgt_key_padding_mask shape: {tgt_key_padding_mask.shape}" if DEBUG else ""
        # )
        # tgt_mask = torch.triu(torch.ones((2, 149)), diagonal=1).bool()
        output = self.transformer_decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        output = self.fc_out(output)
        return output

    def generate_causal_mask(self, seq_len=None):
        assert seq_len, "se_len cannot be None"
        mask = torch.triu(torch.ones((seq_len, seq_len)), diagonal=1).bool()
        # mask = mask.unsqueeze(0).repeat((2, 1, 1))
        return mask

    def fit(self, batch):
        # teacher forcing, shift the input by one position in order to predict the next token
        input_token = batch[:, :-1].to(self.device)
        target_token = batch[:, 1:].to(self.device)

        # generate causal mask to prevent the model from attending to future tokens
        causal_mask = self.generate_causal_mask(seq_len=input_token.size(1)).to(
            self.device
        )

        # forward pass
        # we pass the input to the DecoderOnly and the casual mask, so at time-i
        # the model can only attend to the tokens from time 0 to i and predict the token at time i+1
        # example:
        # semantic_token_batch = [t1, t2, t3, t4]
        # input = [t1, t2, t3], target = [t2, t3, t4], causal_mask = [[1, 0, 0], [1, 1, 0], [1, 1, 1]]
        # iteration 1: input = [t1, t2, t3] mask = [1, 0, 0] -> predict t2
        # iteration 2: input = [t1, t2, t3] mask = [1, 1, 0] -> predict t3
        # iteration 3: input = [t1, t2, t3] mask = [1, 1, 1] -> predict t4
        print(f"input_token.size(1): {input_token.size(1)}" if DEBUG else "")
        print(f"causal_mask shape: {causal_mask.shape}" if DEBUG else "")

        output = self.forward(input_token, tgt_mask=causal_mask)
        print(
            f"target.shape , output.shape prima del reshape: {target_token.shape}, {output.shape},"
        )
        output = output.view(-1, output.size(-1))
        target_token = target_token.reshape(-1)
        print(
            f"target.shape , output.shape dopo il reshape: {target_token.shape}, {output.shape},"
        )

        # output = output.reshape(-1, output.size(-1))
        # target_token = target_token.reshape(-1)
        return output, target_token

    def generate(self, prompt_ids, max_length: int, temperature: float = 1.0):
        self.eval()
        with torch.no_grad():
            for _ in range(max_length):
                logits = self.forward(prompt_ids)
                logits = logits[:, -1, :] / (temperature)
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(
                    probs, num_samples=1
                )  # sample from the distribution
                token_generated = torch.cat([prompt_ids, next_token], dim=1)

            return token_generated


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, dim_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, dim_model, 2).float() * (-math.log(10000.0) / dim_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, 0 : x.size(1), :]
        # print(f"self.pe = {self.pe[:, 0 : x.size(1), :].shape}, x shape {x.shape}")
        return self.dropout(x)


def initialize_transformer_from_gpt(
    initialized_model: TransformerDecoderOnly, gpt_pretrained_model: GPT2LMHeadModel
):

    d_model = gpt_pretrained_model.config.n_embd
    nhead = gpt_pretrained_model.config.n_head
    num_layers = gpt_pretrained_model.config.n_layer
    dim_feedforward = (
        gpt_pretrained_model.config.n_inner
        if gpt_pretrained_model.config.n_inner is not None
        else 4 * d_model
    )

    vocab_size = initialized_model.vocab_size
    initialized_model.embed_dim = d_model
    initialized_model.num_heads = nhead
    initialized_model.layers = num_layers
    initialized_model.feedforward_dim = dim_feedforward

    with torch.no_grad():
        initialized_model.embedding_table.weight.data[:vocab_size, :] = (
            gpt_pretrained_model.transformer.wte.weight.data[:vocab_size, :]
        )

        max_pos_encoding = min(
            initialized_model.positional_encoding.pe.size(1),
            gpt_pretrained_model.transformer.wpe.weight.size(0),
        )
        initialized_model.positional_encoding.pe.data[0, :max_pos_encoding, :] = (
            gpt_pretrained_model.transformer.wpe.weight.data[:max_pos_encoding, :]
        )

        for i in range(num_layers):
            decoder_layer = initialized_model.transformer_decoder.layers[i]
            gpt2_layer = gpt_pretrained_model.transformer.h[i]

            decoder_layer.self_attn.in_proj_weight.data = (
                gpt2_layer.attn.c_attn.weight.data
            )
            decoder_layer.self_attn.in_proj_bias.data = gpt2_layer.attn.c_attn.bias.data
            decoder_layer.self_attn.out_proj.weight.data = (
                gpt2_layer.attn.c_proj.weight.data
            )
            decoder_layer.self_attn.out_proj.bias.data = (
                gpt2_layer.attn.c_proj.bias.data
            )

            decoder_layer.linear1.weight.data = gpt2_layer.mlp.c_fc.weight.data
            decoder_layer.linear1.bias.data = gpt2_layer.mlp.c_fc.bias.data
            decoder_layer.linear2.weight.data = gpt2_layer.mlp.c_proj.weight.data
            decoder_layer.linear2.bias.data = gpt2_layer.mlp.c_proj.bias.data

            decoder_layer.norm1.weight.data = gpt2_layer.ln_1.weight.data
            decoder_layer.norm1.bias.data = gpt2_layer.ln_1.bias.data
            decoder_layer.norm2.weight.data = gpt2_layer.ln_2.weight.data
            decoder_layer.norm2.bias.data = gpt2_layer.ln_2.bias.data

        initialized_model.fc_out.weight[:vocab_size, :] = (
            gpt_pretrained_model.lm_head.weight.data[:vocab_size, :]
        )

    print("Model initialized from GPT2")
    print(
        f"Hyperparameters: d_model={d_model}, nhead={nhead}, num_layers={num_layers}, dim_feedforward={dim_feedforward}"
    )

    return initialized_model


# Example usage

# gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')

# initialize_transformer_from_gpt(TransformerDecoderOnly(500, 768, 12, 12, 3072, 0.1), gpt2_model)
