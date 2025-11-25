import math
import os

import torch
from torch import nn
from torch.nn import functional as F
from hierarchic_encoder import HierarchicEncoder

class KGPrompt(nn.Module):
    def __init__(
        self, 
        model_hidden_size, 
        token_hidden_size, 
        n_entity, 
        num_virtual_tokens,
        args,
    ):
        super(KGPrompt, self).__init__()
        self.hidden_size = model_hidden_size
        self.num_virtual_tokens = num_virtual_tokens
        
        # if crosswoz dataset，center_lon = 116.413494, center_lat = 39.989686
        # if multiwoz dataset，center_lon = 0.125851, center_lat = 52.205894
        self.hierarchic_encoder = HierarchicEncoder(
            bert_model_path=args.text_encoder,
            center_lon=116.413494,
            center_lat=39.989686,
            scales_in_meters=[100.0, 1000.0, 10000.0],
            num_domains=3,
            metadata_hidden_dim=64,
            final_hidden_dim=model_hidden_size,
            n_head=4,
            max_sequence_length=32
        )
        
        entity_hidden_size = self.hidden_size // 2
        self.node_embeds = nn.Parameter(torch.empty(n_entity, entity_hidden_size))
        stdv = math.sqrt(6.0 / (self.node_embeds.size(-2) + self.node_embeds.size(-1)))
        self.node_embeds.data.uniform_(-stdv, stdv)

        self.query_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.key_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.value_proj = nn.Linear(self.hidden_size, self.hidden_size)

        self.entity_proj1 = nn.Sequential(
            nn.Linear(entity_hidden_size, entity_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(entity_hidden_size // 2, entity_hidden_size),
        )
        self.entity_proj2 = nn.Linear(entity_hidden_size, self.hidden_size)

        self.token_proj1 = nn.Sequential(
            nn.Linear(token_hidden_size, token_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(token_hidden_size // 2, token_hidden_size),
        )
        self.token_proj2 = nn.Linear(token_hidden_size, self.hidden_size)

        self.prompt_proj1 = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, self.hidden_size),
        )
        self.prompt_proj2 = nn.Linear(self.hidden_size, self.num_virtual_tokens * self.hidden_size)

        self.film_generator = nn.Linear(self.hidden_size, self.hidden_size * 2)


    def get_entity_embeds(self):
        entity_embeds = self.node_embeds
        entity_embeds = self.entity_proj1(entity_embeds) + entity_embeds
        entity_embeds = self.entity_proj2(entity_embeds)
        return entity_embeds
    
    def get_hierarchic_embeds(self, address, lonlat, metadata):
        hierarchic_embedding, _ = self.hierarchic_encoder(
            address_lists=address,
            location_lists=lonlat,
            domain_lists=metadata['domain'],
            rating_lists=metadata['rating'],
            price_lists=metadata['price']
        )
        return hierarchic_embedding

    def forward(self,token_embeds=None, address=None, lonlat=None, metadata=None):
        batch_size = token_embeds.shape[0]

        # [batch_size, seq_len, hidden_size]
        hierarchic_embedding = self.get_hierarchic_embeds(address, lonlat, metadata)
        token_embeds = self.token_proj1(token_embeds) + token_embeds
        token_embeds = self.token_proj2(token_embeds)  # [batch_size, token_len, hidden_size]
        Q = self.query_proj(token_embeds)          # [batch_size, token_len, hidden_size]
        K = self.key_proj(hierarchic_embedding)    # [batch_size, seq_len, hidden_size]
        V = self.value_proj(hierarchic_embedding)    # [batch_size, seq_len, hidden_size]
        attn_scores = torch.matmul(Q, K.transpose(-2, -1))       
        attn_scores = attn_scores / math.sqrt(K.size(-1))        
        attn_weights = F.softmax(attn_scores, dim=-1) # [batch_size, token_len, seq_len]
        # (B, T_q, T_k) @ (B, T_k, d) -> (B, T_q, d)
        context = torch.matmul(attn_weights, V)
        prompt_embeds = context + token_embeds

        hierarchic_cls_vec = hierarchic_embedding[:, 0, :]
        gamma, beta = self.film_generator(hierarchic_cls_vec).chunk(2, dim=-1)
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        # FiLM
        prompt_embeds = gamma * prompt_embeds + beta
        prompt_embeds = self.prompt_proj1(prompt_embeds) + prompt_embeds
        prompt_input_vec = prompt_embeds[:, 0, :]
        #  (batch, num_virtual_tokens, hidden_size) 
        prompt_vectors = self.prompt_proj2(prompt_input_vec)
        prompt_vectors = prompt_vectors.view(batch_size, self.num_virtual_tokens, self.hidden_size)
        full_entity_embeds = self.get_entity_embeds()
        return prompt_vectors, full_entity_embeds