import torch
from torch import nn
import numpy as np
import pyproj
from typing import List, Tuple, Optional, Union, Dict
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import hashlib
from torch.nn.utils.rnn import pad_sequence


class SinusoidalEncoder(nn.Module):
    def __init__(self, frequency_num=16, max_radius=1000.0, min_radius=1.0):
        super().__init__()
        self.frequency_num = frequency_num
        self.max_radius = max_radius
        self.min_radius = min_radius
        freq_list = self._cal_freq_list()
        # (frequency_num, 1) -> (frequency_num, 2)
        freq_mat = np.expand_dims(freq_list, axis=1)
        self.register_buffer('freq_mat', torch.from_numpy(np.repeat(freq_mat, 2, axis=1)).float())

    def _cal_freq_list(self):
        return np.geomspace(1.0 / self.max_radius, 1.0 / self.min_radius, self.frequency_num)

    def forward(self, coords):
        # [N, F, 2]
        scaled_coords = coords.unsqueeze(1) * self.freq_mat.unsqueeze(0)
        

        x_embed_sin = torch.sin(scaled_coords[..., 0])
        x_embed_cos = torch.cos(scaled_coords[..., 0])
        y_embed_sin = torch.sin(scaled_coords[..., 1])
        y_embed_cos = torch.cos(scaled_coords[..., 1])
        
        # [N, 4 * F]
        full_embed = torch.cat([x_embed_sin, x_embed_cos, y_embed_sin, y_embed_cos], dim=-1)
        return full_embed

    @property
    def embedding_dim(self):
        return 4 * self.frequency_num

class MultiScaleProjectedEncoder(nn.Module):
    def __init__(self, center_lon, center_lat, scales_in_meters, frequency_num=16):
        super().__init__()
        # lcc
        proj_string = (
            f"+proj=lcc +lat_1=30 +lat_2=50 +lat_0={center_lat} +lon_0={center_lon} "
            f"+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
        )
        self.projector = pyproj.Proj(proj_string)
        self.transformer = pyproj.Transformer.from_crs("epsg:4326", self.projector.crs, always_xy=True)

        self.projection_cache: Dict[str, torch.Tensor] = {}
        self.scales = scales_in_meters
        self.encoders = nn.ModuleList()
        for scale in self.scales:
            self.encoders.append(
                SinusoidalEncoder(frequency_num=frequency_num, max_radius=scale)
            )
            
        self.embedding_dim = sum(e.embedding_dim for e in self.encoders)

    def clear_cache(self):
        self.projection_cache.clear()
        
    def forward(self, coords_lon_lat):
        cache_key = hashlib.md5(coords_lon_lat.detach().cpu().numpy().tobytes()).hexdigest()
        device = coords_lon_lat.device
        if cache_key in self.projection_cache:
            projected_coords = self.projection_cache[cache_key].to(device)
        else:
            coords_lon_lat_cpu = coords_lon_lat.detach().cpu().numpy()
            lons = coords_lon_lat_cpu[:, 0]
            lats = coords_lon_lat_cpu[:, 1]
            projected_x, projected_y = self.transformer.transform(lons, lats)
            projected_coords = torch.from_numpy(
                np.stack([projected_x, projected_y], axis=1)
            ).float().to(device)
            self.projection_cache[cache_key] = projected_coords.cpu()

        all_scale_embeddings = []
        for encoder in self.encoders:
            embedding = encoder(projected_coords)
            all_scale_embeddings.append(embedding)

        final_embedding = torch.cat(all_scale_embeddings, dim=-1) # [N, total_embedding_dim]
        
        return final_embedding
    
class AddressEncoder(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.bert = AutoModel.from_pretrained(model_path)
        self.hidden_size = self.bert.config.hidden_size
        for param in self.bert.parameters():
            param.requires_grad = False

        self.segment_attention_query = nn.Parameter(torch.randn(self.hidden_size))
        
        self.segment_cache: Dict[str, torch.Tensor] = {}

    def clear_cache(self):
        self.segment_cache.clear()
        
    def forward(self, address_list):
        all_segments = []
        segments_per_address = []
        for address in address_list:
            segments = [seg.strip() for seg in address.split('[SEP]') if seg.strip()]
            all_segments.extend(segments)
            segments_per_address.append(len(segments))

        if not all_segments:
            return torch.zeros(len(address_list), self.hidden_size).to(self.segment_attention_query.device)

        cache_key = hashlib.md5(''.join(all_segments).encode()).hexdigest()
        device = next(self.parameters()).device

        if cache_key in self.segment_cache:
            pooled_embeddings = self.segment_cache[cache_key].to(device)
        else:
            inputs = self.tokenizer(all_segments, return_tensors='pt', padding=True, truncation=True, max_length=200)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.bert(**inputs)
            
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = inputs['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
            pooled_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

            self.segment_cache[cache_key] = pooled_embeddings.cpu()
        address_segment_embeddings = list(torch.split(pooled_embeddings, segments_per_address, dim=0))
        max_segments = max(segments_per_address) if segments_per_address else 0
        padded_batch = torch.zeros(len(address_list), max_segments, self.hidden_size).to(device)
        attention_mask = torch.zeros(len(address_list), max_segments).to(device)

        for i, vecs in enumerate(address_segment_embeddings):
            seq_len = len(vecs)
            padded_batch[i, :seq_len, :] = vecs
            attention_mask[i, :seq_len] = 1

        scores = torch.matmul(padded_batch, self.segment_attention_query)
        scores.masked_fill_(attention_mask == 0, -1e9)
        weights = torch.softmax(scores, dim=1)
        weighted_embeddings = padded_batch * weights.unsqueeze(-1)
        final_embeddings = torch.sum(weighted_embeddings, dim=1) # [N, hidden_size]
        return final_embeddings
    

class MetadataTransformerEncoder(nn.Module):
    def __init__(self, num_domains: int, hidden_dim: int = 128, n_head: int = 4, n_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        
        self.hidden_dim = hidden_dim

        self.cls_token_embedding = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.domain_embedding = nn.Embedding(num_domains, hidden_dim)
        self.rating_embedding_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        self.missing_rating_embedding = nn.Parameter(torch.randn(1, hidden_dim))
        self.price_embedding_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        self.missing_price_embedding = nn.Parameter(torch.randn(1, hidden_dim))

        self.feature_type_embedding = nn.Embedding(4, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_head,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_layers
        )

        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, domain_id: torch.Tensor, rating: torch.Tensor, price: torch.Tensor):
        
        entity_len = len(domain_id)
        device = domain_id.device

        cls_tokens = self.cls_token_embedding.expand(entity_len, -1, -1)
        domain_tokens = self.domain_embedding(domain_id).unsqueeze(1)

        rating_tokens = torch.zeros(entity_len, 1, self.hidden_dim, device=device, dtype=torch.float32)
        rating_missing_mask = (rating == -1.0)
        rating_present_mask = ~rating_missing_mask
        
        if rating_present_mask.any():
            present_ratings = rating[rating_present_mask].unsqueeze(-1)
            mlp_output = self.rating_embedding_mlp(present_ratings).to(rating_tokens.dtype)
            rating_tokens[rating_present_mask] = mlp_output.unsqueeze(1)
        
        if rating_missing_mask.any():
            rating_tokens[rating_missing_mask] = self.missing_rating_embedding.to(rating_tokens.dtype)
            
        price_tokens = torch.zeros(entity_len, 1, self.hidden_dim, device=device, dtype=torch.float32)
        price_missing_mask = (price == -1.0)
        price_present_mask = ~price_missing_mask
        
        if price_present_mask.any():
            present_prices = torch.log1p(price[price_present_mask]).unsqueeze(-1)
            mlp_output = self.price_embedding_mlp(present_prices).to(price_tokens.dtype)
            price_tokens[price_present_mask] = mlp_output.unsqueeze(1)
            
        if price_missing_mask.any():
            price_tokens[price_missing_mask] = self.missing_price_embedding.to(price_tokens.dtype)

        input_sequence = torch.cat([cls_tokens, domain_tokens, rating_tokens, price_tokens], dim=1)
        
        type_ids = torch.arange(4, device=device).unsqueeze(0)
        type_embeddings = self.feature_type_embedding(type_ids)
        input_sequence_with_types = input_sequence + type_embeddings
        
        transformer_output = self.transformer_encoder(input_sequence_with_types)
        cls_output = transformer_output[:, 0, :]
        final_embedding = self.output_norm(cls_output) # [N, hidden_dim]
        
        return final_embedding
    
      
class HierarchicEncoder(nn.Module):
    def __init__(self, 
                 bert_model_path: str,
                 center_lon: float, center_lat: float, scales_in_meters: List[float],
                 num_domains: int, metadata_hidden_dim: int,
                 final_hidden_dim: int, n_head: int = 4,
                 max_sequence_length: int = 32,
                 ffn_dim_multiplier: int = 4, 
                 dropout: float = 0.1,
                 ):
        super().__init__()

        self.address_encoder = AddressEncoder(model_path=bert_model_path)
        self.location_encoder = MultiScaleProjectedEncoder(center_lon=center_lon, center_lat=center_lat, scales_in_meters=scales_in_meters)
        self.metadata_encoder = MetadataTransformerEncoder(num_domains=num_domains, hidden_dim=metadata_hidden_dim, n_head=n_head, dropout=dropout)
        
        interaction_input_dim = self.address_encoder.hidden_size + self.location_encoder.embedding_dim + self.metadata_encoder.hidden_dim
        self.interaction_mlp = nn.Sequential(
            nn.Linear(interaction_input_dim, final_hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(final_hidden_dim * 2, final_hidden_dim)
        )

        self.pointwise_ffn = nn.Sequential(
            nn.Linear(final_hidden_dim, final_hidden_dim * ffn_dim_multiplier),
            nn.ReLU(),
            nn.Dropout(dropout), 
            nn.Linear(final_hidden_dim * ffn_dim_multiplier, final_hidden_dim)
        )
        self.ffn_norm = nn.LayerNorm(final_hidden_dim)
        self.ffn_dropout = nn.Dropout(dropout)

        self.positional_encoding = nn.Embedding(max_sequence_length, final_hidden_dim)
        self.max_sequence_length = max_sequence_length

    def _encode_rich_nodes(self, addresses: List[str], locations: torch.Tensor, 
                           domains: torch.Tensor, ratings: torch.Tensor, prices: torch.Tensor) -> torch.Tensor:
        addr_embs = self.address_encoder(addresses)
        loc_embs = self.location_encoder(locations)
        meta_embs = self.metadata_encoder(domains, ratings, prices)
        
        combined_embs = torch.cat([addr_embs, loc_embs, meta_embs], dim=1)
        rich_node_embs = self.interaction_mlp(combined_embs)
        return rich_node_embs

    def forward(self, 
                address_lists: Optional[List[List[str]]] = None,
                location_lists: Optional[List[List[Tuple[float, float]]]] = None,
                domain_lists: Optional[List[List[int]]] = None,
                rating_lists: Optional[List[List[float]]] = None,
                price_lists: Optional[List[List[float]]] = None,
               ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        device = next(self.parameters()).device
        nodes_per_group = [len(addr) for addr in address_lists]
        if sum(nodes_per_group) == 0:
            return torch.empty(len(address_lists), 0, self.interaction_mlp[-1].out_features, device=device), None

        flat_addresses = [addr for group in address_lists for addr in group]
        flat_locations = torch.tensor([loc for group in location_lists for loc in group], dtype=torch.float, device=device)
        flat_domains = torch.tensor([dom for group in domain_lists for dom in group], dtype=torch.long, device=device)
        flat_ratings = torch.tensor([rat for group in rating_lists for rat in group], dtype=torch.float, device=device)
        flat_prices = torch.tensor([pri for group in price_lists for pri in group], dtype=torch.float, device=device)

        all_rich_node_embs = self._encode_rich_nodes(
            flat_addresses, flat_locations, flat_domains, flat_ratings, flat_prices
        )
        
        split_node_embs = list(torch.split(all_rich_node_embs, nodes_per_group, dim=0))
        truncated_embs = [seq[:self.max_sequence_length] for seq in split_node_embs]
        
        padded_embs = pad_sequence(truncated_embs, batch_first=True, padding_value=0.0)
        batch_size, seq_len, hidden_dim = padded_embs.shape

        positions = torch.arange(seq_len, device=device)
        pos_embeddings = self.positional_encoding(positions)
        embs_with_pos = padded_embs + pos_embeddings
        
        normed_embs = self.ffn_norm(embs_with_pos)
        ffn_output = self.pointwise_ffn(normed_embs)
        processed_embs = embs_with_pos + self.ffn_dropout(ffn_output)

        return processed_embs, None

    