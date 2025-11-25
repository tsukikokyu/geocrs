"""
Dataset class for reading preprocessed data from pkl files.
Decouples data processing from training completely.
"""

import pickle
import os
from collections import defaultdict
import torch
from torch.utils.data import Dataset
from utils import padded_tensor


class PreprocessedDataset(Dataset):
    """General Dataset class for reading preprocessed data from pkl files."""
    
    def __init__(self, dataset, split, task, debug=False):
        """
        Args:
            dataset: dataset name
            split: dataset split (train/valid/test)
            task: task type (pre/rec)
            debug: whether debug mode is enabled
        """
        super(PreprocessedDataset, self).__init__()
        self.dataset = dataset
        self.split = split
        self.task = task
        self.debug = debug
        
        # Load preprocessed data
        processed_dir = os.path.join('data', dataset, 'processed')
        data_file = os.path.join(processed_dir, f'{split}_{task}_processed.pkl')
        
        if not os.path.exists(data_file):
            raise FileNotFoundError(
                f"Preprocessed data file does not exist: {data_file}\n"
                f"Please run the preprocessing script first: python preprocess_data.py --dataset {dataset} --task {task} ..."
            )
        
        print(f"Loading preprocessed data: {data_file}")
        with open(data_file, 'rb') as f:
            self.data = pickle.load(f)
        
        if debug and len(self.data) > 1024:
            self.data = self.data[:1024]
        
        print(f"Loaded {len(self.data)} preprocessed data samples")
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)


class PreprocessedDataCollator:
    """
    Data collator for handling preprocessed data.
    Since data has been preprocessed, this mainly handles batching and tensor conversion.
    """
    
    def __init__(self, tokenizer, pad_entity_id, prompt_tokenizer=None, 
                 device=None, use_amp=False, debug=False,
                 max_length=None, entity_max_length=None, prompt_max_length=None):
        self.tokenizer = tokenizer
        self.prompt_tokenizer = prompt_tokenizer
        self.device = device
        self.pad_entity_id = pad_entity_id
        self.debug = debug
        
        self.padding = 'max_length' if debug else True
        self.pad_to_multiple_of = 8 if use_amp else None
        
        # Set length-related parameters
        self.max_length = max_length or tokenizer.model_max_length
        self.entity_max_length = entity_max_length or tokenizer.model_max_length
        
        if prompt_tokenizer:
            self.prompt_max_length = prompt_max_length or prompt_tokenizer.model_max_length
    
    def __call__(self, data_batch):
        """Process a batch of data."""
        context_batch = defaultdict(list)
        prompt_batch = defaultdict(list)
        entity_batch = []
        label_batch = []
        pre_id_batch = []
        address_batch = []
        lonlat_batch = []
        price_batch = []
        rating_batch = []
        domain_batch = []
        
        for data in data_batch:
            context_batch['input_ids'].append(data['context'])
            if 'prompt' in data:
                prompt_batch['input_ids'].append(data['prompt'])
            if 'entity' in data:
                entity_batch.append(data['entity'])
            label_batch.append(data['rec'])
            pre_id_batch.append(data['pre_rec'])
            address_batch.append(data['address'])
            lonlat_batch.append(data['lonlat'])
            price_batch.append(data['price'])
            rating_batch.append(data['rating'])
            domain_batch.append(data['domain'])
        
        input_batch = {}
        
        # Process context batch
        context_batch = self.tokenizer.pad(
            context_batch, 
            padding='max_length', 
            max_length=self.max_length, 
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt'
        )
        context_batch['rec_labels'] = torch.as_tensor(label_batch)
        
        # Move data to device if device information is provided
        if self.device is not None:
            for k, v in context_batch.items():
                if isinstance(v, torch.Tensor):
                    context_batch[k] = v.to(self.device)
                elif k == 'rec_labels':
                    context_batch[k] = torch.as_tensor(v, device=self.device)
        
        input_batch['context'] = context_batch
        
        # Process prompt batch (if exists)
        if prompt_batch['input_ids'] and self.prompt_tokenizer:
            prompt_batch = self.prompt_tokenizer.pad(
                prompt_batch, 
                padding='max_length', 
                max_length=self.prompt_max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors='pt'
            )
            
            if self.device is not None:
                for k, v in prompt_batch.items():
                    if isinstance(v, torch.Tensor):
                        prompt_batch[k] = v.to(self.device)
            
            input_batch['prompt'] = prompt_batch
        
        # Process other data
        if pre_id_batch:
            pre_id_batch = padded_tensor(pre_id_batch, pad_idx=-1, pad_tail=True)
            input_batch['pre_id'] = pre_id_batch
        
        input_batch['address'] = address_batch
        input_batch['lonlat'] = lonlat_batch
        input_batch['metadata'] = {
            'price': price_batch, 
            'rating': rating_batch, 
            'domain': domain_batch
        }
        
        return input_batch


class PreprocessedPreDataCollator(PreprocessedDataCollator):
    """Data collator specifically for the pre task."""
    
    def __init__(self, tokenizer, pad_entity_id, prompt_tokenizer, 
                 debug=False, max_length=None, entity_max_length=None,
                 prompt_max_length=None, use_amp=False):
        super().__init__(
            tokenizer=tokenizer,
            pad_entity_id=pad_entity_id,
            prompt_tokenizer=prompt_tokenizer,
            device=None,  # device is not needed in the collator for the pre task
            use_amp=use_amp,
            debug=debug,
            max_length=max_length,
            entity_max_length=entity_max_length,
            prompt_max_length=prompt_max_length
        )


class PreprocessedRecDataCollator(PreprocessedDataCollator):
    """Data collator specifically for the rec task."""
    
    def __init__(self, tokenizer, device, pad_entity_id, prompt_tokenizer,
                 use_amp=False, debug=False, context_max_length=None, 
                 entity_max_length=None, prompt_max_length=None):
        super().__init__(
            tokenizer=tokenizer,
            pad_entity_id=pad_entity_id,
            prompt_tokenizer=prompt_tokenizer,
            device=device,
            use_amp=use_amp,
            debug=debug,
            max_length=context_max_length,
            entity_max_length=entity_max_length,
            prompt_max_length=prompt_max_length
        )
