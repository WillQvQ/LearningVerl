# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Multi-Region Loss Mask Dataset for complex long sequence training
Supports three data formats:
1. position_based: Character position based region marking
2. tag_based: XML tag based region marking  
3. structured: Structured segment based region marking
"""

import torch
import re
import pandas as pd
from typing import List, Dict, Any, Tuple, Union
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask


class MultiRegionLossMaskDataset(Dataset):
    """
    Dataset that supports multi-region loss masking for complex long sequences.
    Extends the base SFTDataset to handle multiple learning regions with different weights.
    """
    
    def __init__(self, parquet_files, tokenizer, config):
        # Multi-region specific configuration
        self.region_config = config.get("multi_region_config", {})
        self.data_format = self.region_config.get("format", "position_based")
        self.max_regions = self.region_config.get("max_regions", 100)
        self.weight_normalize = self.region_config.get("weight_normalize", False)
        self.default_weight = self.region_config.get("default_weight", 0.0)
        self.max_length = config.get("max_length", 1024)
        self.truncation = config.get("truncation", "error")
        self.use_shm = config.get("use_shm", False)
        
        # Handle tokenizer
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer
        
        # Handle parquet files
        if not isinstance(parquet_files, List):
            parquet_files = [parquet_files]
        self.parquet_files = parquet_files
        
        # Format processors
        self.processors = {
            "position_based": self._process_position_based,
            "tag_based": self._process_tag_based,
            "structured": self._process_structured
        }
        
        # Load and prepare data
        self._download()
        self._read_files()
        
        print(f"MultiRegionLossMaskDataset initialized with format: {self.data_format}")
        print(f"Max regions: {self.max_regions}, Weight normalize: {self.weight_normalize}")
        print(f"Dataset size: {len(self.dataframe)}")
    
    def _download(self):
        """Download parquet files to local"""
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_to_local(parquet_file, verbose=True, use_shm=self.use_shm)
    
    def _read_files(self):
        """Read and concatenate parquet files"""
        dataframes = []
        for parquet_file in self.parquet_files:
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes, ignore_index=True)
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, index):
        """Process multi-region data and generate tokens with loss_mask"""
        # Get base item data
        item = self.dataframe.iloc[index]
        
        # Choose processor based on data format
        processor = self.processors.get(self.data_format, self._process_position_based)
        
        # Process data and generate tokens and loss_mask
        input_ids, attention_mask, loss_mask, region_info = processor(item)
        
        # Compute position_ids
        from verl.utils.model import compute_position_id_with_mask
        position_ids = compute_position_id_with_mask(attention_mask.unsqueeze(0)).squeeze(0)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
            # "region_info": region_info  # Save region info for debugging
        }
    
    def _process_position_based(self, item) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """Process position-based marked data"""
        # Extract data from pandas Series
        data = item.get("data", item) if hasattr(item, 'get') else item
        if hasattr(data, 'iloc') or hasattr(data, 'get'):
            # Handle nested data structure
            if 'data' in data:
                data = data['data']
        
        text = data.get("text", "") if hasattr(data, 'get') else str(data.get("text", ""))
        learning_regions = data.get("learning_regions", []) if hasattr(data, 'get') else []
        
        # Encode entire text
        inputs = self.tokenizer(
            text,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_offsets_mapping=True  # Get character to token mapping
        )
        
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        offset_mapping = inputs["offset_mapping"].squeeze(0)
        
        # Initialize loss_mask
        loss_mask = torch.zeros_like(input_ids, dtype=torch.float)
        
        # Process each learning region
        processed_regions = []
        for region in learning_regions:
            start_char = region["start_char"]
            end_char = region["end_char"]
            weight = region.get("weight", 1.0)
            
            # Convert character positions to token positions
            start_token, end_token = self._char_to_token_precise(
                offset_mapping, start_char, end_char
            )
            
            if start_token < len(loss_mask) and end_token <= len(loss_mask):
                loss_mask[start_token:end_token] = weight
                
                processed_regions.append({
                    "char_range": (start_char, end_char),
                    "token_range": (start_token, end_token),
                    "weight": weight,
                    "type": region.get("region_type", "unknown")
                })
        
        # Apply padding/truncation same as base class
        input_ids, attention_mask, loss_mask = self._apply_padding_truncation(
            input_ids, attention_mask, loss_mask
        )
        
        # Weight normalization
        if self.weight_normalize:
            loss_mask = self._normalize_weights(loss_mask)
        
        region_info = {
            "total_regions": len(processed_regions),
            "regions": processed_regions,
            "format": "position_based"
        }
        
        return input_ids, attention_mask, loss_mask, region_info
    
    def _process_tag_based(self, item) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """Process tag-based marked data"""
        # Extract data from pandas Series
        data = item.get("data", item) if hasattr(item, 'get') else item
        if hasattr(data, 'iloc') or hasattr(data, 'get'):
            # Handle nested data structure
            if 'data' in data:
                data = data['data']
        
        text = data.get("text", "") if hasattr(data, 'get') else str(data.get("text", ""))
        default_weight = data.get("default_weight", self.default_weight) if hasattr(data, 'get') else self.default_weight
        
        # Ensure default_weight is a valid number
        if default_weight is None:
            default_weight = self.default_weight
        
        # Parse learning region tags
        regions, clean_text = self._parse_learn_tags(text)
        
        # Encode cleaned text
        inputs = self.tokenizer(
            clean_text,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_offsets_mapping=True
        )
        
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        offset_mapping = inputs["offset_mapping"].squeeze(0)
        
        # Initialize loss_mask with default weight
        loss_mask = torch.full_like(input_ids, default_weight, dtype=torch.float)
        
        # Apply learning regions
        processed_regions = []
        for region in regions:
            start_char = region["start_char"]
            end_char = region["end_char"]
            weight = region["weight"]
            
            start_token, end_token = self._char_to_token_precise(
                offset_mapping, start_char, end_char
            )
            
            if start_token < len(loss_mask) and end_token <= len(loss_mask):
                loss_mask[start_token:end_token] = weight
                
                processed_regions.append({
                    "char_range": (start_char, end_char),
                    "token_range": (start_token, end_token),
                    "weight": weight,
                    "type": region.get("type", "unknown")
                })
        
        # Apply padding/truncation
        input_ids, attention_mask, loss_mask = self._apply_padding_truncation(
            input_ids, attention_mask, loss_mask
        )
        
        region_info = {
            "total_regions": len(processed_regions),
            "regions": processed_regions,
            "format": "tag_based",
            "clean_text": clean_text
        }
        
        return input_ids, attention_mask, loss_mask, region_info
    
    def _process_structured(self, item) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """Process structured segment data"""
        # Extract data from pandas Series
        data = item.get("data", item) if hasattr(item, 'get') else item
        if hasattr(data, 'iloc') or hasattr(data, 'get'):
            # Handle nested data structure
            if 'data' in data:
                data = data['data']
        
        segments = data.get("segments", []) if hasattr(data, 'get') else []
        
        # Ensure segments is a valid list
        if segments is None:
            segments = []
        
        # Rebuild full text and segment mapping
        full_text = ""
        char_to_segment = []
        current_pos = 0
        
        for i, segment in enumerate(segments):
            content = segment["content"]
            start_pos = current_pos
            end_pos = current_pos + len(content)
            
            full_text += content
            char_to_segment.extend([i] * len(content))
            current_pos = end_pos
        
        # Encode full text
        inputs = self.tokenizer(
            full_text,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_offsets_mapping=True
        )
        
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        offset_mapping = inputs["offset_mapping"].squeeze(0)
        
        # Initialize loss_mask
        loss_mask = torch.zeros_like(input_ids, dtype=torch.float)
        
        # Assign weights to each token
        processed_segments = []
        for token_idx, (start_char, end_char) in enumerate(offset_mapping):
            if token_idx >= len(input_ids):
                break
                
            # Find token's corresponding segment
            mid_char = (start_char + end_char) // 2
            if mid_char < len(char_to_segment):
                segment_idx = char_to_segment[mid_char]
                segment = segments[segment_idx]
                
                if segment.get("learn", False):
                    weight = segment.get("weight", 1.0)
                    loss_mask[token_idx] = weight
        
        # Collect segment info
        for i, segment in enumerate(segments):
            if segment.get("learn", False):
                processed_segments.append({
                    "segment_idx": i,
                    "content_preview": segment["content"][:50] + "..." if len(segment["content"]) > 50 else segment["content"],
                    "weight": segment.get("weight", 1.0),
                    "type": segment.get("segment_type", "unknown")
                })
        
        # Apply padding/truncation
        input_ids, attention_mask, loss_mask = self._apply_padding_truncation(
            input_ids, attention_mask, loss_mask
        )
        
        region_info = {
            "total_segments": len(segments),
            "learning_segments": len(processed_segments),
            "segments": processed_segments,
            "format": "structured"
        }
        
        return input_ids, attention_mask, loss_mask, region_info
    
    def _char_to_token_precise(self, offset_mapping, start_char, end_char):
        """Precise character to token position conversion"""
        start_token = 0
        end_token = len(offset_mapping)
        
        for i, (token_start, token_end) in enumerate(offset_mapping):
            if token_start <= start_char < token_end:
                start_token = i
            if token_start < end_char <= token_end:
                end_token = i + 1
                break
        
        return start_token, end_token
    
    def _parse_learn_tags(self, text):
        """Parse <LEARN> tags"""
        regions = []
        clean_text = text
        offset = 0
        
        # Regex to match LEARN tags
        pattern = r'<LEARN\s+([^>]+)>(.*?)</LEARN>'
        
        for match in re.finditer(pattern, text, re.DOTALL):
            # Parse attributes
            attrs_str = match.group(1)
            content = match.group(2)
            
            # Parse attributes
            attrs = {}
            for attr_match in re.finditer(r'(\w+)="([^"]*)"', attrs_str):
                attrs[attr_match.group(1)] = attr_match.group(2)
            
            # Calculate position in cleaned text
            tag_start = match.start() - offset
            content_start = tag_start
            content_end = content_start + len(content)
            
            regions.append({
                "start_char": content_start,
                "end_char": content_end,
                "weight": float(attrs.get("weight", "1.0")),
                "type": attrs.get("type", "unknown"),
                "content": content
            })
            
            # Update offset (remove tag length)
            tag_length = len(match.group(0)) - len(content)
            offset += tag_length
        
        # Remove all tags, keep content
        clean_text = re.sub(pattern, r'\2', text, flags=re.DOTALL)
        
        return regions, clean_text
    
    def _apply_padding_truncation(self, input_ids, attention_mask, loss_mask):
        """Apply padding/truncation consistent with base dataset"""
        sequence_length = input_ids.shape[0]
        
        if sequence_length < self.max_length:
            # Padding
            padded_input_ids = torch.ones(size=(self.max_length - sequence_length,), dtype=input_ids.dtype) * self.tokenizer.pad_token_id
            padded_attention_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=attention_mask.dtype)
            padded_loss_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=loss_mask.dtype)
            
            input_ids = torch.cat((input_ids, padded_input_ids))
            attention_mask = torch.cat((attention_mask, padded_attention_mask))
            loss_mask = torch.cat((loss_mask, padded_loss_mask))
            
        elif sequence_length > self.max_length:
            # Truncation
            if self.truncation == "left":
                input_ids = input_ids[-self.max_length:]
                attention_mask = attention_mask[-self.max_length:]
                loss_mask = loss_mask[-self.max_length:]
            elif self.truncation == "right":
                input_ids = input_ids[:self.max_length]
                attention_mask = attention_mask[:self.max_length]
                loss_mask = loss_mask[:self.max_length]
            elif self.truncation == "error":
                raise NotImplementedError(f"{sequence_length=} is larger than {self.max_length=}")
            else:
                raise NotImplementedError(f"Unknown truncation method {self.truncation}")
        
        return input_ids, attention_mask, loss_mask
    
    def _normalize_weights(self, loss_mask):
        """Weight normalization"""
        active_mask = loss_mask > 0
        if active_mask.sum() > 0:
            active_weights = loss_mask[active_mask]
            normalized_weights = active_weights / active_weights.mean()
            loss_mask[active_mask] = normalized_weights
        return loss_mask
