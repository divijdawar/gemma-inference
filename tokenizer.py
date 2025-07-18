from typing import List, Union, Sequence, Optional
import sentencepiece as spm
import os

#Constants 
max_len = 8192
vocab_size = 262144
n_layers = 35
model_dim = 2048
ple_dim = 256

class Tokenizer: 
    
    def __init__(self, model_path: str): 
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.model_path = model_path
        self.sp = spm.SentencePieceProcessor(model_file=self.model_path)
        self.bos_id = self.sp.piece_to_id("<bos>")
        self.eos_id = self.sp.piece_to_id("<eos>")
        self.pad_id = self.sp.piece_to_id("<pad>")
        self.unk_id = self.sp.unk_id()

    def encode(
        self, 
        text: Union[str, Sequence[str]], 
        special_tokens: bool = True, 
        max_len: Optional[int] = None, 
        padding: bool = False
    ) -> Union[List[int], List[List[int]]]:
        """
        Returns the python list of token ids
        """
        if max_len is None:
            max_len = self.max_len if hasattr(self, 'max_len') else 8192
            
        if isinstance(text, str): 
            text = [text]
            single_input = True 
        else:
            text = list(text)
            single_input = False 
        
        encoded_input: List[List[int]] = []

        for t in text: 
            ids: List[int] = list(self.sp.encode(t, out_type=int))

            if special_tokens: 
                ids = [self.bos_id] + ids + [self.eos_id]
            
            if padding and len(id) < max_len: 
                id.extend([self.pad_id] * (max_len) - len(id))

            encoded_input.append(ids)

        return encoded_input[0] if single_input else encoded_input