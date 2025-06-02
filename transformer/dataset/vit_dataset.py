import torch
from torch.utils.data import Dataset
from PIL import Image

from transformer.utils.vocab_gen import (
    clean_text,
    tokenize
)

class ViTDataset(Dataset):
    def __init__(
        self,
        dataframe,
        max_seq_length,
        image_transform,
        sos_token_id,
        eos_token_id,
        pad_token_id,
        unk_token_id,
        vocab,
        image_column = 'path',
        text_column = 'text',
    ):
        super().__init__()

        self.dataframe = dataframe
        self.max_seq_length = max_seq_length
        self.image_transform = image_transform
        self.sos_token_id = sos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.unk_token_id = unk_token_id
        self.vocab = vocab
        self.image_column = image_column
        self.text_column = text_column
    
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_path = self.dataframe.iloc[idx][self.image_column]
        out_text = self.dataframe.iloc[idx][self.text_column]

        pil_image = Image.open(image_path)
        image_tensor = self.image_transform(pil_image)

        out_text = clean_text(out_text)
        tokenized_word = tokenize(out_text)
        token_ids = [self.vocab.get(word, self.unk_token_id) for word in tokenized_word]

        decoder_input_list = [self.sos_token_id] + token_ids

        if len(decoder_input_list) > self.max_seq_length:
            decoder_input_list = decoder_input_list[:self.max_seq_length]

        padding_len_input = self.max_seq_length - len(decoder_input_list)
        decoder_input_ids = torch.tensor(decoder_input_list + [self.pad_token_id] * padding_len_input)

        tgt_out_list = token_ids + [self.eos_token_id]
        if len(tgt_out_list) > self.max_seq_length:
            tgt_out_list = tgt_out_list[:self.max_seq_length]
            if tgt_out_list[-1] != self.eos_token_id:
                tgt_out_list[-1] = self.eos_token_id

        padding_len_input = self.max_seq_length - len(tgt_out_list)
        decoder_out_ids = torch.tensor(tgt_out_list + [self.pad_token_id] * padding_len_input)

        return image_tensor, decoder_input_ids, decoder_out_ids


