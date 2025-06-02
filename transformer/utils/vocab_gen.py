import re
from collections import Counter
import json

VOCAB_SIZE_TARGET = 5000
PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"

# ! I haven't tested this part yet so use it carefully

def clean_text(text):
        # Just for vocab
        text = text.lower()
        text = re.sub(r'[^\w\s\.\,]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()

        return text
    
def tokenize(text):
    tokens = text.split(' ')
    return [token for token in tokens if token]

def generate_vocab( dfs=[], text_cols = 'text', folder_path='.'):
    all_token_for_vocab = []
    for df in dfs:
        for text in df[text_cols]:
            cleaned = clean_text(text)
            tokens = tokenize(cleaned)
            all_token_for_vocab.extend(tokens)
    token_counts = Counter(all_token_for_vocab)

    vocab = {
        PAD_TOKEN: 0,
        SOS_TOKEN: 1,
        EOS_TOKEN: 2,
        UNK_TOKEN: 3,
    }
    idx_counter = len(vocab)
    for word, _ in token_counts.most_common(VOCAB_SIZE_TARGET - len(vocab)):
        if word not in vocab:
            vocab[word] = idx_counter
            id_counter += 1

    id2token = {idx: token for token, idx in vocab.items()}

    save_vocab(folder_path, vocab, id2token)

def save_vocab(folder_path, vocab, id2token):
    with open(f'{folder_path}/vocab.json', 'w') as f:
        json.dump(vocab, f, indent=4)
    with open(f'{folder_path}id2token.json', 'w') as f:
            json.dump(id2token, f, indent=4)