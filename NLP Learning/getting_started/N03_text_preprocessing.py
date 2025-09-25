import re
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer

from N01_reproducibility_and_data_download import *

def clean_text(text):
    """Basic text cleaning function"""
    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove Twitter-specific content
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'#', '', text)     # Remove hashtag symbols but keep the words
    text = re.sub(r'rt\s+', '', text)  # Remove RT (retweet) indicators

    # Remove HTML entities (like &)
    text = re.sub(r'&\w+;', '', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Handle encoding errors like 'Û'
    text = re.sub(r'Û', '', text)

    # Remove special characters and digits
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Apply cleaning to the text column
train_df['cleaned_text'] = train_df['text'].apply(clean_text)
test_df['cleaned_text'] = test_df['text'].apply(clean_text)

# Display a few examples of cleaned text
# print("Original vs Cleaned:")
# for i in range(3):
#     print(f"Original: {train_df['text'].iloc[i]}")
#     print(f"Cleaned: {train_df['cleaned_text'].iloc[i]}")
#     print()

# Turning Text into Tensors

# Load pretrained tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Example of tokenization
example_text = "PyTorch is great for NLP"
tokens = tokenizer.tokenize(example_text)
token_ids = tokenizer.encode(example_text)

# Creat a function to tokenize our text data
def tokenize_text(texts, tokenizer, max_length=128):
    """
    Tokenize a list of texts using the provided tokenizer
    Returns input IDs and attention masks
    """
    # Tokenize all texts at once
    encodings = tokenizer(
        list(texts),
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    return encodings['input_ids'], encodings['attention_mask']

# Creating Datasets and Dataloaders
# Split data into train and validation sets
train_texts, val_texts, train_targets, val_targets = train_test_split(
    train_df['cleaned_text'].values,
    train_df['target'].values,
    test_size=0.1,
    random_state=42,
    stratify=train_df['target']  # Maintain class distribution
)

# Set the batch size for effecient training
batch_size = 16

# Process training data
train_input_ids, train_attention_masks = tokenize_text(train_texts, tokenizer)
train_targets = torch.tensor(train_targets, dtype=torch.long)

# Process validation data
val_input_ids, val_attention_masks = tokenize_text(val_texts, tokenizer)
val_targets = torch.tensor(val_targets, dtype=torch.long)

# Create tensor datasets
train_dataset = TensorDataset(
    train_input_ids,
    train_attention_masks,
    train_targets
)
val_dataset = TensorDataset(
    val_input_ids,
    val_attention_masks,
    val_targets
)

# Create dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size
)

# Look at a single batch
batch = next(iter(train_loader))
input_ids, attention_mask, targets = batch

if __name__ == '__main__':
    print(f"Original text: {example_text}")
    print(f"Tokens: {tokens}")
    print(f"Token IDs: {token_ids}\n")

    print(f"Training texts: {len(train_texts)}")
    print(f"Validation texts: {len(val_texts)}\n")

    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Attention mask shape: {attention_mask.shape}")
    print(f"Targets shape: {targets.shape}")