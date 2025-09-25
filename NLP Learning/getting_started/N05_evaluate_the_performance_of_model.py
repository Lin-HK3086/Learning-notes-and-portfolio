import torch.nn.functional as F
from transformers import DistilBertForSequenceClassification

from N03_text_preprocessing import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=2  # Binary classification
    )
model.load_state_dict(torch.load('D:\Python Program\\NLP Learning\getting_started\models\model01.pth', map_location=device))
model.to(device)

# Process test data
test_input_ids, test_attention_masks = tokenize_text(test_df['cleaned_text'].values, tokenizer)

# Create test dataloader
test_dataset = TensorDataset(test_input_ids, test_attention_masks)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Generate predictions
model.eval()
test_preds = []
test_probs = []

with torch.no_grad():
    for batch in test_loader:
        # Unpack and move batch to device
        input_ids, attention_mask = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        logits = outputs.logits
        probs = F.softmax(logits, dim=1)

        # Get predictions
        _, preds = torch.max(logits, dim=1)

        # Store predictions and probabilities
        test_preds.extend(preds.cpu().numpy())
        test_probs.extend(probs[:, 1].cpu().numpy())

# Add predictions to test dataframe
test_df['predicted_target'] = test_preds
test_df['disaster_probability'] = test_probs

# Display a sample of predictions
print("Sample predictions on the test set:")
sample_results = test_df[['text', 'predicted_target', 'disaster_probability']].sample(10)
print(sample_results)