import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score
from transformers import DistilBertForSequenceClassification

from N03_text_preprocessing import *


# Training function
def train_epoch(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    for batch in data_loader:
        # Unpack and move batch to device
        input_ids, attention_mask, targets = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        targets = targets.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=targets
        )

        loss = outputs.loss
        logits = outputs.logits

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Calculate accuracy
        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == targets)
        total_predictions += len(targets)

    # Calculate average loss and accuracy
    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions.double() / total_predictions

    return avg_loss, accuracy

# Evaluation function
def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for batch in data_loader:
            # Unpack and move batch to device
            input_ids, attention_mask, targets = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=targets
            )

            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()

            # Calculate accuracy
            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == targets)
            total_predictions += len(targets)

            # Store targets and predictions for F1 score
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Calculate average loss and accuracy
    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions.double() / total_predictions
    f1 = f1_score(all_targets, all_preds)

    return avg_loss, accuracy, f1

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load pretrained model
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=2  # Binary classification
    )

    # Move model to device
    model = model.to(device)

    # Set up optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    # Training loop
    epochs = 10
    best_f1 = 0

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")

        # Evaluate
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, device)
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}, Val F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            # In a real scenario, we'd save the model here
            # torch.save(model.state_dict(), "best_model.pt")
        print()

    # Safe the model
    torch.save(model.state_dict(), f"D:\Python Program\\NLP Learning\getting_started\models\model01.pth")

    # Detailed evaluation
    model.eval()
    all_targets = []
    all_preds = []
    all_probs = []  # For prediction probabilities

    with torch.no_grad():
        for batch in val_loader:
            # Unpack and move batch to device
            input_ids, attention_mask, targets = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            logits = outputs.logits
            probs = F.softmax(logits, dim=1)

            # Get predictions
            _, preds = torch.max(logits, dim=1)

            # Store targets, predictions, and probabilities
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # For positive class

    # Classification report
    print(classification_report(all_targets, all_preds, target_names=['Not Disaster', 'Disaster']))

    # Confusion matrix
    cm = pd.crosstab(
        pd.Series(all_targets, name='Actual'),
        pd.Series(all_preds, name='Predicted')
    )

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.show()