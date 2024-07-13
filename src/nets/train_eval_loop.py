from tqdm import tqdm
import torch
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
import numpy as np


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(train_loader):
        images = images.to(device)
        labels = labels.float().to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(train_loader.dataset)
    train_f1 = f1_score(np.argmax(all_labels, axis=1), all_preds, average='macro')
    return epoch_loss, train_f1

def eval_model(model, dataloader, criterion, device, cm=False):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.float().to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    val_f1 = f1_score(np.argmax(all_labels, axis=1), all_preds, average='macro')
    
    if cm == False:
        return epoch_loss, val_f1
    if cm == True:
        return (
            epoch_loss,
            val_f1,
            confusion_matrix(np.argmax(all_labels, axis=1), all_preds),
            accuracy_score(np.argmax(all_labels, axis=1), all_preds),
            precision_score(np.argmax(all_labels, axis=1), all_preds),
            recall_score(np.argmax(all_labels, axis=1), all_preds),
        )