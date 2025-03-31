import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, f1_score, roc_auc_score, roc_curve, auc
from dataset3 import SpineDataset3
import timm_3d
import argparse
from tqdm import tqdm
import wandb
from torchvision import transforms

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train a 3D model for regression.')
parser.add_argument('--save_dir', type=str, default='/code/timm_3d_classification/TotalSegmentator_ckpt/Resnet50_tinycheckpoints_test_pretrain+balce_0.65', help='Directory to save model checkpoints.')
parser.add_argument('--train_dir', type=str, default='/code/TotalSegmentator_train', help='Path to the directory containing .nii.gz files.')
parser.add_argument('--val_dir', type=str, default='/code/TotalSegmentator_val_test', help='Path to the directory containing .nii.gz files.')
parser.add_argument('--test_dir', type=str, default='/code/TotalSegmentator_test', help='Path to the directory containing .nii.gz files.')
parser.add_argument('--label_file', type=str, default='/code/DEXA_vs_CT_subject_ID_and_T-score_delete377.xlsx', help='Path to the Excel file containing Subject IDs and T-scores.')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training.')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer.')
parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs to train the model.')
parser.add_argument('--logit_adj_post', type=int, default=1, choices=[0, 1], help='adjust logits post hoc')
parser.add_argument('--logit_adj_train', type=int, default=1, choices=[0, 1], help='adjust logits in trainingc')
parser.add_argument('--tro_train', default=0.65, type=float, help='tro for logit adj train')
parser.add_argument('--name', type=str, default='Res50_classification_bs8_lr0.001_totalsegemnetar_test_pretrain+balce_0.65', help='WandB display name.')

args = parser.parse_args()

wandb.init(entity = 'visual-intelligence-laboratory', project = 'Austin', name = args.name)
# wandb.init(project = 'Austin', name = args.name)
data_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30)
    ])

# Define dataset 
train_dataset = SpineDataset3(args.train_dir, args.label_file)
val_dataset = SpineDataset3(args.val_dir, args.label_file)
test_dataset = SpineDataset3(args.test_dir, args.label_file)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# Define the model, loss function, and optimizer
model = timm_3d.create_model('resnet50.a1_in1k', pretrained=True, num_classes=3)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

# Move model to GPU if available
model.to(device)

# Function to calculate metrics
def calculate_metrics(preds, labels, probs):
    accuracy = accuracy_score(labels, preds)
    sensitivity = recall_score(labels, preds, average=None)
    f1 = f1_score(labels, preds, average='weighted')
    auc_score_ovr = roc_auc_score(labels, probs, multi_class='ovr')

    cm = confusion_matrix(labels, preds)
    specificity_list = []

    for i in range(len(cm)):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - tp)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificity_list.append(specificity)

    mean_sensitivity = np.mean(sensitivity.tolist())
    mean_specificity = np.mean(specificity_list)

    # Calculate ROC curves
    auc_scores = []
    for i in range(probs.shape[1]):
        auc_score = roc_auc_score(labels == i, probs[:, i])
        auc_scores.append(auc_score)

    return accuracy, mean_sensitivity, mean_specificity, f1, auc_scores, auc_score_ovr

# compute the base probabilities
def compute_adjustment(train_loader, tro, device):
    """compute the base probabilities"""

    label_freq = {}
    for i, (inputs, target) in enumerate(train_loader):
        target = target.to(device)
        for j in target:
            key = int(j.item())
            label_freq[key] = label_freq.get(key, 0) + 1
    label_freq = dict(sorted(label_freq.items()))
    label_freq_array = np.array(list(label_freq.values()))
    label_freq_array = label_freq_array / label_freq_array.sum()
    adjustments = np.log(label_freq_array ** tro + 1e-12)
    adjustments = torch.from_numpy(adjustments)
    adjustments = adjustments.to(device)
    return adjustments

# logits_adjustments
logit_adjustments = compute_adjustment(train_loader, args.tro_train, device)

# Define training and evaluation phases
def train_or_evaluate(model, dataloader, criterion, optimizer, device, train=True):
    if train:
        model.train()
    else:
        model.eval()
    
    running_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []
    
    for inputs, labels in tqdm(dataloader, desc="Training" if train else "Evaluating"):
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.set_grad_enabled(train):
            outputs = model(inputs)
            
            if train:
                outputs = outputs + logit_adjustments
                loss = criterion(outputs, labels)
            else:
                loss = criterion(outputs + logit_adjustments, labels)
                outputs = outputs - logit_adjustments

            probs = nn.Softmax(dim=1)(outputs)
            preds = outputs.argmax(dim=1)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            all_preds.extend(preds.cpu().detach().numpy())
            all_labels.extend(labels.cpu().detach().numpy())
            all_probs.extend(probs.cpu().detach().numpy())
    
    loss = running_loss / len(dataloader.dataset)
    metrics = calculate_metrics(np.array(all_preds), np.array(all_labels), np.array(all_probs))
    return loss, metrics


# Training loop
for epoch in range(args.num_epochs):
    train_loss, train_metrics = train_or_evaluate(model, train_loader, criterion, optimizer, device, train=True)
    val_loss, val_metrics = train_or_evaluate(model, val_loader, criterion, optimizer, device, train=False)
    
    train_accuracy, train_sensitivity, train_specificity, train_f1, train_auc, train_auc_ovr = train_metrics
    val_accuracy, val_sensitivity, val_specificity, val_f1, val_auc, val_auc_ovr = val_metrics
    
    # Logging
    wandb.log({
        'epoch': epoch + 1,
        'train_loss': train_loss,
        'train_accuracy': train_accuracy,
        'train_sensitivity': train_sensitivity,
        'train_specificity': train_specificity,
        'train_f1': train_f1,
        'train_auc': train_auc,
        'train_auc_ovr':train_auc_ovr,
        'val_loss': val_loss,
        'val_accuracy': val_accuracy,
        'val_sensitivity': val_sensitivity,
        'val_specificity': val_specificity,
        'val_f1': val_f1,
        'val_auc': val_auc,
        'val_auc_ovr':val_auc_ovr,
    })

    print(f"Epoch {epoch + 1}/{args.num_epochs}")
    auc_str_train = ", ".join(f"{auc:.4f}" for auc in train_auc)
    print(f"Training Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}, Sensitivity: {train_sensitivity:.4f}, Specificity: {train_specificity:.4f}, F1: {train_f1:.4f}, AUC: [{auc_str_train}], AUC_OVR: {train_auc_ovr:.4f}")
    auc_str_val = ", ".join(f"{auc:.4f}" for auc in val_auc)
    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, Sensitivity: {val_sensitivity:.4f}, Specificity: {val_specificity:.4f}, F1: {val_f1:.4f}, AUC: [{auc_str_val}], AUC_OVR: {val_auc_ovr:.4f}")

    # Save checkpoint
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    torch.save(model.state_dict(), os.path.join(args.save_dir, f'epoch_{epoch+1}.ckpt'))
    scheduler.step()
