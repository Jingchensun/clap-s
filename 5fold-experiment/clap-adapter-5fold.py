import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from msclap import CLAP
from esc50_dataset import ESC50
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

def train_one_epoch(clap_model, model, dataloader, text_embeddings, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(dataloader):
        x, _, one_hot_target = batch
        one_hot_target = one_hot_target.view(one_hot_target.size(0), -1).to(device)
        
        audio_embeddings = clap_model.get_audio_embeddings(x, resample=True).to(device)
        
        audio_embeddings = model(audio_embeddings)
        
        similarity = clap_model.compute_similarity(audio_embeddings, text_embeddings)
        
        loss = F.cross_entropy(similarity, one_hot_target.argmax(dim=1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
        
    average_loss = total_loss / num_batches
    return average_loss

def evaluate(clap_model, model, dataloader, text_embeddings, device):
    model.eval()
    y_preds, y_labels = [], []
    
    for batch in tqdm(dataloader):
        x, _, one_hot_target = batch
        one_hot_target = one_hot_target.view(one_hot_target.size(0), -1).to(device)
        
        audio_embeddings = clap_model.get_audio_embeddings(x, resample=True).to(device)
        audio_embeddings = model(audio_embeddings)
        
        similarity = clap_model.compute_similarity(audio_embeddings, text_embeddings)
        
        y_pred = F.softmax(similarity.detach().cpu(), dim=1).numpy()
        y_preds.append(y_pred)
        y_labels.append(one_hot_target.detach().cpu().numpy())
    
    y_labels, y_preds = np.concatenate(y_labels, axis=0), np.concatenate(y_preds, axis=0)

    y_preds_single = np.argmax(y_preds, axis=1)
    y_labels_single = np.argmax(y_labels, axis=1)
    
    acc = accuracy_score(y_labels_single, y_preds_single)
    return acc

def main(root_path, test_file, model_version, use_cuda, download_dataset, epochs, save_path, checkpoint_path=None, eval=False):
    dataset = ESC50(root=root_path, test_file=test_file)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    prompt = 'this is the sound of '
    y = [prompt + x for x in dataset.classes]

    device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')

    clap_model = CLAP(version=model_version, use_cuda=use_cuda)
    text_embeddings = clap_model.get_text_embeddings(y).to(device)

    test_file_name = os.path.splitext(os.path.basename(test_file))[0]
    log_file = 'log/' + f"{test_file_name}_5fold.txt"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    print(checkpoint_path)
    if eval and checkpoint_path and os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        model = Adapter(1024, 4).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        acc = evaluate(clap_model, model, DataLoader(dataset, batch_size=256), text_embeddings, device)
        print(f"Accuracy of the loaded model: {acc}")
        with open(log_file, 'a') as f:
            f.write(f"test_file: {test_file}\n")
            f.write(f"checkpoint path: {checkpoint_path}\n")
            f.write(f"Accuracy of the loaded model: {acc}\n")
            f.write(f"\n")
        return
    else:
        for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
            print(f'Fold {fold + 1}')
            train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
            val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
            train_loader = DataLoader(dataset, batch_size=32, sampler=train_sampler)
            val_loader = DataLoader(dataset, batch_size=16, sampler=val_sampler)

            model = Adapter(1024, 4).to(device)

            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2, betas=(0.9, 0.999), eps=1e-8)
            
            # Set up the warmup and cosine annealing learning rate scheduler
            total_steps = len(train_loader) * epochs
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, 
                max_lr=0.01,  # A higher learning rate during the warmup phase
                total_steps=total_steps, 
                pct_start=0.2,  # First 20% of the steps for warmup
                anneal_strategy='cos', 
                final_div_factor=100  # The final learning rate is 1/100th of the initial learning rate
            )
            best_acc = 0.0
            for epoch in range(epochs):
                print(f'Epoch {epoch + 1}/{epochs}')
                
                average_loss = train_one_epoch(clap_model, model, train_loader, text_embeddings, optimizer, scheduler, device)
                acc = evaluate(clap_model, model, val_loader, text_embeddings, device)
                
                print(f'Fold {fold + 1} Epoch {epoch + 1} Average Loss: {average_loss}')
                print(f'Fold {fold + 1} Epoch {epoch + 1} Accuracy: {acc}')
                
                with open(log_file, 'a') as f:
                    f.write(f'Fold {fold + 1} Epoch {epoch + 1} Average Loss: {average_loss}\n')
                    f.write(f'Fold {fold + 1} Epoch {epoch + 1} Accuracy: {acc}\n')
                    
                if acc > best_acc:
                    best_acc = acc
                    save_dir = os.path.dirname(save_path)
                    os.makedirs(save_dir, exist_ok=True)
                    model_save_path = os.path.join(save_dir, f"{test_file_name}_fold{fold + 1}_best_acc.pth")
                    torch.save(model.state_dict(), model_save_path)
                    print(f'Best model saved with accuracy: {best_acc}')
                    with open(log_file, 'a') as f:
                        f.write(f'Best model saved with accuracy: {best_acc}\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run CLAP zero-shot classification on ESC50 dataset with 5-fold cross-validation')
    parser.add_argument('--root_path', type=str, required=True, help='Root path to ESC-50 dataset')
    parser.add_argument('--test_file', type=str, required=True, help='Root path to recorded dataset')
    parser.add_argument('--model_version', type=str, default='2023', help='Version of CLAP model to use')
    parser.add_argument('--use_cuda', type=bool, default=True, help='Use CUDA for computation')
    parser.add_argument('--download_dataset', type=bool, default=False, help='Download the ESC-50 dataset if not available')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--save_path', type=str, default='checkpoint/best_model.pth', help='Path to save the best model')
    parser.add_argument('--checkpoint_path', type=str, help='Path to an existing checkpoint for evaluation')
    parser.add_argument('--eval', type=bool, default=False, help='Evaluate the model from a checkpoint')

    args = parser.parse_args()
    main(**vars(args))
