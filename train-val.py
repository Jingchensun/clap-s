import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from msclap import CLAP
from esc50_dataset_split import ESC50
from fiber_dataset import Fiber
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

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
        return self.fc(x)

def train_one_epoch(clap_model, model, dataloader, text_embeddings, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader):
        file_path, target, one_hot_target = batch
        one_hot_target = one_hot_target.view(one_hot_target.size(0), -1).to(device)
        
        audio_embeddings = clap_model.get_audio_embeddings(file_path, resample=True).to(device)
        audio_embeddings = model(audio_embeddings)
        similarity = clap_model.compute_similarity(audio_embeddings, text_embeddings)
        loss = F.cross_entropy(similarity, one_hot_target.argmax(dim=1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def evaluate(clap_model, model, dataloader, text_embeddings, device, class_names):
    model.eval()
    y_preds, y_labels = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            file_path, _, one_hot_target = batch
            one_hot_target = one_hot_target.view(one_hot_target.size(0), -1).to(device)
            
            audio_embeddings = clap_model.get_audio_embeddings(file_path, resample=True).to(device)
            audio_embeddings = model(audio_embeddings)
            similarity = clap_model.compute_similarity(audio_embeddings, text_embeddings)
            
            y_preds.append(F.softmax(similarity.cpu(), dim=1).numpy())
            y_labels.append(one_hot_target.cpu().numpy())
    
    y_preds = np.concatenate(y_preds, axis=0)
    y_labels = np.concatenate(y_labels, axis=0)

    overall_acc = accuracy_score(np.argmax(y_labels, axis=1), np.argmax(y_preds, axis=1))
    per_class_acc = precision_score(np.argmax(y_labels, axis=1), np.argmax(y_preds, axis=1), average=None)

    return overall_acc, per_class_acc

def main(root_path, audio_dataset, model_version, use_cuda, download_dataset, epochs, save_path, seed, checkpoint_path=None, eval=False):
    set_seed(seed)
    
    train_set = ESC50(root=root_path,  subset='train', audio_dataset=audio_dataset, shot=-1)
    val_set = ESC50(root=root_path, subset='val', audio_dataset=audio_dataset)
    test_set = ESC50(root=root_path, subset='test', audio_dataset=audio_dataset)

    # data_path = '/home/onsi/jsun/dataset/lbi_gunshot/'
    # train_set = Fiber(root=data_path, subset='train', audio_dataset=audio_dataset, shot=-1)
    # val_set = Fiber(root=data_path, subset='val', audio_dataset=audio_dataset)
    # test_set = Fiber(root=data_path, subset='test', audio_dataset=audio_dataset)

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64)
    test_loader = DataLoader(test_set, batch_size=64)

    prompt = 'this is the sound of '
    y = [prompt + x for x in train_set.classes]

    device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
    clap_model = CLAP(version=model_version, use_cuda=use_cuda)
    text_embeddings = clap_model.get_text_embeddings(y).to(device)

    log_file = f'log-recorded-fullshot/{os.path.splitext(os.path.basename(audio_dataset))[0]}.txt'
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    model = Adapter(1024, 4).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.001, total_steps=len(train_loader) * epochs, pct_start=0.2, anneal_strategy='cos', final_div_factor=100)

    if eval and checkpoint_path and os.path.isfile(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        test_acc, per_class_acc = evaluate(clap_model, model, test_loader, text_embeddings, device, test_set.classes)
        print(f"Accuracy of the loaded model: {test_acc}")
        with open(log_file, 'a') as f:
            f.write(f"audio_dataset: {audio_dataset}\n")
            f.write(f"checkpoint path: {checkpoint_path}\n")
            f.write(f"Accuracy of the loaded model: {test_acc}\n")
            f.write(f"\n")
        
        # Save per-class accuracy
        # per_class_log_file = f"log-single-task/{audio_dataset}_per_class_acc.txt"
        # with open(per_class_log_file, 'w') as f:
        #     for class_name, acc in zip(test_set.classes, per_class_acc):
        #         f.write(f"{class_name}: {acc}\n")

        return

    best_acc = 0.0
    for epoch in range(epochs):
        avg_loss = train_one_epoch(clap_model, model, train_loader, text_embeddings, optimizer, scheduler, device)
        val_acc, _ = evaluate(clap_model, model, val_loader, text_embeddings, device, val_set.classes)
        
        print(f'Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} - Validation Accuracy: {val_acc:.4f}')
        with open(log_file, 'a') as f:
            f.write(f'Epoch {epoch + 1} - Loss: {avg_loss:.4f} - Validation Accuracy: {val_acc:.4f}\n')

        if val_acc > best_acc:
            best_acc = val_acc
            save_dir = os.path.dirname(save_path)
            os.makedirs(save_dir, exist_ok=True)
            model_save_path = os.path.join(save_dir, f"{audio_dataset}_best_acc.pth")
            torch.save(model.state_dict(), model_save_path)
            print(f'Best model saved with accuracy: {best_acc:.4f}')
            with open(log_file, 'a') as f:
                f.write(f'Best model saved with accuracy: {best_acc:.4f}\n')

    model.load_state_dict(torch.load(model_save_path))
    test_acc, per_class_acc = evaluate(clap_model, model, test_loader, text_embeddings, device, test_set.classes)
    print(f'Test Accuracy: {test_acc:.4f}')
    with open(log_file, 'a') as f:
        f.write(f'Test Accuracy: {test_acc:.4f}\n')

    # # Save per-class accuracy
    # per_class_log_file = f"log-single-task/{audio_dataset}_per_class_acc.log"
    # with open(per_class_log_file, 'w') as f:
    #     for class_name, acc in zip(test_set.classes, per_class_acc):
    #         f.write(f"{class_name}: {acc}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run CLAP zero-shot classification on ESC50 dataset with train, val, test split')
    parser.add_argument('--root_path', type=str, required=True, help='Root path to ESC-50 dataset')
    parser.add_argument('--audio_dataset', type=str, required=True, help='Path to the audio dataset')
    parser.add_argument('--model_version', type=str, default='2023', help='Version of CLAP model to use')
    parser.add_argument('--use_cuda', type=bool, default=True, help='Use CUDA for computation')
    parser.add_argument('--download_dataset', type=bool, default=False, help='Download the ESC-50 dataset if not available')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--save_path', type=str, default='checkpoint/best_model.pth', help='Path to save the best model')
    parser.add_argument('--checkpoint_path', type=str, help='Path to an existing checkpoint for evaluation')
    parser.add_argument('--eval', type=bool, default=False, help='Evaluate the model from a checkpoint')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    args = parser.parse_args()
    main(**vars(args))
