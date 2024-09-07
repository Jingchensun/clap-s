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
import yaml

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def build_cache_model(cfg, clap_model, train_loader_cache, device, audio_dataset, shot):
    cache_dir = os.path.join('cache_dir', audio_dataset)
    os.makedirs(cache_dir, exist_ok=True)

    if cfg['load_cache'] == False:    
        cache_keys = []
        cache_values = []

        with torch.no_grad():
            train_features = []
            print('Processing cache without augmentation...')
            for batch in tqdm(train_loader_cache):
                file_path, target, one_hot_target = batch
                one_hot_target = one_hot_target.view(one_hot_target.size(0), -1).to(device)

                audio_embeddings = clap_model.get_audio_embeddings(file_path, resample=True).to(device)
                train_features.append(audio_embeddings)
                cache_values.append(one_hot_target)
            
            cache_keys = torch.cat(train_features, dim=0)
            cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
            cache_keys = cache_keys.permute(1, 0)
            cache_values = torch.cat(cache_values, dim=0)
            torch.save(cache_keys, os.path.join(cache_dir, f"shot{shot}_train-keys.pt"))
            torch.save(cache_values, os.path.join(cache_dir, f"shot{shot}_train-values.pt"))

    else:
        cache_keys = torch.load(os.path.join(cache_dir, f"shot{shot}_train-keys.pt"))
        cache_values = torch.load(os.path.join(cache_dir, f"shot{shot}_train-values.pt"))

    return cache_keys, cache_values

def pre_load_features(cfg, split, clap_model, loader, device, audio_dataset):
    cache_dir = os.path.join('cache_dir', audio_dataset)
    os.makedirs(cache_dir, exist_ok=True)

    if cfg['load_pre_feat'] == False:
        features, labels = [], []

        with torch.no_grad():
            for batch in tqdm(loader):
                file_path, _, one_hot_target = batch
                one_hot_target = one_hot_target.view(one_hot_target.size(0), -1).to(device)
                
                audio_embeddings = clap_model.get_audio_embeddings(file_path, resample=True).to(device)
                features.append(audio_embeddings)
                labels.append(one_hot_target)
            features = torch.cat(features, dim=0)   
            features /= features.norm(dim=-1, keepdim=True)
            labels = torch.cat(labels, dim=0)

            torch.save(features, os.path.join(cache_dir, f"{split}_keys.pt"))
            torch.save(labels, os.path.join(cache_dir, f"{split}_values.pt"))
    else:
        features = torch.load(os.path.join(cache_dir, f"{split}_keys.pt"))
        labels = torch.load(os.path.join(cache_dir, f"{split}_values.pt"))
    
    return features, labels

def search_hp(cfg, cache_keys, cache_values, features, labels, clip_weights, adapter=None):

    if cfg['search_hp'] == True:
    
        beta_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in range(cfg['search_step'][0])]
        alpha_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in range(cfg['search_step'][1])]

        best_acc = 0
        best_beta, best_alpha = 0, 0

        for beta in beta_list:
            for alpha in alpha_list:
                if adapter:
                    affinity = adapter(features)
                else:
                    affinity = features @ cache_keys

                cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
                clip_logits = 33.3795 * features @ clip_weights

                tip_logits = clip_logits + cache_logits * alpha
                y_preds = F.softmax(tip_logits.detach().cpu(), dim=1).numpy()
                y_labels = labels.cpu().numpy()
                acc = accuracy_score(np.argmax(y_labels, axis=1), np.argmax(y_preds, axis=1))
                if acc > best_acc:
                    print("New best setting, beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}".format(beta, alpha, acc))
                    best_acc = acc
                    best_beta = beta
                    best_alpha = alpha

        print("\nAfter searching, the best accuarcy: {:.2f}.\n".format(best_acc))

    return best_beta, best_alpha

def run_tip_adapter(cfg, clap_model, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights, log_file):
    
    with open(log_file, 'a') as log:
        log.write("\n-------- Searching hyperparameters on the val set. --------\n")
        logit_scale = 33.3795

        # Zero-shot CLIP
        clip_logits = logit_scale * val_features @ clip_weights
        y_preds = F.softmax(clip_logits.cpu(), dim=1).numpy()
        y_labels = val_labels.cpu().numpy()
        acc = accuracy_score(np.argmax(y_labels, axis=1), np.argmax(y_preds, axis=1))

        # log.write("\n**** Zero-shot CLIP's val accuracy: {:.2f}. ****\n".format(acc))

        # Tip-Adapter
        beta, alpha = cfg['init_beta'], cfg['init_alpha']
        
        affinity = val_features @ cache_keys 
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        
        tip_logits = clip_logits + cache_logits * alpha

        y_preds = F.softmax(tip_logits.cpu(), dim=1).numpy()
        y_labels = val_labels.cpu().numpy()
        acc = accuracy_score(np.argmax(y_labels, axis=1), np.argmax(y_preds, axis=1))
        # log.write("**** Tip-Adapter's val accuracy: {:.2f}. ****\n".format(acc))

        # Search Hyperparameters
        best_beta, best_alpha = search_hp(cfg, cache_keys, cache_values, val_features, val_labels, clip_weights)


        log.write("\n-------- Evaluating on the test set. --------\n")

        # Zero-shot CLIP
        clip_logits = logit_scale * test_features @ clip_weights
        y_preds = F.softmax(clip_logits.cpu(), dim=1).numpy()
        y_labels = test_labels.cpu().numpy()
        acc = accuracy_score(np.argmax(y_labels, axis=1), np.argmax(y_preds, axis=1))
        log.write("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(acc))

        # Tip-Adapter    
        affinity = test_features @ cache_keys
        cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values
        
        tip_logits = clip_logits + cache_logits * best_alpha
        y_preds = F.softmax(tip_logits.cpu(), dim=1).numpy()
        y_labels = test_labels.cpu().numpy()
        acc = accuracy_score(np.argmax(y_labels, axis=1), np.argmax(y_preds, axis=1))
        log.write("**** Tip-Adapter's test accuracy: {:.2f}. ****\n".format(acc))



def run_tip_adapter_F(cfg, clap_model, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights, train_loader_F, device, model_save_path, log_file):
    with open(log_file, 'a') as log:
        # Enable the cached keys to be learnable
        adapter = nn.Linear(cache_keys.shape[0], cache_keys.shape[1], bias=False).to(device)
        adapter.weight = nn.Parameter(cache_keys.t())
        
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=5e-4, eps=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['train_epoch'] * len(train_loader_F))

        # optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3, weight_decay=1e-2)
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(
        # optimizer, max_lr=0.001, total_steps=len(train_loader_F) * cfg['train_epoch'], pct_start=0.2, anneal_strategy='cos', final_div_factor=100)
        
        beta, alpha = cfg['init_beta'], cfg['init_alpha']
        correct_samples, all_samples = 0, 0
        best_acc, best_epoch = 0.0, 0

        for train_idx in range(cfg['train_epoch']):
            # Train
            adapter.train()
            correct_samples, all_samples = 0, 0
            print('Train Epoch: {:} / {:}'.format(train_idx, cfg['train_epoch']))

            epoch_loss = 0.0
            for batch in tqdm(train_loader_F):
                file_path, target, one_hot_target = batch
                one_hot_target = one_hot_target.view(one_hot_target.size(0), -1).to(device)
                with torch.no_grad():
                    audio_embeddings = clap_model.get_audio_embeddings(file_path, resample=True).to(device)
                    audio_embeddings = audio_embeddings / torch.norm(audio_embeddings, dim=-1, keepdim=True)

                affinity = adapter(audio_embeddings)
                cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values

                logit_scale = 33.3795
                clip_logits = logit_scale * audio_embeddings @ clip_weights
                tip_logits = clip_logits + cache_logits * alpha

                y_preds = F.softmax(tip_logits.detach().cpu(), dim=1).numpy()
                y_labels = one_hot_target.cpu().numpy()
                acc = accuracy_score(np.argmax(y_labels, axis=1), np.argmax(y_preds, axis=1))
                correct_samples += acc * len(tip_logits)
                all_samples += len(tip_logits)

                loss = F.cross_entropy(tip_logits, one_hot_target.argmax(dim=1))
                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

            avg_epoch_loss = epoch_loss / len(train_loader_F)
            print("Epoch: {:}, Loss: {:.6f}".format(train_idx, avg_epoch_loss))
            current_lr = scheduler.get_last_lr()[0]
            print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples, correct_samples, all_samples, avg_epoch_loss))

            # Eval
            adapter.eval()

            affinity = adapter(test_features)
            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
            clip_logits = 33.3795 * test_features @ clip_weights
            tip_logits = clip_logits + cache_logits * alpha
            y_preds = F.softmax(tip_logits.detach().cpu(), dim=1).numpy()
            y_labels = test_labels.cpu().numpy()
            acc = accuracy_score(np.argmax(y_labels, axis=1), np.argmax(y_preds, axis=1))

            print("**** Tip-Adapter-F's test accuracy: {:.2f}. ****\n".format(acc))
            if acc > best_acc:
                best_acc = acc
                best_epoch = train_idx
                torch.save(adapter.weight, model_save_path)

        adapter.weight = torch.load(model_save_path)
        log.write(f"**** After fine-tuning, Tip-Adapter-F's best test accuracy: {best_acc:.2f}, at epoch: {best_epoch}. ****\n")

        print("\n-------- Searching hyperparameters on the val set. --------")

        # Search Hyperparameters
        best_beta, best_alpha = search_hp(cfg, cache_keys, cache_values, val_features, val_labels, clip_weights, adapter=adapter)
        log.write(f"**** The best hyperparameters of beta is {best_beta:.2f}, and alpha is {best_alpha:.2f}. ****\n")
        print("\n-------- Evaluating on the test set. --------")
    
        affinity = adapter(test_features)
        cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values
        
        clip_logits = 33.3795 * test_features @ clip_weights
        
        tip_logits = clip_logits + cache_logits * best_alpha

        y_preds = F.softmax(tip_logits.detach().cpu(), dim=1).numpy()
        y_labels = test_labels.cpu().numpy()
        acc = accuracy_score(np.argmax(y_labels, axis=1), np.argmax(y_preds, axis=1))
        print("**** Tip-Adapter-F's test accuracy: {:.2f}. ****\n".format(max(best_acc, acc)))
        log.write("**** After Searching hyperparameters, Tip-Adapter-F's test accuracy: {:.2f}. ****\n".format(max(best_acc, acc)))


def main(root_path, audio_dataset, model_version, use_cuda, save_path, seed, shot, checkpoint_path=None, eval=False):
    set_seed(seed)


    # train_set = ESC50(root=root_path, subset='train', audio_dataset=audio_dataset, shot=shot)
    # val_set = ESC50(root=root_path, subset='val', audio_dataset=audio_dataset)
    # test_set = ESC50(root=root_path, subset='test', audio_dataset=audio_dataset)

    data_path = '/home/jingchen/data/fiber-data/real/'
    train_set = Fiber(root=data_path, subset='train', audio_dataset=audio_dataset, shot=shot, seed = seed)
    val_set = Fiber(root=data_path, subset='val', audio_dataset=audio_dataset)
    test_set = Fiber(root=data_path, subset='test', audio_dataset=audio_dataset)

    train_loader_cache =  DataLoader(train_set, batch_size=64, shuffle=False)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64)
    test_loader = DataLoader(test_set, batch_size=64)

    prompt = 'this is an audio of '
    y = [prompt + x for x in train_set.classes]

    device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
    clap_model = CLAP(version=model_version, use_cuda=use_cuda)
    text_embeddings = clap_model.get_text_embeddings(y).to(device)
    text_embeddings = text_embeddings / torch.norm(text_embeddings, dim=-1, keepdim=True)
    text_embeddings = text_embeddings.T

    log_file = f'log-fiber-support/{os.path.splitext(os.path.basename(audio_dataset))[0]}.txt'
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    cfg = yaml.load(open('configs/oxford_pets.yaml', 'r'), Loader=yaml.Loader)

    # Construct the cache model by few-shot training set
    print("\nConstructing cache model by few-shot visual features and labels.")
    cache_keys, cache_values = build_cache_model(cfg, clap_model, train_loader_cache, device, audio_dataset, shot)

    # Pre-load val features
    print("\nLoading visual features and labels from val set.")
    val_features, val_labels = pre_load_features(cfg, "val", clap_model, val_loader, device, audio_dataset)

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(cfg, "test", clap_model, test_loader, device, audio_dataset)

    # # ------------------------------------------ Tip-Adapter ------------------------------------------
    run_tip_adapter(cfg, clap_model, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, text_embeddings, log_file)

     # ------------------------------------------ Tip-Adapter-F ------------------------------------------
    # save_dir = os.path.dirname(save_path)
    # os.makedirs(save_dir, exist_ok=True)
    # model_save_path = os.path.join(save_dir, f"{audio_dataset}_{shot}_best_acc.pth")
    # run_tip_adapter_F(cfg, clap_model, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, text_embeddings, train_loader, device, model_save_path, log_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run CLAP zero-shot classification on ESC50 dataset with train, val, test split')
    parser.add_argument('--root_path', type=str, required=True, help='Root path to ESC-50 dataset')
    parser.add_argument('--audio_dataset', type=str, required=True, help='Path to the audio dataset')
    parser.add_argument('--model_version', type=str, default='2023', help='Version of CLAP model to use')
    parser.add_argument('--use_cuda', type=bool, default=True, help='Use CUDA for computation')
    parser.add_argument('--save_path', type=str, default='checkpoint/best_model.pth', help='Path to save the best model')
    parser.add_argument('--checkpoint_path', type=str, help='Path to an existing checkpoint for evaluation')
    parser.add_argument('--eval', type=bool, default=False, help='Evaluate the model from a checkpoint')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--shot', type=int, required=True, help='Number of shots for few-shot learning')

    args = parser.parse_args()
    main(**vars(args))
