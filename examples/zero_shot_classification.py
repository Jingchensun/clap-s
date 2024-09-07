"""
This is an example using CLAP to perform zeroshot
    classification on ESC50 (https://github.com/karolpiczak/ESC-50).
"""

from msclap import CLAP
from esc50_dataset import ESC50
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score

# Load dataset
root_path = "root_path" # Folder with ESC-50-master/
dataset = ESC50(root=root_path, download=False) #If download=False code assumes base_folder='ESC-50-master' in esc50_dataset.py
prompt = 'this is the sound of '
y = [prompt + x for x in dataset.classes]

# Load and initialize CLAP
clap_model = CLAP(version = '2023', use_cuda=True)

# Computing text embeddings
text_embeddings = clap_model.get_text_embeddings(y)

# Computing audio embeddings
y_preds, y_labels = [], []
for i in tqdm(range(len(dataset))):
    x, _, one_hot_target = dataset.__getitem__(i)
    print(x) #root_path/ESC-50-master/audio/1-23996-A-35.wav
    audio_embeddings = clap_model.get_audio_embeddings([x], resample=True)
    similarity = clap_model.compute_similarity(audio_embeddings, text_embeddings)
    y_pred = F.softmax(similarity.detach().cpu(), dim=1).numpy()
    y_preds.append(y_pred)
    # print('y_preds', y_pred.shape) #(1, 50)
    # print('one_hot_target', one_hot_target.size()) #torch.Size([1, 50])
    y_labels.append(one_hot_target.detach().cpu().numpy())

y_labels, y_preds = np.concatenate(y_labels, axis=0), np.concatenate(y_preds, axis=0)
# print('y_labels:', y_labels.shape) #(2000, 50)
# print('y_preds:', y_preds.shape) #(2000, 50)
acc = accuracy_score(np.argmax(y_labels, axis=1), np.argmax(y_preds, axis=1))
print('ESC50 Accuracy {}'.format(acc))

"""
The output:

ESC50 Accuracy: 93.9%

"""
