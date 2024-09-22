import os
import pickle
import numpy as np
import torch
import torchaudio
from scipy.signal import butter, filtfilt

# 定义类别字典，将类别名称映射到数字标签
classes = {
    'Background': 0,
    'Starter Gun': 1,
    'Door Slam': 2,
    'Car Alarm': 3,
    'Crackers': 4,
    'Cannon': 5,
    'Fountain Cannon': 6,
    'High Altitude Firework': 7
}

# 定义类别索引到名称的字典
index_to_class = {v: k for k, v in classes.items()}

# 定义数据集文件名
fse_dataset = 'fiber_gunshot'
coil_dataset = 'coil_gunshot'

# fse_dataset = 'data_label_814_421'
# coil_dataset = 'data_label_coil_512_263'

# 设置原始采样率和目标采样率
original_fs = 2000  # 原始采样率
target_fs = 44100  # 目标采样率
duration = 1  # 持续时间（秒）
num_samples = original_fs * duration  # 计算样本数

# 指定数据路径和选择的数据集
data_path = '/home/jingchen/data/fiber-data/gunshot/'  # 修改为实际数据路径
dataset_selection = coil_dataset  # 或 fse_dataset，coil_dataset 根据需要选择

# 读取数据集文件
with open(data_path + dataset_selection + '.pickle', 'rb') as input_file:
    data_dict = pickle.load(input_file)  # 加载数据字典

# 定义高通滤波器参数
flag_hpf = False  # 是否使用高通滤波
cutoff = 200  # 截止频率
order = 5  # 滤波器阶数
nyq = 0.5 * original_fs  # 奈奎斯特频率

# 如果启用了高通滤波，则设计高通滤波器
if flag_hpf:
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)

# 应用高通滤波器的函数
def apply_high_pass_filter(audio_data):
    if flag_hpf:
        filtered_data = filtfilt(b, a, audio_data)
        return filtered_data
    return audio_data

# 处理训练数据
X_tr_lst = data_dict['train_x']  # 获取训练数据列表
X_tr_arr = np.array(X_tr_lst)  # 转换为NumPy数组
X_tr_hpf = np.array([apply_high_pass_filter(x) for x in X_tr_arr])  # 应用高通滤波器
X_tr = torch.FloatTensor(X_tr_hpf)  # 转换为PyTorch浮点张量

# 处理测试数据
X_te_lst = data_dict['test_x']  # 获取测试数据列表
X_te_arr = np.array(X_te_lst)  # 转换为NumPy数组
X_te_hpf = np.array([apply_high_pass_filter(x) for x in X_te_arr])  # 应用高通滤波器
X_te = torch.FloatTensor(X_te_hpf)  # 转换为PyTorch浮点张量

# 处理训练标签
y_tr = torch.tensor(data_dict['train_y'], dtype=torch.long)  # 转换为PyTorch长整型张量

# 处理测试标签
y_te = torch.tensor(data_dict['test_y'], dtype=torch.long)  # 转换为PyTorch长整型张量

# 创建保存目录
output_dir = os.path.join(data_path, dataset_selection + '-resampled')
os.makedirs(output_dir, exist_ok=True)

# 定义重采样器
resampler = torchaudio.transforms.Resample(orig_freq=original_fs, new_freq=target_fs)

# 保存音频文件的函数
def save_audio(audio_tensor, sample_rate, filename):
    # torchaudio 期望音频张量的形状为 (num_channels, num_samples)
    # 如果音频是单通道的，确保张量的形状是 (1, num_samples)
    if len(audio_tensor.shape) == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    torchaudio.save(filename, audio_tensor, sample_rate)

# 保存训练数据
for index, audio in enumerate(X_tr):
    resampled_audio = resampler(audio)  # 重采样
    filename = os.path.join(output_dir, f'train_{index}.wav')
    save_audio(resampled_audio, target_fs, filename)
    print(f'Saved {filename}')

# 保存测试数据
for index, audio in enumerate(X_te):
    resampled_audio = resampler(audio)  # 重采样
    filename = os.path.join(output_dir, f'test_{index}.wav')
    save_audio(resampled_audio, target_fs, filename)
    print(f'Saved {filename}')

# 输出数据集的维度信息
print(f"Training data shape: {X_tr.shape}")
print(f"Training labels shape: {y_tr.shape}")
print(f"Testing data shape: {X_te.shape}")
print(f"Testing labels shape: {y_te.shape}")
