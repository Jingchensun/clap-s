import os
import json
import re
import numpy as np
from collections import OrderedDict

# 定义主目录路径
log_root_dir = 'log-pure-support/'

# 定义正则表达式匹配不同的数据集文件
fiber_pattern = re.compile(r'fiber_gunshot_resampled_seed(\d)\.txt')
coil_pattern = re.compile(r'coil_gunshot_resampled_seed(\d)\.txt')
fiber_coil_pattern = re.compile(r'fiber_coil_resampled_seed(\d)\.txt')
fiber_mandrel_pattern = re.compile(r'fiber_mandrel_4_resampled_seed(\d)\.txt')
microphone_pattern = re.compile(r'microphone_seed(\d)\.txt')  # 修改正则表达式，匹配所有 seed

# 用于存储所有 shot 的结果
fiber_results = {}
coil_results = {}
fiber_coil_results = {}
fiber_mandrel_results = {}
microphone_results = {}

# 定义提取精度的函数，使用正则表达式提取小数
def extract_accuracy(line):
    match = re.search(r'(\d+\.\d+)', line)
    if match:
        return float(match.group(1))
    return None

# 遍历 log_root_dir 目录下的每个 shot 文件夹（如 1shot, 2shot, 4shot 等）
for shot_dir in os.listdir(log_root_dir):
    full_shot_path = os.path.join(log_root_dir, shot_dir)

    # 检查是否为有效的目录
    if os.path.isdir(full_shot_path) and shot_dir.endswith('shot'):
        shot_num = shot_dir.replace('shot', '')  # 提取 shot 数量

        # 将 shot 数量转换为整数（处理 -1 这种特殊情况）
        try:
            shot_num = int(shot_num)
        except ValueError:
            continue

        # 用于存储每个 shot 下的 seed 精度
        fiber_accuracies = {}
        coil_accuracies = {}
        fiber_coil_accuracies = {}
        fiber_mandrel_accuracies = {}
        microphone_accuracies = {}

        # 遍历该 shot 目录下的文件
        for filename in os.listdir(full_shot_path):
            filepath = os.path.join(full_shot_path, filename)

            # 检查文件是否是 fiber_gunshot_resampled_seed*.txt
            fiber_match = fiber_pattern.match(filename)
            if fiber_match:
                seed = fiber_match.group(1)
                with open(filepath, 'r') as file:
                    for line in file:
                        if 'Tip-Adapter test accuracy' in line:
                            accuracy = extract_accuracy(line)
                            if accuracy is not None:
                                fiber_accuracies[f'seed{seed}'] = round(accuracy, 3)
                            break

            # 检查文件是否是 coil_gunshot_resampled_seed*.txt
            coil_match = coil_pattern.match(filename)
            if coil_match:
                seed = coil_match.group(1)
                with open(filepath, 'r') as file:
                    for line in file:
                        if 'Tip-Adapter test accuracy' in line:
                            accuracy = extract_accuracy(line)
                            if accuracy is not None:
                                coil_accuracies[f'seed{seed}'] = round(accuracy, 3)
                            break

            # 检查文件是否是 fiber_coil_resampled_seed*.txt
            fiber_coil_match = fiber_coil_pattern.match(filename)
            if fiber_coil_match:
                seed = fiber_coil_match.group(1)
                with open(filepath, 'r') as file:
                    for line in file:
                        if 'Tip-Adapter test accuracy' in line:
                            accuracy = extract_accuracy(line)
                            if accuracy is not None:
                                fiber_coil_accuracies[f'seed{seed}'] = round(accuracy, 3)
                            break

            # 检查文件是否是 fiber_mandrel_4_resampled_seed*.txt
            fiber_mandrel_match = fiber_mandrel_pattern.match(filename)
            if fiber_mandrel_match:
                seed = fiber_mandrel_match.group(1)
                with open(filepath, 'r') as file:
                    for line in file:
                        if 'Tip-Adapter test accuracy' in line:
                            accuracy = extract_accuracy(line)
                            if accuracy is not None:
                                fiber_mandrel_accuracies[f'seed{seed}'] = round(accuracy, 3)
                            break

            # 检查文件是否是 microphone_seed*.txt
            microphone_match = microphone_pattern.match(filename)
            if microphone_match:
                seed = microphone_match.group(1)
                with open(filepath, 'r') as file:
                    for line in file:
                        if 'Tip-Adapter test accuracy' in line:
                            accuracy = extract_accuracy(line)
                            if accuracy is not None:
                                microphone_accuracies[f'seed{seed}'] = round(accuracy, 3)
                            break

        # 计算每个数据集的均值和方差并保存结果，保留两位小数
        if fiber_accuracies:
            fiber_mean_accuracy = round(np.mean(list(fiber_accuracies.values())), 3)
            fiber_std_accuracy = round(np.std(list(fiber_accuracies.values())), 3)
            fiber_results[shot_num] = {
                'accuracies': fiber_accuracies,
                'mean_accuracy': fiber_mean_accuracy,
                'std_accuracy': fiber_std_accuracy
            }

        if coil_accuracies:
            coil_mean_accuracy = round(np.mean(list(coil_accuracies.values())), 3)
            coil_std_accuracy = round(np.std(list(coil_accuracies.values())), 3)
            coil_results[shot_num] = {
                'accuracies': coil_accuracies,
                'mean_accuracy': coil_mean_accuracy,
                'std_accuracy': coil_std_accuracy
            }

        if fiber_coil_accuracies:
            fiber_coil_mean_accuracy = round(np.mean(list(fiber_coil_accuracies.values())), 3)
            fiber_coil_std_accuracy = round(np.std(list(fiber_coil_accuracies.values())), 3)
            fiber_coil_results[shot_num] = {
                'accuracies': fiber_coil_accuracies,
                'mean_accuracy': fiber_coil_mean_accuracy,
                'std_accuracy': fiber_coil_std_accuracy
            }

        if fiber_mandrel_accuracies:
            fiber_mandrel_mean_accuracy = round(np.mean(list(fiber_mandrel_accuracies.values())), 3)
            fiber_mandrel_std_accuracy = round(np.std(list(fiber_mandrel_accuracies.values())), 3)
            fiber_mandrel_results[shot_num] = {
                'accuracies': fiber_mandrel_accuracies,
                'mean_accuracy': fiber_mandrel_mean_accuracy,
                'std_accuracy': fiber_mandrel_std_accuracy
            }

        if microphone_accuracies:
            microphone_mean_accuracy = round(np.mean(list(microphone_accuracies.values())), 3)
            microphone_std_accuracy = round(np.std(list(microphone_accuracies.values())), 3)
            microphone_results[shot_num] = {
                'accuracies': microphone_accuracies,
                'mean_accuracy': microphone_mean_accuracy,
                'std_accuracy': microphone_std_accuracy
            }

# 排序 shot 数量为 1, 2, 4, 8, 16, -1 的顺序
def sort_shots(shot_dict):
    return OrderedDict(sorted(shot_dict.items(), key=lambda x: (x[0] == -1, x[0])))

# 对结果进行排序
fiber_results_sorted = sort_shots(fiber_results)
coil_results_sorted = sort_shots(coil_results)
fiber_coil_results_sorted = sort_shots(fiber_coil_results)
fiber_mandrel_results_sorted = sort_shots(fiber_mandrel_results)
microphone_results_sorted = sort_shots(microphone_results)

# 保存结果到 JSON 文件
result_dir = 'result'
os.makedirs(result_dir, exist_ok=True)

fiber_output_file = os.path.join(result_dir, 'fiber_acc.json')
with open(fiber_output_file, 'w') as json_file:
    json.dump(fiber_results_sorted, json_file, indent=4)

coil_output_file = os.path.join(result_dir, 'coil_acc.json')
with open(coil_output_file, 'w') as json_file:
    json.dump(coil_results_sorted, json_file, indent=4)

fiber_coil_output_file = os.path.join(result_dir, 'fiber_coil_acc.json')
with open(fiber_coil_output_file, 'w') as json_file:
    json.dump(fiber_coil_results_sorted, json_file, indent=4)

fiber_mandrel_output_file = os.path.join(result_dir, 'fiber_mandrel_acc.json')
with open(fiber_mandrel_output_file, 'w') as json_file:
    json.dump(fiber_mandrel_results_sorted, json_file, indent=4)

microphone_output_file = os.path.join(result_dir, 'microphone_acc.json')
with open(microphone_output_file, 'w') as json_file:
    json.dump(microphone_results_sorted, json_file, indent=4)

print(f"结果已保存到 {result_dir} 目录下的对应 JSON 文件中")
