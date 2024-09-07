# import pandas as pd
# import matplotlib.pyplot as plt

# # 创建数据表
# data = {
#     "Shot": ["zero-shot", "1shot", "2-shot", "4-shot", "8-shot", "16-shot", "Full-shot"],
#     "Source": [93, 94, 94, 94, 96, 98, 99],
#     "Microphone": [76, 81, 86, 90, 93, 95, 98],
#     "Fiber mandrel-all": [44.4, 51.1, 55.6, 60.4, 69.9, 77.0, 87.3],
#     "Fiber Coil": [23, 29, 34, 43, 49, 57, 73],
#     "Average": [50.5, 63.8, 67.4, 71.9, 77.0, 81.8, 89.3]
# }

# df = pd.DataFrame(data)

# # 设置全局字体大小
# plt.rcParams.update({'font.size': 20})

# # 绘制折线图
# plt.figure(figsize=(12, 8))
# for column in df.columns[1:]:
#     plt.plot(df["Shot"], df[column], marker='o', linestyle='-', label=column)

# # 设置图表标签和标题
# plt.xlabel('Shot Number')
# plt.ylabel('Accuracy')
# plt.title('SupportSet Based Few-shot Experiment')
# plt.legend()
# plt.grid(True)
# plt.savefig('plot.png')
# plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# Create the dataframe from the provided data
data = {
    'Model': ['1shot', '1shot', '2-shot', '2-shot', '4-shot', '4-shot', '8-shot', '8-shot', '16-shot', '16-shot', 'Full-shot', 'Full-shot'],
    'Type': ['Adapter', 'Support Set-Trained', 'Adapter', 'Support Set-Trained', 'Adapter', 'Support Set-Trained', 'Adapter', 'Support Set-Trained', 'Adapter', 'Support Set-Trained', 'Adapter', 'Support Set-Trained'],
    'Source Microphone': [91.5, 95, 94.5, 96, 96.5, 96, 98.5, 98, 99.5, 99, 99, 98],
    'Fiber mandrel-all': [79.5, 83, 84, 86, 92, 93, 94, 94, 96, 97, 99, 99],
    'Fiber Coil': [43.2, 53, 54.1, 56.1, 64.9, 66.9, 75.4, 76, 83.1, 82.9, 91.5, 91.3],
    'Fiber-Gunshot': [60.3, 65.3, 66.4, 68, 74.5, 75.5, 81.0, 81.3, 86.3, 86.5, 91.5, 92.3],
    'Coil-Gunshot': [55.58, 56, 62.95, 65, 73.63, 72, 75.06, 75, 75.77, 75, 84.8, 85]
}

df = pd.DataFrame(data)

# Plotting function
def plot_line_chart(df, column_name, ax):
    shots = df['Model'].unique()
    adapter_values = df[df['Type'] == 'Adapter'][column_name].values
    support_set_values = df[df['Type'] == 'Support Set-Trained'][column_name].values
    
    ax.plot(shots, adapter_values, marker='o', label='Adapter')
    ax.plot(shots, support_set_values, marker='o', label='Support Set-Trained')
    ax.set_title(column_name)
    ax.set_xlabel('Shots')
    ax.set_ylabel('Performance')
    ax.legend()

# Create subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Plot each dataset
plot_line_chart(df, 'Source Microphone', axes[0, 0])
plot_line_chart(df, 'Fiber mandrel-all', axes[0, 1])
plot_line_chart(df, 'Fiber Coil', axes[0, 2])
plot_line_chart(df, 'Fiber-Gunshot', axes[1, 0])
plot_line_chart(df, 'Coil-Gunshot', axes[1, 1])

# Hide the empty subplot
axes[1, 2].axis('off')

plt.tight_layout()
plt.show()
plt.savefig('plot2.png')
