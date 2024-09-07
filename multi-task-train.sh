# 定义数据集目录
audio_dirs="
source
microphone
fiber_mandrel_summed_resample
fiber_mandrel_1_resampled
fiber_mandrel_2_resampled
fiber_mandrel_3_resampled
fiber_mandrel_4_resampled
fiber_mandrel_5_resampled
fiber_mandrel_6_resampled
fiber_mandrel_7_resampled
fiber_coil_resampled
"

# 转换为空格分隔的字符串列表
audio_dirs_list=$(echo $audio_dirs | tr '\n' ' ')

python multi-task-train-val.py \
    --root_path /home/onsi/jsun/dataset/ \
    --test_files $audio_dirs_list \
    --model_version 2023 \
    --use_cuda True \
    --download_dataset False \
    --epochs 50 \
    --save_path 'checkpoint-multi-task-old/best_model.pth' \
    # --checkpoint_path '/home/onsi/jsun/clap/checkpoint-multi-task-seed6/best_acc.pth'\
    # --eval True
