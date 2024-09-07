# source
# microphone
# fiber_mandrel_summed_resample
# fiber_mandrel_1_resampled
# fiber_mandrel_2_resampled
# fiber_mandrel_3_resampled
# fiber_mandrel_4_resampled
# fiber_mandrel_5_resampled
# fiber_mandrel_6_resampled
# fiber_mandrel_7_resampled
# fiber_coil_resampled

# checkpoint_path:
# source_fold4_best_acc.pth
# microphone_fold4_best_acc.pth
# fiber_mandrel_summed_resample_fold4_best_acc.pth
# fiber_mandrel_1_resampled_fold4_best_acc.pth
# fiber_mandrel_2_resampled_fold4_best_acc.pth
# fiber_mandrel_3_resampled_fold4_best_acc.pth
# fiber_mandrel_4_resampled_fold4_best_acc.pth
# fiber_mandrel_5_resampled_fold4_best_acc.pth
# fiber_mandrel_6_resampled_fold4_best_acc.pth
# fiber_mandrel_7_resampled_fold4_best_acc.pth
# fiber_coil_resampled_fold4_best_acc.pth


#!/bin/bash
audio_dirs=(
    "source"
    "microphone"
    "fiber_mandrel_summed_resample"
    "fiber_mandrel_1_resampled"
    "fiber_mandrel_2_resampled"
    "fiber_mandrel_3_resampled"
    "fiber_mandrel_4_resampled"
    "fiber_mandrel_5_resampled"
    "fiber_mandrel_6_resampled"
    "fiber_mandrel_7_resampled"
    "fiber_coil_resampled"
)

checkpoint_paths=(
best_acc.pth
)

for audio_dir in "${audio_dirs[@]}"; do
    for checkpoint_path in "${checkpoint_paths[@]}"; do
        python train-val.py \
            --root_path /home/onsi/jsun/dataset/ \
            --test_file "$audio_dir" \
            --model_version 2023 \
            --use_cuda True \
            --download_dataset False \
            --epochs 50 \
            --save_path 'checkpoin/' \
            --checkpoint_path "checkpoin-multitask/${checkpoint_path}" \
            --eval True
    done
done