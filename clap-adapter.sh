#############Train Model Script###############
# fiber_mandrel_summed_resample
# microphone
# fiber_mandrel_4_resampled
# fiber_coil_resampled
# ##
# fiber_gunshot_resampled
# coil_gunshot_resampled
            #--root_path /home/jingchen/data/fiber-data/gunsho-only-resample/ \
            #  /home/jingchen/data/fiber-data/lab/
audio_dirs="
fiber_gunshot_resampled
coil_gunshot_resampled
"

for audio_dir in $audio_dirs
do
    for seed in 1 2 3
    do
        python clap-adpter.py \
            --root_path /home/jingchen/data/fiber-data/gunsho-only-resample/ \
            --audio_dataset "$audio_dir" \
            --model_version 2023 \
            --use_cuda True \
            --download_dataset False \
            --epochs 20 \
            --save_path 'check-adapter/' \
            --seed "$seed" \
            --shot '24'
            # --checkpoint_path '/home/onsi/jsun/clap/2-check-gunshot-all/check-gounshot-1shot/fiber_gunshot-resampled_best_acc.pth' \
            # --eval True
    done
done


#############Test Model Script###############

# audio_dirs=(
#     "source"
#     "microphone"
#     "fiber_mandrel_summed_resample"
#     "fiber_mandrel_1_resampled"
#     "fiber_mandrel_2_resampled"
#     "fiber_mandrel_3_resampled"
#     "fiber_mandrel_4_resampled"
#     "fiber_mandrel_5_resampled"
#     "fiber_mandrel_6_resampled"
#     "fiber_mandrel_7_resampled"
#     "fiber_coil_resampled"
# )

# checkpoint_paths=(
# best_acc.pth
# )

# for audio_dir in "${audio_dirs[@]}"; do
#     for checkpoint_path in "${checkpoint_paths[@]}"; do
#         python train-val.py \
#             --root_path /home/onsi/jsun/dataset/ \
#             --test_file "$audio_dir" \
#             --model_version 2023 \
#             --use_cuda True \
#             --download_dataset False \
#             --epochs 50 \
#             --save_path 'checkpoin/' \
#             --checkpoint_path "checkpoin-multitask/${checkpoint_path}" \
#             --eval True
#     done
# done