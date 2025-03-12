audio_dirs='
underwater_data
'

for audio_dir in $audio_dirs
do
    for seed in 1 2 3
    do
        CUDA_VISIBLE_DEVICES=1 python clap_support.py \
            --root_path data \
            --dataset Fiber \
            --audio_dataset "$audio_dir" \
            --model_version 2023 \
            --use_cuda True \
            --seed "$seed" \
            --shot '-1' \
            --save_path 'check-s' 
    done
done