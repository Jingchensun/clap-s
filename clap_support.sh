audio_dirs='
fiber_gunshot
coil_gunshot
'

for audio_dir in $audio_dirs
do
    for seed in 1
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