audio_dirs='
fiber_gunshot_resampled
coil_gunshot_resampled
'

for audio_dir in $audio_dirs
do
    for seed in 1
    do
        CUDA_VISIBLE_DEVICES=1 python clap_support_plus.py \
            --root_path /home/jingchen/data/fiber-data/gunshot-only-resample \
            --dataset Fiber-firework \
            --audio_dataset "$audio_dir" \
            --model_version 2023 \
            --use_cuda True \
            --seed "$seed" \
            --shot '1' \
            --save_path 'check-s' 
    done
done