audio_dirs="
fiber_gunshot_resampled
coil_gunshot_resampled
"

for audio_dir in $audio_dirs
do
    for seed in 1 
    do
        CUDA_VISIBLE_DEVICES=1 python clap_adapter.py \
            --root_path /home/jingchen/data/fiber-data/gunshot-only-resample/ \
            --dataset Fiber-firework \
            --audio_dataset "$audio_dir" \
            --model_version 2023 \
            --use_cuda True \
            --download_dataset False \
            --epochs 20 \
            --save_path 'check-adapter/' \
            --seed "$seed" \
            --shot '1'
    done
done