audio_dirs="audio"

for audio_dir in $audio_dirs
do
    for seed in 1 
    do
        python clap_adapter.py \
            --root_path /home/jingchen/data/clap-audio/ESC-50-master \
            --audio_dataset "$audio_dir" \
            --model_version 2023 \
            --use_cuda True \
            --download_dataset False \
            --epochs 20 \
            --save_path 'check-adapter/' \
            --seed "$seed" \
            --shot '1'
            # --checkpoint_path '/home/jingchen/clap-s/check-adapter' \
            # --eval True
    done
done