audio_dirs='audio'

for audio_dir in $audio_dirs
do
    for seed in 1
    do
        python clap_support.py \
            --root_path /home/jingchen/data/clap-audio/ESC-50-master \
            --audio_dataset "$audio_dir" \
            --model_version 2023 \
            --use_cuda True \
            --seed "$seed" \
            --shot '2' \
            --save_path 'check-s/' \
            # --checkpoint_path '/home/jingchen/clap/2-check-gunshot-all/check-gounshot-1shot/fiber_gunshot-resampled_best_acc.pth'\
            # --eval True
    done
done