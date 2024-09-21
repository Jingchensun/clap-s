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