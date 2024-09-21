# microphone
# fiber_mandrel_4_resampled
# fiber_coil_resampled
# fiber_gunshot_resampled
# coil_gunshot_resampled
#/home/jingchen/data/fiber-data/gunsho-only-resample 
#/home/jingchen/data/fiber-data/lab/
audio_dirs='
microphone
fiber_mandrel_4_resampled
fiber_coil_resampled
'

for audio_dir in $audio_dirs
do
    for seed in 1 2 3 4 5
    do
        python clap-support.py \
            --root_path /home/jingchen/data/fiber-data/lab/ \
            --audio_dataset "$audio_dir" \
            --model_version 2023 \
            --use_cuda True \
            --seed "$seed" \
            --shot '2' \
            --save_path 'check-support-treff/' \
            # --checkpoint_path '/home/onsi/jsun/clap/2-check-gunshot-all/check-gounshot-1shot/fiber_gunshot-resampled_best_acc.pth'\
            # --eval True
    done
done