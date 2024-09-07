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
# ##
# fiber_gunshot_resampled
# coil_gunshot_resampled
audio_dirs='
source
microphone
fiber_mandrel_summed_resample
fiber_mandrel_1_resampled
fiber_mandrel_2_resampled
fiber_mandrel_3_resampled
fiber_mandrel_4_resampled
fiber_mandrel_5_resampled
fiber_mandrel_6_resampled
fiber_mandrel_7_resampled
fiber_coil_resampled
'

for audio_dir in $audio_dirs
do
python train-val.py \
    --root_path /home/onsi/jsun/dataset/ \
    --audio_dataset "$audio_dir" \
    --model_version 2023 \
    --use_cuda True \
    --download_dataset False \
    --epochs 20 \
    --save_path 'check-gunshot-few-shot/' \
    # --checkpoint_path '/home/onsi/jsun/clap/2-check-gunshot-all/check-gounshot-1shot/fiber_gunshot-resampled_best_acc.pth'\
    # --eval True
done