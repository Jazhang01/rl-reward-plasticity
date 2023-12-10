run_mujoco() {
    python experiments/run_plasticity_mujoco_sac.py \
        --env_name=$1 \
        --seed=$2 \
        --reward_wrapper=$3 \
        --load_dir=$4 \
        --invert_env=$5 \
        --save_dir="checkpoints" \
        --max_steps=500000 \
        --buffer_size=500000 \
        --random_buffer_size=100000 \
        --batch_size=128
}
export -f run_mujoco

# Test plasticity on 6 environments (3 normal, 3 inverted), 5 seeds, with the default reward function.
parallel --ungroup -j1 --delay 5s run_mujoco ::: "Hopper-v4" "HalfCheetah-v4" "Humanoid-v4" ::: 1 ::: "identity" ::: "None" ::: False True
# run_mujoco "Hopper-v4" 1 "identity" "None" False

# test plasticity on some simple environments
# parallel -j1 --delay 5s run_mujoco ::: "Hopper-v4" "HalfCheetah-v4" "Humanoid-v4" ::: 1 ::: "identity" ::: "None" ::: "False" "True"

# parallel -j1 --delay 5s python experiments/run_plasticity_mujoco_sac.py --env_name=Hopper-v4 --seed={1} --max_steps=500000 --batch_size=128 --buffer_size=500000 --random_buffer_size=100000 ::: 1
# parallel -j1 --delay 5s python experiments/run_plasticity_mujoco_sac.py --env_name=Humanoid-v4 --seed={1} --max_steps=1000000 --batch_size=128 --buffer_size=1000000 --random_buffer_size=100000 ::: 1
# parallel -j1 --delay 5s python experiments/run_plasticity_mujoco_sac.py --env_name=HalfCheetah-v4 --seed={1} --max_steps=500000 --batch_size=128 --buffer_size=500000 --random_buffer_size=100000 ::: 1

# test plasticity when reward is altered

# parallel -j1 --delay 5s python experiments/run_plasticity_mujoco_sac.py --env_name=HalfCheetah-v4 --seed={1} --max_steps=500000 --batch_size=128 --buffer_size=500000 --random_buffer_size=100000 --reward_wrapper={2} ::: 1 ::: "zero" "one" "unit_noise"
# parallel -j1 --delay 5s python experiments/run_plasticity_mujoco_sac.py --env_name=Hopper-v4 --seed={1} --max_steps=500000 --batch_size=128 --buffer_size=500000 --random_buffer_size=100000 --reward_wrapper={2} ::: 1 ::: "zero" "one" "unit_noise"
# parallel -j1 --delay 5s python experiments/run_plasticity_mujoco_sac.py --env_name=Humanoid-v4 --seed={1} --max_steps=1000000 --batch_size=128 --buffer_size=1000000 --random_buffer_size=100000 --reward_wrapper={2} ::: 1 ::: "zero" "one" "unit_noise"
