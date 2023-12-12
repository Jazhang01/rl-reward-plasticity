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

run_mujoco_invert() {
    python experiments/run_plasticity_mujoco_sac.py \
        --env_name=$1 \
        --seed=$2 \
        --reward_wrapper=$3 \
        --load_dir=$4 \
        --rand_permutation_freq=100000 \
        --invert_env=True \
        --max_steps=500000 \
        --buffer_size=500000 \
        --random_buffer_size=100000 \
        --batch_size=128
}
export -f run_mujoco_invert

# preallocate a smaller amount of memory to allow for parallel jobs to run well
export XLA_PYTHON_CLIENT_MEM_FRACTION=.45

# Test plasticity on 6 environments (3 normal, 3 inverted), 5 seeds, with the default reward function.
# parallel --ungroup -j2 --delay 5s run_mujoco ::: "Hopper-v4" "HalfCheetah-v4" "Humanoid-v4" ::: 1 ::: "identity" ::: "None" ::: False True
# run_mujoco "Hopper-v4" 1 "identity" "None" False
# run_mujoco_invert "Hopper-v4" 1 "identity" "checkpoints/rl-reward-plasticity/sac_test/sac_test_sac_Hopper-v4_20231209_200602/checkpoint_100000"

# run baseline invert tests
parallel --ungroup -j2 --delay 5s run_mujoco_invert \
    ::: "Hopper-v4" "HalfCheetah-v4" "Humanoid-v4" \
    ::: 1 \
    ::: "identity" \
    ::: "None"

# Hopper-v4 performance checks
parallel --ungroup -j2 --delay 5s run_mujoco_invert \
    ::: "Hopper-v4" \
    ::: 1 \
    ::: "identity" \
    ::: \
    "checkpoints/rl-reward-plasticity/sac_test/sac_test_sac_Hopper-v4_20231209_200602/checkpoint_100000" \
    "checkpoints/rl-reward-plasticity/sac_test/sac_test_sac_Hopper-v4_20231209_200602/checkpoint_200000" \
    "checkpoints/rl-reward-plasticity/sac_test/sac_test_sac_Hopper-v4_20231209_200602/checkpoint_300000" \
    "checkpoints/rl-reward-plasticity/sac_test/sac_test_sac_Hopper-v4_20231209_200602/checkpoint_400000" \
    "checkpoints/rl-reward-plasticity/sac_test/sac_test_sac_Hopper-v4_20231209_200602/checkpoint_500000"

# HalfCheetah-v4 performance checks
parallel --ungroup -j2 --delay 5s run_mujoco_invert \
    ::: "HalfCheetah-v4" \
    ::: 1 \
    ::: "identity" \
    ::: \
    "checkpoints/rl-reward-plasticity/sac_test/sac_test_sac_HalfCheetah-v4_20231209_203929/checkpoint_100000" \
    "checkpoints/rl-reward-plasticity/sac_test/sac_test_sac_HalfCheetah-v4_20231209_203929/checkpoint_200000" \
    "checkpoints/rl-reward-plasticity/sac_test/sac_test_sac_HalfCheetah-v4_20231209_203929/checkpoint_300000" \
    "checkpoints/rl-reward-plasticity/sac_test/sac_test_sac_HalfCheetah-v4_20231209_203929/checkpoint_400000" \
    "checkpoints/rl-reward-plasticity/sac_test/sac_test_sac_HalfCheetah-v4_20231209_203929/checkpoint_500000"

# Humanoid-v4 performance checks
parallel --ungroup -j2 --delay 5s run_mujoco_invert \
    ::: "Humanoid-v4" \
    ::: 1 \
    ::: "identity" \
    ::: \
    "checkpoints/rl-reward-plasticity/sac_test/sac_test_sac_Humanoid-v4_20231209_211215/checkpoint_100000" \
    "checkpoints/rl-reward-plasticity/sac_test/sac_test_sac_Humanoid-v4_20231209_211215/checkpoint_200000" \
    "checkpoints/rl-reward-plasticity/sac_test/sac_test_sac_Humanoid-v4_20231209_211215/checkpoint_300000" \
    "checkpoints/rl-reward-plasticity/sac_test/sac_test_sac_Humanoid-v4_20231209_211215/checkpoint_400000" \
    "checkpoints/rl-reward-plasticity/sac_test/sac_test_sac_Humanoid-v4_20231209_211215/checkpoint_500000"

# test plasticity on some simple environments
# parallel -j1 --delay 5s run_mujoco ::: "Hopper-v4" "HalfCheetah-v4" "Humanoid-v4" ::: 1 ::: "identity" ::: "None" ::: "False" "True"

# parallel -j1 --delay 5s python experiments/run_plasticity_mujoco_sac.py --env_name=Hopper-v4 --seed={1} --max_steps=500000 --batch_size=128 --buffer_size=500000 --random_buffer_size=100000 ::: 1
# parallel -j1 --delay 5s python experiments/run_plasticity_mujoco_sac.py --env_name=Humanoid-v4 --seed={1} --max_steps=1000000 --batch_size=128 --buffer_size=1000000 --random_buffer_size=100000 ::: 1
# parallel -j1 --delay 5s python experiments/run_plasticity_mujoco_sac.py --env_name=HalfCheetah-v4 --seed={1} --max_steps=500000 --batch_size=128 --buffer_size=500000 --random_buffer_size=100000 ::: 1

# test plasticity when reward is altered

# parallel -j1 --delay 5s python experiments/run_plasticity_mujoco_sac.py --env_name=HalfCheetah-v4 --seed={1} --max_steps=500000 --batch_size=128 --buffer_size=500000 --random_buffer_size=100000 --reward_wrapper={2} ::: 1 ::: "zero" "one" "unit_noise"
# parallel -j1 --delay 5s python experiments/run_plasticity_mujoco_sac.py --env_name=Hopper-v4 --seed={1} --max_steps=500000 --batch_size=128 --buffer_size=500000 --random_buffer_size=100000 --reward_wrapper={2} ::: 1 ::: "zero" "one" "unit_noise"
# parallel -j1 --delay 5s python experiments/run_plasticity_mujoco_sac.py --env_name=Humanoid-v4 --seed={1} --max_steps=1000000 --batch_size=128 --buffer_size=1000000 --random_buffer_size=100000 --reward_wrapper={2} ::: 1 ::: "zero" "one" "unit_noise"
