run_pointmaze() {
    # DQN
    python experiments/run_plasticity_maze_sac.py \
        --env_name=$1 \
        --seed=$2 \
        --load_dir=$3 \
        --use_explore_agent=$4 \
        --use_noise_agent=$5 \
        --reward_wrapper="identity" \
        --reward_bonus="rnd" \
        --save_dir="checkpoints" \
        --use_eval_agent=True \
        --max_steps=500000 \ --save_interval=50000 \
        --buffer_size=500000 \
        --start_steps=10000 \
        --batch_size=1024 \
        --utd_ratio=1 \
        --dqn_config.hidden_dims="(256,256)" \
        --dqn_config.use_layer_norm=False \
        --rnd_config.hidden_dims="(256,256)" \
        --rnd_config.latent_dim=256 \
        --rnd_config.rnd_coeff=0.1 \
        --rnd_config.normalize_rewards=False \
        --rnd_config.use_actions=False
}
export -f run_pointmaze

run_antmaze() {
    # SAC / RED-Q
    python experiments/run_plasticity_maze_sac.py \
        --env_name=$1 \
        --seed=$2 \
        --load_dir=$3 \
        --use_explore_agent=$4 \
        --use_noise_agent=$5 \
        --reward_wrapper="identity" \
        --reward_bonus="rnd" \
        --save_dir="checkpoints" \
        --use_eval_agent=True \
        --max_steps=500000 \
        --save_interval=50000 \
        --buffer_size=500000 \
        --start_steps=5000 \
        --batch_size=256 \
        --utd_ratio=10 \
        --sac_config.hidden_dims="(256,256,256)" \
        --sac_config.backup_entropy=False \
        --sac_config.num_qs=10 \
        --sac_config.num_min_qs=1 \
        --sac_config.use_layer_norm=True \
        --rnd_config.hidden_dims="(256,256,256)" \
        --rnd_config.rnd_coeff=1
}
export -f run_antmaze

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

export XLA_PYTHON_CLIENT_MEM_FRACTION=.45

# maze auxiliary tasks
# parallel --ungroup -j2 --delay 5s run_antmaze \
#     ::: "antmaze-medium-diverse-test-v2" \
#     ::: 1 2 \
#     ::: None \
#     ::: False \
#     ::: False
parallel --ungroup -j1 --delay 5s run_pointmaze \
    ::: "PointMaze_LargeDense-v3" \
    ::: 1 2 \
    ::: None \
    ::: False \
    ::: False

# mujoco with gaussian noise
parallel --ungroup -j2 --delay 5s run_mujoco \
    ::: "Hopper-v4" "HalfCheetah-v4" "Humanoid-v4" \
    ::: 1 \
    ::: "unit_noise" \
    ::: "None" \
    ::: False
