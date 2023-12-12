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
        --max_steps=500000 \
        --save_interval=50000 \
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

# debug
# run_antmaze "antmaze-medium-diverse-v2" 0 "None" True True
# run_pointmaze "PointMaze_Large-v3" 0 "None" True True

# antmaze_checkpoint="/home/jason/Desktop/coursework/cs285/rl-reward-plasticity/checkpoints/rl-reward-plasticity/sac_test/sac_test_sac_antmaze-medium-diverse-v2_20231211_215613/checkpoint_eval_300000"
# run_antmaze "antmaze-medium-diverse-test-v2" 0 $antmaze_checkpoint False False
# run_antmaze "antmaze-medium-diverse-test-v2" 0 None False False

pointmaze_checkpoint="/home/jason/Desktop/coursework/cs285/rl-reward-plasticity/checkpoints/rl-reward-plasticity/sac_test/sac_test_sac_PointMaze_Large-v3_20231211_175240/checkpoint_explore_500000"
run_pointmaze "PointMaze_LargeDense-v3" 0 $pointmaze_checkpoint False False


# initial experiments
# for seed in 1 2 3 4; do
#     run_antmaze "antmaze-medium-diverse-v2" $seed "None" True True
#     run_pointmaze "PointMaze_Large-v3" $seed "None" True True
# done


# generalize experiments
# for seed in 1 2; do
#     for checkpoint in LIST HERE; do
#         run_antmaze "antmaze-medium-diverse-test-v2" $seed $checkpoint False False
#         run_pointmaze "PointMaze_LargeDense-v3" $seed $checkpoint False False
#     done
# done
