run_plasticity() {
    python experiments/run_plasticity.py \
        --env_name=$1 \
        --checkpoints=$2 \
        --rand_qs_weight=$3 \
        --config.hidden_dims=$4
}
export -f run_plasticity

run_mujoco() {
    python experiments/run_plasticity_mujoco_sac.py \
        --env_name=$1 \
        --seed=$2 \
        --reward_wrapper=$3 \
        --load_dir=$4 \
        --invert_env=False \
        --save_dir="checkpoints" \
        --max_steps=500000 \
        --buffer_size=500000 \
        --random_buffer_size=100000 \
        --batch_size=128 \
        --config.hidden_dims="(64,64)"
}
export -f run_mujoco

run_antmaze_plasticity() {
    python experiments/run_plasticity.py \
        --env_name="antmaze-medium-diverse-v2" \
        --checkpoints=$1 \
        --checkpoint_keyword=$2 \
        --rand_qs_weight=10 \
        --redq_config.hidden_dims="(256,256,256)" \
        --redq_config.backup_entropy=False \
        --redq_config.num_qs=10 \
        --redq_config.num_min_qs=1 \
        --redq_config.use_layer_norm=True
}
export -f run_antmaze_plasticity

run_pointmaze_plasticity() {
    python experiments/run_plasticity.py \
        --env_name="PointMaze_Large-v3" \
        --checkpoints=$1 \
        --checkpoint_keyword=$2 \
        --rand_qs_weight=10 \
        --dqn_config.hidden_dims="(256,256)" \
        --dqn_config.use_layer_norm=$3
}
export -f run_pointmaze_plasticity

# network size 256
# run_plasticity "Hopper-v4" "checkpoints/rl-reward-plasticity/sac_test/sac_test_sac_Hopper-v4_20231209_200602" 10
# run_plasticity "HalfCheetah-v4" "checkpoints/rl-reward-plasticity/sac_test/sac_test_sac_HalfCheetah-v4_20231209_203929" 10
# run_plasticity "Humanoid-v4" "checkpoints/rl-reward-plasticity/sac_test/sac_test_sac_Humanoid-v4_20231209_211215" 10

# run_pointmaze_plasticity "checkpoints/rl-reward-plasticity/sac_test/sac_test_sac_PointMaze_Large-v3_20231212_071041" "eval" True
run_pointmaze_plasticity "checkpoints/rl-reward-plasticity/sac_test/sac_test_sac_PointMaze_Large-v3_20231212_071041" "explore" True
run_pointmaze_plasticity "checkpoints/rl-reward-plasticity/sac_test/sac_test_sac_PointMaze_Large-v3_20231212_071041" "noise" True

# run_antmaze_plasticity "checkpoints/rl-reward-plasticity/sac_test/sac_test_sac_antmaze-medium-diverse-v2_20231212_090117" "eval"
# run_antmaze_plasticity "checkpoints/rl-reward-plasticity/sac_test/sac_test_sac_antmaze-medium-diverse-v2_20231212_090117" "eval"
# run_antmaze_plasticity "checkpoints/rl-reward-plasticity/sac_test/sac_test_sac_antmaze-medium-diverse-v2_20231212_090117" "eval"

# network size 64
# run_plasticity "Hopper-v4" "checkpoints/rl-reward-plasticity/sac_test/sac_test_sac_Hopper-v4_20231210_084347" 10 "(64,64)"
# run_plasticity "HalfCheetah-v4" "checkpoints/rl-reward-plasticity/sac_test/sac_test_sac_HalfCheetah-v4_20231210_085554" 10 "(64,64)"
# run_plasticity "Humanoid-v4" "checkpoints/rl-reward-plasticity/sac_test/sac_test_sac_Humanoid-v4_20231210_090929" 10 "(64,64)"

# parallel --ungroup -j1 --delay 5s run_plasticity \
#     ::: \
#     "Humanoid-v4" \
#     ::: \
#     checkpoints/rl-reward-plasticity/sac_test/sac_test_sac_Humanoid-v4_20231209_211215 \
#     ::: \
#     checkpoints/rl-reward-plasticity/sac_test/sac_test_sac_Humanoid-v4_20231209_211215/buffer_500000.pkl \
#     ::: \
#     10 100 1000
# 
# parallel --ungroup -j1 --delay 5s run_plasticity \
#     ::: \
#     "Hopper-v4" \
#     ::: \
#     checkpoints/rl-reward-plasticity/sac_test/sac_test_sac_Hopper-v4_20231209_200602 \
#     ::: \
#     checkpoints/rl-reward-plasticity/sac_test/sac_test_sac_Hopper-v4_20231209_200602/buffer_500000.pkl \
#     ::: \
#     50 100 1000
# 
# parallel --ungroup -j1 --delay 5s run_mujoco \
#     ::: "Hopper-v4" "HalfCheetah-v4" "Humanoid-v4" \
#     ::: 1 \
#     ::: "identity" \
#     ::: "None"
