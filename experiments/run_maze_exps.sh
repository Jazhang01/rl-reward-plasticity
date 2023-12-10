run_maze() {
    python experiments/run_plasticity_maze_sac.py \
        --env_name=$1 \
        --seed=$2 \
        --reward_wrapper=$3 \
        --load_dir=$4 \
        --reward_bonus=$5 \
        --save_dir="checkpoints" \
        --max_steps=500000 \
        --save_interval=50000 \
        --buffer_size=500000 \
        --random_buffer_size=100000 \
        --start_steps=5000 \
        --batch_size=256 \
        --utd_ratio=5 \
        --config.hidden_dims="(256,256,256)" \
        --config.backup_entropy=False \
        --config.num_qs=10 \
        --config.num_min_qs=1 \
        --config.use_layer_norm=True 
}
export -f run_maze

# run_maze "PointMaze_Large-v3" 1 "identity" "None" "rnd"
run_maze "AntMaze_Medium-v4" 1 "identity" "None" "rnd"
