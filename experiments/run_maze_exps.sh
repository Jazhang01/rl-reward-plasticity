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
        --batch_size=128
}
export -f run_maze

# run_maze "PointMaze_Large-v3" 1 "identity" "None" "rnd"
run_maze "AntMaze_Medium-v4" 1 "identity" "None" "rnd"
