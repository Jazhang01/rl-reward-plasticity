# test plasticity on some simple environments
parallel -j1 --delay 5s python experiments/run_plasticity_mujoco_sac.py --env_name=HalfCheetah-v4 --seed={1} --max_steps=500000 --batch_size=128 --buffer_size=500000 --random_buffer_size=100000 ::: 1
parallel -j1 --delay 5s python experiments/run_plasticity_mujoco_sac.py --env_name=Hopper-v4 --seed={1} --max_steps=500000 --batch_size=128 --buffer_size=500000 --random_buffer_size=100000 ::: 1
parallel -j1 --delay 5s python experiments/run_plasticity_mujoco_sac.py --env_name=Humanoid-v4 --seed={1} --max_steps=1000000 --batch_size=128 --buffer_size=1000000 --random_buffer_size=100000 ::: 1

# test plasticity when reward is altered
parallel -j1 --delay 5s python experiments/run_plasticity_mujoco_sac.py --env_name=HalfCheetah-v4 --seed={1} --max_steps=500000 --batch_size=128 --buffer_size=500000 --random_buffer_size=100000 --reward_wrapper={2} ::: 1 ::: "one" "zero" "unit_noise"
parallel -j1 --delay 5s python experiments/run_plasticity_mujoco_sac.py --env_name=Hopper-v4 --seed={1} --max_steps=500000 --batch_size=128 --buffer_size=500000 --random_buffer_size=100000 --reward_wrapper={2} ::: 1 ::: "one" "zero" "unit_noise"
parallel -j1 --delay 5s python experiments/run_plasticity_mujoco_sac.py --env_name=Humanoid-v4 --seed={1} --max_steps=1000000 --batch_size=128 --buffer_size=1000000 --random_buffer_size=100000 --reward_wrapper={2} ::: 1 ::: "one" "zero" "unit_noise"
