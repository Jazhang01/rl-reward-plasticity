from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import wandb

from jaxrl_m.evaluation import add_to, flatten


def visual_evaluate(policy_fn, env, num_episodes):
    stats = defaultdict(list)
    trajectories = []
    frames = []

    max_z_coord = 0

    for i in range(num_episodes):
        xy_positions = []
        (observation, _), done = env.reset(), False
        while not done:
            max_z_coord = max(max_z_coord, observation[2])

            # AntMaze and PointMaze will have (x, y) positions in the first two coordinates
            xy_positions.append(observation[:2])  
            if i == 0:
                frame = np.transpose(env.render(), axes=[2, 0, 1])
                frames.append(frame)
    
            action = policy_fn(observation)
            observation, _, term, trunc, info = env.step(action)
            done = term or trunc
            add_to(stats, flatten(info))
        add_to(stats, flatten(info, parent_key="final"))
        trajectories.append(xy_positions)

    for k, v in stats.items():
        stats[k] = np.mean(v)
    stats['max_z_coord'] = max_z_coord
    
    visuals = {
        'eval_video': wandb.Video(np.array(frames), fps=4),
        **xy_trajectory_visualization(trajectories)
    }

    eval_info = {**stats, **visuals}
    return eval_info


def xy_trajectory_visualization(trajectories):
    fig, axes = plt.subplots(nrows=1, ncols=1)
    for traj in trajectories:
        xs, ys = [], []
        for xy_pos in traj:
            x, y = xy_pos
            xs.append(x)
            ys.append(y)
        
        alpha = [i / len(traj) for i in range(len(traj))]
        axes.scatter(xs, ys, alpha=alpha)
    
    image = wandb.Image(fig)
    plt.close()

    return {'trajectories': image}
