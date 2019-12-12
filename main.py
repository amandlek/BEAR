import gym
import numpy as np
import torch
import argparse
import os
import time
import datetime
import h5py

import utils
import DDPG
import algos
import TD3
from logger import logger, setup_logger
from logger import create_stats_ordered_dict
import point_mass

import mujoco_py

FILE_PATH = os.path.dirname(os.path.abspath(__file__))

# Runs policy for X episodes and returns average reward
def evaluate_policy(policy, eval_episodes=10, horizon=1000):
    avg_reward = 0.
    avg_success = 0.
    all_rewards = []
    success_rate = 0.
    for _ in range(eval_episodes):
        obs = env.reset()
        env.set_robot_joint_positions([0, -1.18, 0.00, 2.18, 0.00, 0.57, 1.5708])
        obs = env._get_observation()
        done = False
        success = False
        cntr = 0
        try:
            # while ((not done)):
            for _ in range(horizon):
                obs = np.concatenate([obs["robot-state"], obs["object-state"]])
                action = policy.select_action(np.array(obs))
                obs, reward, done, _ = env.step(action)
                avg_reward += reward
                success = success or env._check_success()
                cntr += 1
        except mujoco_py.builder.MujocoException as e:
            print("WARNING: got exception {}".format(e))
            avg_reward = 0.
            success = False
        all_rewards.append(avg_reward)
        success_rate += float(success)

    avg_reward /= eval_episodes
    success_rate /= eval_episodes
    for j in range(eval_episodes-1, 1, -1):
        all_rewards[j] = all_rewards[j] - all_rewards[j-1]

    all_rewards = np.array(all_rewards)
    std_rewards = np.std(all_rewards)
    median_reward = np.median(all_rewards)
    print ("---------------------------------------")
    print ("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
    print ("Success Rate over %d episodes: %f" % (eval_episodes, success_rate))
    print ("---------------------------------------")
    return avg_reward, std_rewards, median_reward, success_rate

def evaluate_policy_discounted(policy, eval_episodes=10, horizon=1000):
    avg_reward = 0.
    avg_success = 0.
    all_rewards = []
    success_rate = 0.
    gamma = 0.99
    for _ in range(eval_episodes):
        obs = env.reset()
        env.set_robot_joint_positions([0, -1.18, 0.00, 2.18, 0.00, 0.57, 1.5708])
        obs = env._get_observation()
        done = False
        success = False
        cntr = 0
        gamma_t = 1
        try:
            # while ((not done)):
            for _ in range(horizon):
                obs = np.concatenate([obs["robot-state"], obs["object-state"]])
                action = policy.select_action(np.array(obs))
                obs, reward, done, _ = env.step(action)
                avg_reward += (gamma_t * reward)
                success = success or env._check_success()
                gamma_t = gamma * gamma_t
                cntr += 1
        except mujoco_py.builder.MujocoException as e:
            print("WARNING: got exception {}".format(e))
            avg_reward = 0.
            success = False
        all_rewards.append(avg_reward)
        success_rate += float(success)
    avg_reward /= eval_episodes
    success_rate /= eval_episodes
    for j in range(eval_episodes-1, 1, -1):
        all_rewards[j] = all_rewards[j] - all_rewards[j-1]

    all_rewards = np.array(all_rewards)
    std_rewards = np.std(all_rewards)
    median_reward = np.median(all_rewards)
    print ("---------------------------------------")
    print ("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
    print ("Success Rate over %d episodes: %f" % (eval_episodes, success_rate))
    print ("---------------------------------------")
    return avg_reward, std_rewards, median_reward, success_rate

def make_env(env_name):
    import robosuite
    env = robosuite.make(
        env_name,
        has_renderer=False,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=False,
        gripper_visualization=False,
        reward_shaping=True,
        control_freq=100,
    )
    return env

def setup(name):
    # tensorboard logging
    from tensorboardX import SummaryWriter
    t_now = time.time()
    time_str = datetime.datetime.fromtimestamp(t_now).strftime('%Y%m%d%H%M%S')
    exp_dir = os.path.join(FILE_PATH, "./experiments/{}".format(name), time_str)
    os.makedirs(exp_dir)

    output_dir = os.path.join(FILE_PATH, "./trained_models/{}".format(name), time_str)
    os.makedirs(output_dir)

    return SummaryWriter(exp_dir), output_dir

def make_buffer(hdf5_path):
    """
    Add transition tuples from batch file to replay buffer.
    """
    f = h5py.File(hdf5_path, "r")  
    demos = list(f["data"].keys())
    total_transitions = f["data"].attrs["total"]
    print("Loading {} transitions from {}...".format(total_transitions, hdf5_path))
    env_name = f["data"].attrs["env"]

    rb = None

    for i in range(len(demos)):
        ep = demos[i]
        obs = f["data/{}/obs".format(ep)][()]
        actions = f["data/{}/actions".format(ep)][()]
        rewards = f["data/{}/rewards".format(ep)][()]
        next_obs = f["data/{}/next_obs".format(ep)][()]
        dones = f["data/{}/dones".format(ep)][()]

        ### important: this is action clipping! ###
        actions = np.clip(actions, -1., 1.)

        if rb is None:
            rb = utils.ReplayBuffer(state_dim=obs[0].shape[0], action_dim=actions[0].shape[0])

        zipped = zip(obs, actions, rewards, next_obs, dones)
        for item in zipped:
            ob, ac, rew, next_ob, done = item
            # Expects tuples of (state, next_state, action, reward, done)
            rb.add((ob, next_ob, ac, rew, done))
    f.close()

    return rb, env_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--env_name", default="Hopper-v2")                          # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)                                      # Sets Gym, PyTorch and Numpy seeds
    # parser.add_argument("--buffer_type", default="Robust")                          # Prepends name to filename.
    # parser.add_argument("--eval_freq", default=5e3, type=float)                     # How often (time steps) we evaluate
    parser.add_argument("--eval_freq", default=1000, type=float)                     # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=float)         # Max time steps to run environment for
    # parser.add_argument("--buffer_name", default=None, type=str)            # the path to the buffer file
    parser.add_argument("--version", default='0', type=str)                 # Basically whether to do min(Q), max(Q), mean(Q) over multiple Q networks for policy updates
    parser.add_argument("--lamda", default=0.5, type=float)                 # Unused parameter -- please ignore 
    parser.add_argument("--threshold", default=0.05, type=float)            # Unused parameter -- please ignore
    parser.add_argument('--use_bootstrap', default=False, type=bool)        # Whether to use bootstrapped ensembles or plain ensembles
    parser.add_argument('--algo_name', default="OursBCQ", type=str)         # Which algo to run (see the options below in the main function)
    parser.add_argument('--mode', default='hardcoded', type=str)            # Whether to do automatic lagrange dual descent or manually tune coefficient of the MMD loss (prefered "auto")
    parser.add_argument('--num_samples_match', default=10, type=int)        # number of samples to do matching in MMD
    parser.add_argument('--mmd_sigma', default=10.0, type=float)            # The bandwidth of the MMD kernel parameter
    parser.add_argument('--kernel_type', default='laplacian', type=str)     # kernel type for MMD ("laplacian" or "gaussian")
    parser.add_argument('--lagrange_thresh', default=10.0, type=float)      # What is the threshold for the lagrange multiplier
    parser.add_argument('--distance_type', default="MMD", type=str)         # Distance type ("KL" or "MMD")
    # parser.add_argument('--log_dir', default='./data_hopper/', type=str)    # Logging directory
    parser.add_argument('--use_ensemble_variance', default='True', type=str)       # Whether to use ensemble variance or not
    parser.add_argument('--use_behaviour_policy', default='False', type=str)       
    parser.add_argument('--cloning', default="False", type=str)
    parser.add_argument('--num_random', default=10, type=int)
    parser.add_argument('--margin_threshold', default=10, type=float)       # for DQfD baseline
    
    ### ADDED: some more arguments ###
    parser.add_argument("--batch", type=str)
    parser.add_argument("--name", type=str) # name of exp

    args = parser.parse_args()

    writer, output_dir = setup(args.name)

    # Use any random seed, and not the user provided seed
    # seed = np.random.randint(10, 1000)
    seed = args.seed
    algo_name = args.algo_name
    
    buffer_name = args.batch #args.buffer_name

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if not os.path.exists("./logs"):
        os.makedirs("./logs")

    # Load buffer

    # first load transitions into the buffer and then dump into a temporary file, then read
    replay_buffer, env_name = make_buffer(buffer_name)
    replay_buffer.add_bootstrap_to_buffer(bootstrap_dim=4)

    file_name = algo_name + "_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_0.1" % (env_name, str(seed), str(args.version), str(args.lamda), str(args.threshold), str(args.use_bootstrap), str(args.mode),\
         str(args.kernel_type), str(args.num_samples_match), str(args.mmd_sigma), str(args.lagrange_thresh), str(args.distance_type), str(args.use_behaviour_policy), str(args.num_random))
    print ("---------------------------------------")
    print ("Settings: " + file_name)
    print ("---------------------------------------")

    # replay_buffer = utils.ReplayBuffer()
    # if args.env_name == 'Multigoal-v0':
    #   replay_buffer.load_point_mass(buffer_name, bootstrap_dim=4, dist_cost_coeff=0.01)
    # else:
    #   replay_buffer.load(buffer_name, bootstrap_dim=4)

    # env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Make env
    env = make_env(env_name)

    obs = env.reset()
    obs = np.concatenate([obs["robot-state"], obs["object-state"]])

    state_dim = obs.shape[0] #env.observation_space.shape[0]
    action_dim = env.dof #env.action_space.shape[0] 
    max_action = 1. #float(env.action_space.high[0])
    print (state_dim, action_dim)
    print ('Max action: ', max_action)

    variant = dict(
        algorithm=algo_name,
        version=args.version,
        env_name=env_name,
        seed=seed,
        lamda=args.lamda,
        threshold=args.threshold,
        use_bootstrap=str(args.use_bootstrap),
        bootstrap_dim=4,
        delta_conf=0.1,
        mode=args.mode,
        kernel_type=args.kernel_type,
        num_samples_match=args.num_samples_match,
        mmd_sigma=args.mmd_sigma,
        lagrange_thresh=args.lagrange_thresh,
        distance_type=args.distance_type,
        use_ensemble_variance=args.use_ensemble_variance,
        use_data_policy=args.use_behaviour_policy,
        num_random=args.num_random,
        margin_threshold=args.margin_threshold,
    )
    log_dir = "./logs/"
    setup_logger(file_name, variant=variant, log_dir=log_dir + file_name)

    if algo_name == 'BCQ':
        policy = algos.BCQ(state_dim, action_dim, max_action)
    elif algo_name == 'TD3':
        policy = TD3.TD3(state_dim, action_dim, max_action)
    elif algo_name == 'BC':
        policy = algos.BCQ(state_dim, action_dim, max_action, cloning=True)
    elif algo_name == 'DQfD':
        policy = algos.DQfD(state_dim, action_dim, max_action, lambda_=args.lamda, margin_threshold=float(args.margin_threshold))
    elif algo_name == 'KLControl':
        policy = algos.KLControl(2, state_dim, action_dim, max_action)
    elif algo_name == 'BEAR':
        policy = algos.BEAR(2, state_dim, action_dim, max_action, delta_conf=0.1, use_bootstrap=False,
            version=args.version,
            lambda_=float(args.lamda),
            threshold=float(args.threshold),
            mode=args.mode,
            num_samples_match=args.num_samples_match,
            mmd_sigma=args.mmd_sigma,
            lagrange_thresh=args.lagrange_thresh,
            use_kl=(True if args.distance_type == "KL" else False),
            use_ensemble=(False if args.use_ensemble_variance == "False" else True),
            kernel_type=args.kernel_type)
    elif algo_name == 'BEAR_IS':
        policy = algos.BEAR_IS(2, state_dim, action_dim, max_action, delta_conf=0.1, use_bootstrap=False,
            version=args.version,
            lambda_=float(args.lamda),
            threshold=float(args.threshold),
            mode=args.mode,
            num_samples_match=args.num_samples_match,
            mmd_sigma=args.mmd_sigma,
            lagrange_thresh=args.lagrange_thresh,
            use_kl=(True if args.distance_type == "KL" else False),
            use_ensemble=(False if args.use_ensemble_variance == "False" else True),
            kernel_type=args.kernel_type)
    
    evaluations = []

    episode_num = 0
    done = True 

    training_iters = 0
    num_epochs = 0
    last_time_saved = -1.
    best_success_rate = -1.
    while training_iters < args.max_timesteps: 
        pol_vals = policy.train(replay_buffer, iterations=int(args.eval_freq))

        ret_eval, var_ret, median_ret, success_rate = evaluate_policy(policy)
        evaluations.append(ret_eval)
        np.save("./results/" + file_name, evaluations)

        training_iters += args.eval_freq
        print ("Training iterations: " + str(training_iters))
        logger.record_tabular('Training Epochs', int(training_iters // int(args.eval_freq)))
        logger.record_tabular('AverageReturn', ret_eval)
        logger.record_tabular('VarianceReturn', var_ret)
        logger.record_tabular('MedianReturn', median_ret)

        # record state on TB
        for x in logger._tabular:
            writer.add_scalar(x[0], float(x[1]), num_epochs)
        writer.file_writer.flush()
        num_epochs += 1

        logger.dump_tabular()

        # save model every hour or when last record is beat
        if time.time() - last_time_saved > 0: # 3600:
            params_to_save = policy.get_dict_to_save()
            path_to_save = os.path.join(output_dir, "model_epoch_{}.pth".format(num_epochs))
            torch.save(params_to_save, path_to_save)
            last_time_saved = time.time()

        if success_rate > best_success_rate:
            best_success_rate = success_rate
            params_to_save = policy.get_dict_to_save()
            path_to_save = os.path.join(output_dir, "model_epoch_{}_best_{}.pth".format(num_epochs, success_rate))
            torch.save(params_to_save, path_to_save)
