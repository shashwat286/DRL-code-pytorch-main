import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
import argparse
from normalization import Normalization, RewardScaling
from replaybuffer import ReplayBuffer
from ppo_discrete import PPO_discrete
from pathlib import Path
import pickle
import time
import matplotlib.pyplot as plt


def evaluate_policy(args, seed, env, agent, state_norm):
    times = 3
    evaluate_reward = 0
    for _ in range(times):
        s = env.reset(seed=seed)[0]
        if args.use_state_norm:  # During the evaluating,update=False
            s = state_norm(s, update=False)
        done = False
        episode_reward = 0
        while not done:
            a = agent.evaluate(s)  # We use the deterministic policy during the evaluating
            s_, r, term, trun, _ = env.step(a)
            if args.use_state_norm:
                s_ = state_norm(s_, update=False)
            episode_reward += r
            s = s_
        evaluate_reward += episode_reward

    return evaluate_reward / times


def main(args, env_name, number, seed):
    env = gym.make(env_name,render_mode="rgb_array")
    env_evaluate = gym.make(env_name)  # When evaluating the policy, we need to rebuild an environment
    # Set random seed
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env_evaluate.reset(seed=seed)
    env_evaluate.action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    args.state_dim = env.observation_space.shape[0]
    print("State_dim",args.state_dim)
    args.action_dim = env.action_space.n
    args.max_episode_steps = env._max_episode_steps  # Maximum number of steps per episode
    print("env={}".format(env_name))
    print("state_dim={}".format(args.state_dim))
    print("action_dim={}".format(args.action_dim))
    print("max_episode_steps={}".format(args.max_episode_steps))

    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating
    total_steps = 0  # Record the total steps during the training

    trajectory =[]

    replay_buffer = ReplayBuffer(args)
    agent = PPO_discrete(args)

    # Build a tensorboard
    writer = SummaryWriter(log_dir='runs/PPO_discrete/env_{}_number_{}_seed_{}'.format(env_name, number, seed))

    state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
    if args.use_reward_norm:  # Trick 3:reward normalization
        reward_norm = Normalization(shape=1)
    elif args.use_reward_scaling:  # Trick 4:reward scaling
        reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

    # relative_directory = Path('models')
    # file_actor_name = 'mountain_car_actor_PPO7761.pth'
    # file_critic_name = 'mountain_car_critic_PPO7761.pth'
    # actor_path = relative_directory / file_actor_name
    # critic_path = relative_directory / file_critic_name
    # agent.actor.load_state_dict(torch.load(actor_path,weights_only=True))
    # agent.critic.load_state_dict(torch.load(critic_path,weights_only=True))
    # agent.actor.eval()
    # agent.critic.eval()

    #     #Testing loop over episodes
    # for episode in range(1, 11):
    #     state, _ = env.reset(seed=seed)
    #         # print("test state",state.shape)
            
    #     done = False
    #     truncation = False
    #     step_size = 0
    #     episode_reward = 0

    #     while not done:
    #         with torch.no_grad():
    #             action, a_logprob = agent.choose_action(state)
    #             #print("test action",action)
    #         next_state, reward, done, truncation, _ = env.step(action)
    #             # print("test next state",next_state)
    #             # print("test reward",reward)
    #             # print("test done",done)
    #         state = next_state
    #         episode_reward += reward
    #         step_size += 1

    #         # Print log
    #         result = (f"Episode: {episode}, "
    #                   f"Steps: {step_size:}, "
    #                   f"Reward: {episode_reward:.2f}, ")
    #         print(result)

#     with open("trajectories/mountain_car_PPO_trajectory_7761.pkl", "rb") as f:
#         trajectory = pickle.load(f)

# # Replay the saved trajectory
#     state_index = 0
#     env.reset()
#     env.env.state = [0,0]  # Set the environment to the initial state

#     for state, next_state, action, reward in trajectory:
#         #env.render()  # Render the environment
#         print("Index:",state_index,"State:",next_state)
#         img = env.render()
#         plt.imshow(img)
#         plt.axis('off')
#         plt.show()
#         time.sleep(0.1)  # Add a delay for smoother visualization
#         env.env.state = next_state  # Set the state explicitly
#         state_index += 1

#     env.close()

    #env = gym.wrappers.RecordVideo(env, Path('video_folder'))

    while total_steps < args.max_train_steps:
        s = env.reset(seed=seed)[0]
        print("State",s)
        if args.use_state_norm:
            s = state_norm(s)
        if args.use_reward_scaling:
            reward_scaling.reset()
        episode_steps = 0
        term = False
        trun = False
        while not term and not trun:
            print("total steps:",total_steps)
            episode_steps += 1
            a, a_logprob = agent.choose_action(s)  # Action and the corresponding log probability
            #print("Action",a)
            s_, r, term, trun, _ = env.step(a)
            trajectory.append((s,s_,a,r))
            
            #print("State",s_)
            if term:
                print("terminated")
                # relative_directory = Path('models')
                # file_actor_name = 'mountain_car_actor_PPO' + str(total_steps) + '.pth'
                # file_critic_name = 'mountain_car_critic_PPO' + str(total_steps) + '.pth'
                # trajectory_name = 'mountain_car_PPO_trajectory_' + str(total_steps) + '.pkl'
                # actor_path = relative_directory / file_actor_name
                # critic_path = relative_directory / file_critic_name
                # trajectory_path = Path('trajectories') / trajectory_name
                # torch.save(agent.actor.state_dict(), actor_path)
                # torch.save(agent.critic.state_dict(), critic_path)
                # print('\n~~~~~~Interval Save: Model saved.\n')
                # with open(trajectory_path, "wb") as f:
                #     pickle.dump(trajectory, f)

            if args.use_state_norm:
                s_ = state_norm(s_)
            if args.use_reward_norm:
                r = reward_norm(r)
            elif args.use_reward_scaling:
                r = reward_scaling(r)
            # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
            # dw means dead or win,there is no next state s';
            # but when reaching the max_episode_steps,there is a next state s' actually.
            if term and episode_steps != args.max_episode_steps:
                dw = True
            else:
                dw = False

            replay_buffer.store(s, a, a_logprob, r, s_, dw, term)
            s = s_
            total_steps += 1

            # When the number of transitions in buffer reaches batch_size,then update
            if replay_buffer.count == args.batch_size:
                agent.update(replay_buffer, total_steps)
                replay_buffer.count = 0

            # Evaluate the policy every 'evaluate_freq' steps
            # if total_steps % args.evaluate_freq == 0:
            #     print("total steps:",total_steps)
            #     evaluate_num += 1
            #     evaluate_reward = evaluate_policy(args, seed, env_evaluate, agent, state_norm)
            #     evaluate_rewards.append(evaluate_reward)
            #     # print("evaluate_num:{} \t evaluate_reward:{} \t".format(evaluate_num, evaluate_reward))
            #     # writer.add_scalar('step_rewards_{}'.format(env_name), evaluate_rewards[-1], global_step=total_steps)
            #     # # Save the rewards
            #     # if evaluate_num % args.save_freq == 0:
            #     #     np.save('./data_train/PPO_discrete_env_{}_number_{}_seed_{}.npy'.format(env_name, number, seed), np.array(evaluate_rewards))
            
     

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameter Setting for PPO-discrete")
    parser.add_argument("--max_train_steps", type=int, default=int(2e5), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=5e3, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")

    args = parser.parse_args()

    env_name = ['CartPole-v1', 'MountainCar-v0']
    env_index = 1
    main(args, env_name=env_name[env_index], number=1, seed=0)
