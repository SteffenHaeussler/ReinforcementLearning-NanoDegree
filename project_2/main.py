from collections import namedtuple, deque
from os.path import abspath, dirname, join
import time

import numpy as np
import torch

from unityagents import UnityEnvironment
from agent.agent import Agent

agent_config = {
    'BUFFER_SIZE': int(1e6),  # replay buffer size
    'BATCH_SIZE': 128,        # minibatch size
    'GAMMA': 0.99,            # discount factor
    'TAU': 1e-3,              # for soft update of target parameters
    'LR_ACTOR': 1e-4,         # learning rate of the actor
    'LR_CRITIC': 1e-3,        # learning rate of the critic
    'WEIGHT_DECAY': 0.0,        # L2 weight decay
    "UPDATE_FREQUENCY": 4,    # how long to wait to update the network
    "UPDATE_STEPS": 1    # how often to update the network
}


def ddpg(env, n_episodes=1000, print_every=20):
    scores_deque = deque(maxlen=100)
    scores_global = []

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]

    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))

    env_info = env.reset(train_mode=True)[brain_name]
    agent = Agent(state_size=state_size, action_size=action_size, num_agents=num_agents, agent_config=agent_config, device=device, random_seed=123)

    time_start = time.time()

    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]      # reset the environment
        states = env_info.vector_observations                  # get the current state (for each agent)
        scores = np.zeros(num_agents)                          # initialize the score (for each agent)
        agent.reset()
        score_average = 0

        for timestep in range(1000):
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]           # send all actions to tne environment
            next_states = env_info.vector_observations         # get next state (for each agent)
            rewards = env_info.rewards                         # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished
            agent.step(states, actions, rewards, next_states, dones, timestep)
            states = next_states                               # roll over states to next time step
            scores += rewards                                  # update the score (for each agent)
            if np.any(dones):                                  # exit loop if episode finished
                break

        score = np.mean(scores)
        scores_deque.append(score)
        score_average = np.mean(scores_deque)
        scores_global.append(score)

        print('\rEpisode: {}\tAverage Score: {:.2f}\tMax: {:.2f}\tMin: {:.2f}'.format(i_episode, score_average, np.max(scores), np.min(scores)), end="")

        if i_episode % print_every == 0 or (len(scores_deque) == 100 and np.mean(scores_deque) >= 30) :
            torch.save(agent.actor_local.state_dict(), 'model_actor_local.pth')
            torch.save(agent.critic_local.state_dict(), 'model_critic_local.pth')
            torch.save(agent.actor_target.state_dict(), 'model_actor_target.pth')
            torch.save(agent.critic_target.state_dict(), 'model_critic_target.pth')

            s = (int)(time.time() - time_start)
            print('\rEpisode {}\tAverage Score: {:.2f}, Time: {:02}:{:02}:{:02} *** '\
                  .format(i_episode, score_average, s//3600, s%3600//60, s%60))

        if len(scores_deque) == 100 and np.mean(scores_deque) >= 30:
            print('Environment solved !')
            break

    return scores_global


def main():

    env = UnityEnvironment("/usr/src/app/Reacher_Linux_NoVis/Reacher.x86_64")

    scores = ddpg(env)

    return None


if __name__ == "__main__":

    main()
