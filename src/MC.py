#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Yonglin Wang
# Date: 1/13/2021

import gym  # OpenAI Gym
import numpy as np
from IPython.display import clear_output
from time import sleep
import random
from collections import defaultdict
import os
import itertools

RANDOM_SEED = 2020

def cls():
    """
    cross-platform helper function to clear screen in console
    :return:
    """
    os.system('cls' if os.name=='nt' else 'clear')


def create_random_policy(env) -> dict:
    """
    create a random policy that assigns equal probability to actions in each state TODO: "iterate through the observation space sampling probability distribution??"讲的什么玩意
    :param env:
    :return: policy dictionary of {state_index: {action_index: action_probability}}
    """
    # initialize an empty policy dictionary
    policy = {}

    for key in range(0, env.observation_space.n):
        # p = {}
        # for action in range(0, env.action_space.n):
        #     # To avoid the assumption of Exploring Starts, we choose to consider only policies that
        #     # are stochastic with a nonzero probability of selecting all actions in each state. (S&B 5.4)
        #     # One solution is that we assume all possible actions are equiprobable (=random policy).
        #     p[action] = 1 / env.action_space.n
        #
        # # record current state's action probabilities
        # # TODO simplify this with comprehension
        # policy[key] = p

        policy[key] = {action: 1/env.action_space.n for action in range(env.action_space.n)}

    return policy


def create_state_action_dictionary(env, policy: dict) -> dict:
    """
    return initial Q (state-action) matrix storing 0 as reward for each action in each state
    :param env:
    :param policy: 2D dictionary of {state_index: {action_index: action_probability}}
    :return: Q matrix dictionary of {state_index: {action_index: reward}}
    """
    # Note: We use optimal action-value function instead of value function
    # to cache the one-step-ahead searches (trade off space for time)

    # initialize Q matrix
    Q = {}

    # traverse states in a policy
    for key in policy.keys():
        # for each state, assign each action 0 as reward
        Q[key] = {a: 0.0 for a in range(0, env.action_space.n)}
    return Q


def run_game(env,
             policy,
             display=True) -> list:
    """
    Run one forward pass (e.g. 1 episode) of the game, and return time step results
    :param env: FrozenLake environment
    :param policy: 2D dictionary of {state_index: {action_index: action_probability}}
    :param display: whether to render agent-environment interaction
    :return: list of (state, action, reward) triples at each time step
    """
    # reset the environment for a new episode
    env.reset()

    # create episode array and end flag
    episode = []
    finished = False

    while not finished:
        # record current state
        s = env.env.s

        # display current time step resulting from last state action pair
        if display:
            clear_output(True)
            env.render()
            sleep(1)

        # create current time step of (state, action, reward) array
        timestep = [s]

        # choose a random probability threshold for selecting actions TODO Probability? 那不就是1？
        n = random.uniform(0, sum(policy[s].values()))
        top_range = 0

        # iterate over Q matrix of policies, and select action with highest probability
        action = None
        for prob in policy[s].items():
            top_range += prob[1]

            # if the TODO i don't get this... why are we choosing based on policy matrix???
            if n < top_range:
                action = prob[0]
                break

        # gather consequences of action
        state, reward, finished, info = env.step(action)

        # record action and reward into time step
        timestep.append(action)
        timestep.append(reward)

        # record time step into episode
        episode.append(timestep)

    # display final time step
    if display:
        clear_output(True)
        env.render()
        sleep(1)

    return episode


# wrap run game episode and adjust policy after episodes
def eval_policy(policy: dict, env, episode_num=100) -> float:
    # record wins
    wins = 0

    # run the designated number of episodes
    for i in range(episode_num):
        # retrieve the reward of last time step in the last episode
        reward = run_game(env, policy, display=False)[-1][-1]
        if reward == 1:
            wins += 1

    return wins / episode_num


def mc_first_eps_soft(env,
                      episodes=5000,
                      policy=None,
                      epsilon=0.01,
                      discount=1,
                      display=False,
                      display_every=500):
    """
    Based on given environment and parameters, train a given/random policy, using on-policy
    first-visit MC control for epsilon-soft policies.
    :param env: FrozenLake environment
    :param episodes: integer number of total episodes to train policy
    :param policy: 2D dictionary of {state_index: {action_index: action_probability}}
    :param epsilon: float epsilon value in e-soft policy, usually very small
    :param discount: float in [0,1] of discount value; no discount applied if 1
    :param display: whether to render agent-environment interaction every certain number of episodes
    :param display_every: integer number of episodes before each display; closer to 1, slower the training
    :return:
    """
    # create new random policy if none specified
    if not policy:
        policy = create_random_policy(env)

    # initialize empty Q matrix, note: it records action values rather than state values
    Q = create_state_action_dictionary(env, policy)

    # record cumulative reward values for each state action pairs for updating state-action values
    # Note that in MC we take average of returns ACROSS episodes
    returns = defaultdict(list)

    for e_index in range(episodes):
        G = 0   # cumulative reward

        # Forward pass
        # compute time step info (state, action, value) for current episode
        if display and e_index % display_every == 0:
            print(f"Now displaying Agent behavior after {e_index} episodes...")
            episode = run_game(env, policy, display=True)
        else:
            episode = run_game(env, policy, display=False)

        # Back propagation

        # helper set to record seen pairs in each episode
        seen_sa_pairs = set()

        # traverse time steps backwards
        for i in reversed(range(0, len(episode))):
            s_t, a_t, r_t = episode[i]  # state, action, reward at time t
            state_action = (s_t, a_t)

            # record cumulative rewards with discount factor
            G = discount * G + r_t

            # if state_action is new, update action-value matrix and policy TODO why only the seen ones? visit once?
            # TODO add every visit version?
            if state_action not in seen_sa_pairs:
                # correlate rewards with state action pairs
                returns[state_action].append(G)

                # update value matrix for current SA pair, averaging across all time steps in all episodes
                Q[s_t][a_t] = sum(returns[state_action]) / len(returns[state_action])

                # Find index of an action with maximum action value (a_star), with ties broken arbitrarily
                Q_list = [val for _, val in Q[s_t].items()]
                max_value = max(Q_list)
                a_star = random.choice([i for i, x in enumerate(Q_list) if x == max_value])

                # update action probability for s_t in policy TODO why is policy updated this way? is epsilon used correctly here?
                for a in policy[s_t].keys():
                    if a == a_star:
                        policy[s_t][a] = 1 - epsilon + (epsilon / abs(sum(policy[s_t].values())))
                    else:
                        policy[s_t][a] = epsilon / abs(sum(policy[s_t].values()))

            # record current SA pair as seen at the end of time step
            seen_sa_pairs.add(state_action)

    return policy


if __name__ == "__main__":
    ###
    # params to tune
    slippery = False    # if slippery, actions will not be executed 100% of the time on Lake
    episodes = 5000     # number of episodes to train a policy
    rounds = 4          # total rounds of policy reset-train-test
    display = True     # whether to display the simulation (very slow)
    display_every = 2000 # display once after this number of episodes

    ###
    round_acc_pair = []

    random.seed(RANDOM_SEED)
    env = gym.make('FrozenLake8x8-v0', is_slippery=slippery)
    env.seed(RANDOM_SEED)

    # Debug with slippery, rand state 2020, 5000 episodes:
    # Round 1: 0.22, Round 2: 0.04, 3: 0.12, 4: 0.42
    for t in range(rounds):
        print(f"Round {t + 1}")
        policy = mc_first_eps_soft(env, episodes=episodes, display=display, display_every=display_every)
        acc = eval_policy(policy, env)
        round_acc_pair.append((t + 1, acc))
        print(f"Accuracy: {acc}")