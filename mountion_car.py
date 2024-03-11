import gymnasium as gym
import numpy as np
import random

# env for training the q_table
env = gym.make('MountainCar-v0', render_mode="rgb_array")

# Divide position and velocity into segments
pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 20)    # Between -1.2 and 0.6
vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 20)    # Between -0.07 and 0.07


# create the q_table 
q_table = np.zeros((len(pos_space), len(vel_space), 3))

# Divide position and velocity into segments with agent
def get_state_and_velocity(state_and_velocity, def_pos_space=pos_space, def_vel_space=vel_space):
    state = np.digitize(state_and_velocity[0], def_pos_space)
    velocity = np.digitize(state_and_velocity[1], def_vel_space)
    
    return int(state), int(velocity)


choice_list = ['x'] *100
learning_rate = 0.1
gamma = 0.99
action_space = [0, 1, 2] #  0 for left, 1 for stay, 2 for right

for i in range(10000):
    done = False
    
    now = env.reset()[0]
    while not done:
        now_state, now_velocity = get_state_and_velocity(now)
        
        # Choose action by % chance of choice_list
        selected_item = random.choice(choice_list)
        if selected_item == "x": 
            action = random.choice(action_space)
        elif selected_item == "y":
            try:
                action = np.argmax(q_table[now_state, now_velocity])
            except:
                action = random.choice(action_space)
        

        next = env.step(action)
        next_state, next_velocity = get_state_and_velocity(next[0])
        reward = next[1]
        done = next[2]
        
       # Update Q-value using Bellman equation
        q_table[now_state, now_velocity, action] = (1 - learning_rate) * q_table[now_state, now_velocity, action] + learning_rate * (reward + gamma * np.max(q_table[next_state, next_velocity]))
        now = next[0]
        
    # chance choice_list every 100 steps 
    if (i+1) % 100 == 0:
        choice_list.remove("x")
        choice_list.append("y")
        

# env2 for testing the final q table
env2 = gym.make('MountainCar-v0', render_mode="human")

for i in range(10):
    done = False
    
    now = env2.reset()[0]
    while not done:
        now_state, now_velocity = get_state_and_velocity(now)
        
        # get maximum predicted value from the q table
        action = np.argmax(q_table[now_state, now_velocity])
        
        next = env2.step(action)
        next_state, next_velocity = get_state_and_velocity(next[0])
        reward = next[1]
        done = next[2]
    
        
       # Update Q-value using Bellman equation
        q_table[now_state, now_velocity, action] = (1 - learning_rate) * q_table[now_state, now_velocity, action] + learning_rate * (reward + gamma * np.max(q_table[next_state, next_velocity]))
        now = next[0]
