# Dependencies
import numpy as np
import tensorflow as tf
from random import uniform
# Environment
from TrainEnv import TrainEnv # rename environment to be used for training
# Agent
from agent import Agent


env = TrainEnv(visualize=False,integrator_accuracy = 5e-3)
env.upd_VA(3.0)
observation = env.reset( )

agent = Agent(env)
GlobalAgent = agent
T_steps = 5000
W_steps = 1000
#agent.load_weights( )
h = agent.test_get_steps(nb_episodes=1, visualize=True, nb_max_episode_steps=1000)
data = [h]
for i in range(1000): # Train in smaller batches to allow for interuption
    print("\n\niteration:",i)
    print(agent.env.get_VA())
    agent.fit(nb_steps=T_steps, visualize=False, verbose=2, nb_max_episode_steps=W_steps)
    ## Always save new weights
    agent.save_weights( )
    
    steps_ = agent.test_get_steps(nb_episodes=1, visualize=True, nb_max_episode_steps=W_steps)
    data.append(h)
    np.savez('osim-rl/steps_plot.npz',data=data)
    if steps_>(W_steps*7)//10:
        T_steps = (T_steps*5)//4
        W_steps = (W_steps*3)//2
        agent.search_VA()

from matplotlib import pyplot as plt
plt.plot(data)
plt.title('run steps')
plt.xlabel('iterations')
plt.savefig('osim-rl/steps_plot.png')

# Finally, evaluate our algorithm for 5 episodes.
h = agent.test(nb_episodes=5, visualize=True, nb_max_episode_steps=1000)
env.close()
