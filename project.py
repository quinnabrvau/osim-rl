# Dependencies
import numpy as np
import tensorflow as tf
from random import uniform
# Environment
from TrainEnv import TrainEnv # rename environment to be used for training
# Agent
from agent import Agent


env = TrainEnv(visualize=False,integrator_accuracy = 5e-5)
observation = env.reset( )

agent = Agent(env)

agent.load_weights( )

for i in range(0): # Train in smaller batches to allow for interuption
    print("\n\niteration:",i)
    agent.fit(nb_steps=5000, visualize=False, verbose=2)
    ## Always save new weights
    agent.save_weights( )
    
    env.get_VA()
#    env.upd_VA(uniform(0,1.5))

# Finally, evaluate our algorithm for 5 episodes.
agent.test(nb_episodes=10, visualize=True, nb_max_episode_steps=1000)
env.close()
