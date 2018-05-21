
import numpy as np

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam

from rl.agents import DDPGAgent


#Reference: https://github.com/keras-rl/keras-rl/blob/master/examples/ddpg_mujoco.py
class Agent:
    def __init__(self,env):
        nb_actions = env.action_space.shape[0]
        self.actor = self.build_actor(env)
        self.critic, action_input = self.build_critic(env)
        self.loss = self.build_loss()

        self.memory = SequentialMemory(limit=100000, window_length=1)
        self.random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.1)
        self.agent = DDPGAgent(   nb_actions=nb_actions, actor=self.actor, 
                                  critic=self.critic, critic_action_input=action_input,
                                  memory=self.memory, nb_steps_warmup_critic=1000, 
                                  nb_steps_warmup_actor=1000,
                                  random_process=self.random_process, 
                                  gamma=.99, target_model_update=1e-3,
                                  processor=MujocoProcessor()  )
        self.agent.compile([Adam(lr=1e-4), Adam(lr=1e-3)], metrics=self.loss)

    def build_loss(self):
        return ['mse']

    def build_actor(self,env):
        nb_actions = env.action_space.shape[0]
        actor = Sequential()
        actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
        actor.add(Dense(400))
        actor.add(Activation('relu'))
        actor.add(Dense(300))
        actor.add(Activation('relu'))
        actor.add(Dense(nb_actions))
        actor.add(Activation('tanh'))
        actor.summary()

        inD = Input(shape=(1,) + env.observation_space.shape)
        out = actor(inD)

        return Model(inD,out)

    def build_critic(self,env):
        nb_actions = env.action_space.shape[0]
        action_input = Input(shape=(nb_actions,), name='action_input')
        observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
        flattened_observation = Flatten()(observation_input)
        x = Dense(400)(flattened_observation)
        x = Activation('relu')(x)
        x = Concatenate()([x, action_input])
        x = Dense(300)(x)
        x = Activation('relu')(x)
        x = Dense(1)(x)
        x = Activation('linear')(x)

        critic = Model(inputs=[action_input, observation_input], outputs=x)
        critic.summary()

        return critic, action_input





