import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate, LeakyReLU
from keras.optimizers import Adam

from rl.processors import WhiteningNormalizerProcessor
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess


#Reference: https://github.com/keras-rl/keras-rl/blob/master/examples/ddpg_mujoco.py
class Agent:
    def __init__(self,env):
        nb_actions = env.action_space.shape[0]
        
        self.env = env
        self.actor = self.build_actor(env)
        self.critic, action_input = self.build_critic(env)
        self.loss = self.build_loss()

        self.memory = SequentialMemory(limit=100000, window_length=1)
        self.random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=0.15, mu=0.5, sigma=0.5)
        self.agent = DDPGAgent(   nb_actions=nb_actions, actor=self.actor, 
                                  critic=self.critic, critic_action_input=action_input,
                                  memory=self.memory, nb_steps_warmup_critic=1000, 
                                  nb_steps_warmup_actor=1000,
                                  random_process=self.random_process, 
                                  gamma=.99, target_model_update=1e-3,
                                  processor=None  )
        self.agent.compile([Adam(lr=1e-4), Adam(lr=1e-3)], metrics=self.loss)

    def build_loss(self):
        return ['mse']

    def build_actor(self,env):
        nb_actions = env.action_space.shape[0]
        actor = Sequential()
        actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
        actor.add(Dense(400))
        actor.add(LeakyReLU(alpha=0.2))
        actor.add(Dense(400))
        actor.add(LeakyReLU(alpha=0.2))
        actor.add(Dense(nb_actions,
                        activation='tanh' ) )
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
        x = LeakyReLU(alpha=0.2)(x)
        x = Concatenate()([x, action_input])
        x = Dense(400)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dense(1)(x)
        x = Activation('linear')(x)

        critic = Model(inputs=[action_input, observation_input], outputs=x)
        critic.summary()

        return critic, action_input
    
    def fit(self, **kwargs):
        return self.agent.fit(self.env,**kwargs)
    
    def test(self, **kwargs):
        return self.agent.test(self.env,**kwargs)
    
    def save_weights(self,filename='ddpg_{}_weights.h5f'):
        self.agent.save_weights(filename.format("opensim"), overwrite=True)
        
    def load_weights(self,filename='ddpg_{}_weights.h5f'):
        self.agent.load_weights(filename.format("opensim"))