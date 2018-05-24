import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate, LeakyReLU
from keras.optimizers import Adam

from rl.core import Processor
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

import numpy as np

from math import pi

#Reference: https://github.com/keras-rl/keras-rl/blob/master/examples/ddpg_mujoco.py
class Agent:
    def __init__(self,env):
        self.nb_actions = env.action_space.shape[0]
        self.nb_states  = env.observation_space.shape[0]
        
        self.env = env
        self.actor = self.build_actor(env)
        self.critic, action_input = self.build_critic(env)
        self.loss = self.build_loss()

        self.memory = SequentialMemory(limit=100000, window_length=1)
        self.random_process = OrnsteinUhlenbeckProcess(size=self.nb_actions, 
                                  theta=0.15, mu=0.5, sigma=0.5)
        self.agent = DDPGAgent(   nb_actions=self.nb_actions, 
                                  actor=self.actor, 
                                  critic=self.critic, critic_action_input=action_input,
                                  memory=self.memory, nb_steps_warmup_critic=1000, 
                                  nb_steps_warmup_actor=1000,
                                  random_process=self.random_process, 
                                  gamma=.99, target_model_update=1e-3,
                                  processor=None  )
        self.agent.compile([Adam(lr=1e-4), Adam(lr=1e-3)], metrics=self.loss)
        self.sym_actor = self.build_sym_actor()
        self.sym_actor.compile(optimizer='Adam',loss='mse')


    def build_loss(self):
        return ['mse']

    def build_actor(self,env):
        actor = Sequential()
        actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
        actor.add(Dense(400))
        actor.add(LeakyReLU(alpha=0.2))
        actor.add(Dense(400))
        actor.add(LeakyReLU(alpha=0.2))
        actor.add(Dense(self.nb_actions,
                        activation='tanh' ) )
        actor.summary()

        inD = Input(shape=(1,) + env.observation_space.shape)
        out = actor(inD)

        return Model(inD,out)

    def build_critic(self,env):
        action_input = Input(shape=(self.nb_actions,), name='action_input')
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
    
    def build_sym_actor(self):
        stateSwaps = [ 
          (6,8),  (7,9),   #hip_l, hip_r
          (10,12),(11,13), #knee_l, knee_r
          (14,16),(15,17), #ankle_l, ankle_r
          (24,26),(25,27), #toes_l, toes_r
          (23,26),(24,27), #talus_l, talus_r
          ]
        actionSwaps = [ (0,9),(1,10),(2,11),(3,12),(4,13),(5,14),
                        (6,15),(7,16),(8,17) ]
        stateSwapMat = np.zeros((self.nb_states,self.nb_states))
        actionSwapMat = np.zeros((self.nb_actions,self.nb_actions))
        stateSwapMat[0,0]
        for (i,j) in stateSwaps:
            stateSwapMat[i,j] = 1
            stateSwapMat[j,i] = 1
        for (i,j) in actionSwaps:
            actionSwapMat[i,j] = 1
            actionSwapMat[j,i] = 1
        def ssT(shape,dtype=None):
            if shape!=stateSwapMat.shape:
                raise Exception("State Swap Tensor Shape Error")
            return K.variable(stateSwapMat,dtype=dtype)
        def asT(shape,dtype=None):
            if shape!=actionSwapMat.shape:
                raise Exception("Action Swap Tensor Shape Error")
            return K.variable(actionSwapMat,dtype=dtype)
        
        model1 = Sequential()
        model1.add( Dense(self.nb_states,
                         input_shape =(1,) + self.env.observation_space.shape,
                         trainable=False,
                         kernel_initializer=ssT,
                         bias_initializer='zeros' ) )
        inD = Input(shape=(1,) + self.env.observation_space.shape)
        symState = model1(inD)
        symPol = self.actor(symState)
        model2 = Sequential()
        model2.add( Dense(self.nb_actions,
                         input_shape = (1,self.nb_actions),
                         trainable=False,
                         kernel_initializer=asT,
                         bias_initializer='zeros' ) )
        out = model2(symPol)
        
        return Model(inD,out)
        
        
        
    
    def fit(self, **kwargs):
        out = self.agent.fit(self.env,**kwargs)
        print("Do symetric loss back propigation")
        states = np.random.normal(0,pi/2,(kwargs['nb_steps'],1,self.nb_states))
        actions = self.actor.predict_on_batch(states)
        self.sym_actor.train_on_batch(states,actions)
        return out
    
    def test(self, **kwargs):
        return self.agent.test(self.env,**kwargs)
    
    def save_weights(self,filename='osim-rl/ddpg_{}_weights.h5f'):
        self.agent.save_weights(filename.format("opensim"), overwrite=True)
        
    def load_weights(self,filename='osim-rl/ddpg_{}_weights.h5f'):
        self.agent.load_weights(filename.format("opensim"))
        
#class SymetricProcessor(Processor):
#    def process_reward(self,reward):
#        if GlobalAgent is None:
#            return reward
#        symetric_reward = 0
#        
#        state = np.random.normal(0,pi/2,(nb_actions,))
#        pol = GlobalAgent.actor.predict(state)
#        
#        symState = transState(state)
#        symPol = transAction(GlobalAgent.actor.predict(symState))
#        
#        symetric_reward = np.sum(np.abs(np.subtract(pol,symPol)))/1000
#        
#        return reward-symetric_reward
# 
#swaps = [ (2,3), #hip_l, hip_r
#          (4,5), #knee_l, knee_r
#          (6,7), #ankle_l, ankle_r
#          (17,20),(18,21),(19,22), #toes_l, toes_r
#          (23,26),(24,27),(25,28)#talus_l, talus_r
#          ]
#def transState(state):
#    for swap in swaps:
#        foo = state[swap[0]]
#        state[swap[0]]=state[swap[1]]
#        state[swap[1]]=foo
#    return state
#
#def transAction(action):
#    action[:9], action[9:] = action[9:], action[:9]

if __name__=='__main__':
    from osim.env import L2RunEnv as ENV 
    env = ENV(visualize=False)
    agent = Agent(env)
    env.osim_model.list_elements()
    
    
    
    
    
    
    
    