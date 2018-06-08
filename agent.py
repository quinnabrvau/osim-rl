import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.layers import BatchNormalization, LeakyReLU, GaussianNoise
from keras.optimizers import Adam

from rl.util import WhiteningNormalizer
from rl.processors import WhiteningNormalizerProcessor
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

import numpy as np
from math import sin, cos, pi

#Reference: https://github.com/keras-rl/keras-rl/blob/master/examples/ddpg_mujoco.py
class Agent:
    def __init__(self,env):
        self.nb_actions = env.action_space.shape[0]
        self.nb_states  = env.observation_space.shape[0]
        self.env = env
        
        self.actor = self.build_actor(env)
        self.actor.compile('Adam','mse')
        self.critic, action_input = self.build_critic(env)
        self.loss = self.build_loss()
        self.processor = WhiteningNormalizerProcessor()

        self.memory = SequentialMemory(limit=5000000, window_length=1)
        self.random_process = OrnsteinUhlenbeckProcess(size=self.nb_actions, 
                                  theta=0.75, mu=0.5, sigma=0.25)
        self.agent = DDPGAgent(   nb_actions=self.nb_actions, 
                                  actor=self.actor, 
                                  critic=self.critic, critic_action_input=action_input,
                                  memory=self.memory, nb_steps_warmup_critic=100, 
                                  nb_steps_warmup_actor=100,
                                  random_process=self.random_process, 
                                  gamma=.99, target_model_update=1e-3,
                                  processor=self.processor  )
        self.agent.compile([Adam(lr=1e-4), Adam(lr=1e-3)], metrics=self.loss)
        self.sym_actor = self.build_sym_actor()
        self.sym_actor.compile(optimizer='Adam',loss='mse')


    def build_loss(self):
        return ['mse']

    def build_actor(self,env):
        actor = Sequential()
        actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
        actor.add(Dense(64,activation='tanh'))
        actor.add(GaussianNoise(0.05))
        actor.add(Dense(64,activation='tanh'))
        actor.add(GaussianNoise(0.05))
        actor.add(Dense(self.nb_actions,
                        activation='hard_sigmoid' ) )
        actor.summary()

        inD = Input(shape=(1,) + env.observation_space.shape)
        out = actor(inD)

        return Model(inD,out)

    def build_critic(self,env):
        action_input = Input(shape=(self.nb_actions,), name='action_input')
        observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
        flattened_observation = Flatten()(observation_input)
        x = Dense(64,activation='relu')(flattened_observation)
        x = Concatenate()([x, action_input])
        x = Dense(32,activation='relu')(x)
        x = Dense(1)(x)

        critic = Model(inputs=[action_input, observation_input], outputs=x)
        critic.summary()

        return critic, action_input
    
    def build_sym_actor(self):
        stateSwap = []
        actionSwap = []
        state_desc = self.env.get_state_desc()
        for x in state_desc.keys():
            keys = list(state_desc[x].keys())
            for (k,key) in enumerate(keys):
                if '_r' in key:
                    i = keys.index(key.replace('_r','_l'))
                    if i != -1:
                        stateSwap += [ (k,i),(i,k) ]
        muscle_list = []
        for i in range(self.env.osim_model.muscleSet.getSize()):
            muscle_list.append( self.env.osim_model.muscleSet.get(i).getName() )
        for (k,key) in enumerate(muscle_list):
            if '_r' in key:
                i = muscle_list.index(key.replace('_r','_l'))
                if i != -1:
                    actionSwap += [ (k,i),(i,k) ]

        stateSwapMat = np.zeros((self.nb_states,self.nb_states))
        actionSwapMat = np.zeros((self.nb_actions,self.nb_actions))
        stateSwapMat[0,0]
        for (i,j) in stateSwap:
            stateSwapMat[i,j] = 1
        for (i,j) in actionSwap:
            actionSwapMat[i,j] = 1
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
        if 'nb_max_episode_steps' in kwargs.keys():
            self.env.spec.timestep_limit=kwargs['nb_max_episode_steps']
        else:
            self.env.spec.timestep_limit=self.env.time_limit
        out = self.agent.fit(self.env,**kwargs)
        print("\n\ndo symetric loss back propigation\n\n")
        states = np.random.normal(0,10,(kwargs['nb_steps']//200,1,self.nb_states))
        actions = self.actor.predict_on_batch(states)
        self.sym_actor.train_on_batch(states,actions)
        return out
    
    def test(self, **kwargs):
        print("testing")
        print("gravity:",self.env.get_grav(),"VA:",self.env.get_VA())
        if 'nb_max_episode_steps' in kwargs.keys():
            self.env.spec.timestep_limit=kwargs['nb_max_episode_steps']
        else:
            self.env.spec.timestep_limit=self.env.time_limit
        return self.agent.test(self.env,**kwargs)
    
    def test_get_steps(self, **kwargs):
        return self.test(**kwargs).history['nb_steps'][-1]

    def save_weights(self,filename='osim-rl/ddpg_{}_weights.h5f'):
        self.agent.save_weights(filename.format("opensim"), overwrite=True)
        self.save_processor()
        
    def load_weights(self,filename='osim-rl/ddpg_{}_weights.h5f'):
        self.agent.load_weights(filename.format("opensim"))
        self.load_processor()
        
    def search_VA(self):
        # 1-D line search
        state = self.env.get_VA()
        goal = 0.0
        if abs(state-goal)<0.01:
            self.env.upd_VA(goal)
            return
        steps = self.test_get_steps(nb_episodes=1, visualize=False, nb_max_episode_steps=1000)
        dv = 0.0
        dsteps = steps
        while (state-dv>goal and dsteps > 0.8*steps):
            dv += 0.02
            self.env.upd_VA(state-dv)
            dsteps = self.test_get_steps(nb_episodes=1, visualize=False, nb_max_episode_steps=1000)
        if abs( (state-dv) - goal )<0.01:
            self.env.upd_VA(goal)
        else:
            dv -= 0.02
            self.env.upd_VA(state-dv)
            
    def save_processor(self):
        np.savez( 'osim-rl/processor.npz',
                  _sum=self.processor.normalizer._sum,
                  _count=np.array([self.processor.normalizer._count]),
                  _sumsq=self.processor.normalizer._sumsq,
                  mean=self.processor.normalizer.mean,
                  std=self.processor.normalizer.std )
        
    def load_processor(self):
        f = np.load( 'osim-rl/processor.npz' )
        dtype = f['_sum'].dtype
        if (self.processor.normalizer==None):
            self.processor.normalizer = WhiteningNormalizer(shape=(1,)+self.env.observation_space.shape, dtype=dtype)
        self.processor.normalizer._sum = f['_sum']
        self.processor.normalizer._count = int(f['_count'][0])
        self.processor.normalizer._sumsq = f['_sumsq']
        self.processor.normalizer.mean = f['mean']
        self.processor.normalizer.std = f['std']

if __name__=='__main__':
    from osim.env import L2RunEnv as ENV 
    env = ENV(visualize=True)
    env.reset()
    agent = Agent(env)
    env.osim_model.list_elements()
    agent.fit(nb_steps=150, visualize=False, verbose=2, nb_max_episode_steps=1000)
    h = agent.test(nb_episodes=5, visualize=True, nb_max_episode_steps=1000)
    agent.search_VA()
    agent.save_weights()
    agent2 = Agent(env)
    agent2.load_weights()
    
    

