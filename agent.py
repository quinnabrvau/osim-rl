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
from math import sin, cos, pi

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
                                  theta=0.75, mu=0.5, sigma=0.25)
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
        actor.add(Dense(200))
        actor.add(LeakyReLU(alpha=0.2))
        actor.add(Dense(200))
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
        x = Dense(200)(flattened_observation)
        x = LeakyReLU(alpha=0.2)(x)
        x = Concatenate()([x, action_input])
        x = Dense(200)(x)
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
#        print("Do symetric loss back propigation")
#        states = np.random.normal(0,pi/2,(kwargs['nb_steps'],1,self.nb_states))
#        actions = self.actor.predict_on_batch(states)
#        self.sym_actor.train_on_batch(states,actions)
        return out
    
    def test(self, **kwargs):
        return self.agent.test(self.env,**kwargs)
    
    def test_get_steps(self, **kwargs):
        print("testing")
        print("gravity:",self.env.get_grav(),"VA:",self.env.get_VA())
        return self.test(**kwargs).history['nb_steps'][-1]
    
    def save_weights(self,filename='ddpg_{}_weights.h5f'):
        self.agent.save_weights(filename.format("opensim"), overwrite=True)
        
    def load_weights(self,filename='ddpg_{}_weights.h5f'):
        self.agent.load_weights(filename.format("opensim"))
        
    def search_VA(self):
        va_state = [self.env.get_grav(),self.env.get_VA()]
        va_goal = [1.0, 0.0]
        if va_state[0]==va_goal[0] and va_state[1]==va_goal[1]:
            return
        theta = np.linspace(0,pi/2,5)
        print(theta)
        theta_, dist_ = 0, 0
        print("current test")
#        target_percent = 0.6 * self.test_get_steps( nb_episodes=1, 
#                                                    visualize=True, 
#                                                    nb_max_episode_steps=1000 )
        direct = [ va_goal[0]-va_state[0], va_goal[1]-va_state[1] ]
        t_dist = direct[0]**2+direct[1]**2
        if t_dist < 0.1:
            self.env.upd_grav(va_goal[0])
            self.env.upd_VA(va_goal[1])
            return
#        for t in theta:
#            d = 1
#            self.env.upd_grav(va_state[0]+d*direct[0]*cos(t))
#            self.env.upd_VA(va_state[1]+d*direct[1]*sin(t))
#            print(t,d)
#            test_percent = self.test_get_steps(     nb_episodes=1, 
#                                                    visualize=True, 
#                                                    nb_max_episode_steps=1000 )
#            while(test_percent<target_percent and d>0.05):
#                print("test percent < target precent", test_percent<target_percent,
#                      "d",d)
#                d -= (d/4)
#                self.env.upd_grav(va_state[0]+d*direct[0]*cos(t))
#                self.env.upd_VA(va_state[1]+d*direct[1]*sin(t))
#                print(t,d)
#                test_percent = self.test_get_steps( nb_episodes=1, 
#                                                    visualize=True, 
#                                                    nb_max_episode_steps=1000 )
#            if d>dist_:
#                theta_, dist_ = t, d
#        dist_ = direct[0]*direct[0]- direct[1]*direct[1]
        dist_ = t_dist / 4
        self.env.upd_grav(va_state[0]+direct[0]/2)
        self.env.upd_VA(va_state[1]+direct[1]/2)
        
        

if __name__=='__main__':
    from osim.env import L2RunEnv as ENV 
    env = ENV(visualize=True)
    agent = Agent(env)
    env.osim_model.list_elements()
    
    
    
    
    
    
    
    