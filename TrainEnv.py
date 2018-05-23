#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 20 20:42:18 2018

@author: quinn
"""
import warnings
import opensim
import os
from osim.env import L2RunEnv as ENV # rename environment to be used for training

class TrainEnv(ENV):
#    model_path = os.path.join(os.path.dirname(__file__), 'models/Skel_model_with_VA.osim')
    def __init__(self,**kwargs):
        ENV.__init__(self,**kwargs)
        self.Grav = self.osim_model.model.getGravity()
        self.Grav.set(1,-6)
        self.upd_VA(4)
            
    def upd_VA(self,new_force=1.0):
        self.Grav.set(0,new_force)
        self.osim_model.model.setGravity(self.Grav)
#        self.osim_model.model.upd_gravity()
            
    def get_VA(self):
        return self.osim_model.model.getGravity().get(0)
 
        
        
        
        
# TODO: define virtual assistant forces on agent
# TODO: define search through easier environments
# TODO: make environment harder once the agent has trained for challenge
        
        
if __name__=='__main__':
    env = TrainEnv(visualize=True)
    sim = env.osim_model
    print(env.get_VA())
    env.upd_VA(8.0)
    print(env.get_VA())
    
    