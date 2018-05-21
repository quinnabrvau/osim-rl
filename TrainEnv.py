#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 20 20:42:18 2018

@author: quinn
"""
from opensim import Force
import os
from osim.env import L2RunEnv as ENV # rename environment to be used for training

class TrainEnv(ENV):
    model_path = os.path.join(os.path.dirname(__file__), 'Skel_model_with_VA.osim')
    def __init__(self,**kwargs):
        ENV.__init__(self,**kwargs)
        
        self.VA_force = self.osim_model.get_force("VA_forward")
        print(self.VA_force.getInputNames())
        print(self.VA_force)
        
    def set_VA_force(self,force):
        if force>1:
            force = 1
        if force<0:
            force = 0
            
        self.VA_force
        
        
# TODO: define virtual assistant forces on agent
# TODO: define search through easier environments
# TODO: make environment harder once the agent has trained for challenge
        
        
if __name__=='__main__':
    env = TrainEnv(visualize=False)