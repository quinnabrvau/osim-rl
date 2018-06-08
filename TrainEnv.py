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
    terminal_height = 0.7
    primary_joint = "ground_pelvis"
    def __init__(self,**kwargs):
        ENV.__init__(self,**kwargs)
        self.grav = self.osim_model.model.getGravity()
        self.upd_VA(0.0)    
    
    def upd_VA(self,new_force=0.0):
        self.grav.set(0,new_force)
        self.osim_model.model.setGravity(self.grav)
            
    def get_VA(self):
        return self.osim_model.model.getGravity().get(0)
    
    def reward(self):
        state_desc = self.get_state_desc()
        p_state_desc = self.get_prev_state_desc()
        if not p_state_desc:
            return 0
        #hieght reward for standing tall
        hieght_reward = state_desc["body_pos"]["pelvis"][1] -(self.terminal_height)
        #velocity reward for moving forward
        velocity_reward = (state_desc["joint_pos"][self.primary_joint][0] -
                           p_state_desc["joint_pos"][self.primary_joint][0] )
        return hieght_reward+velocity_reward

        
if __name__=='__main__':
    env = TrainEnv(visualize=False)
    env.reset()
    sim = env.osim_model
    print(env.get_VA())
    env.upd_VA(8.0)
    print(env.get_VA())
    state_desc = env.osim_model.get_state_desc()
    print('ground_pelvis pos',state_desc["joint_pos"]["ground_pelvis"])
    print('ground_pelvis vel',state_desc["joint_vel"]["ground_pelvis"])

    for joint in ["hip_l","hip_r","knee_l","knee_r","ankle_l","ankle_r",]:
        print(joint+' pos',state_desc["joint_pos"][joint])
        print(joint+' vel',state_desc["joint_vel"][joint])

    for body_part in ["head", "pelvis", "torso", "toes_l", "toes_r", "talus_l", "talus_r"]:
        print(body_part,state_desc["body_pos"][body_part][0:2])

    print('center of mass pos',state_desc["misc"]["mass_center_pos"])
    print('center of mass vel',state_desc["misc"]["mass_center_vel"])

    
    