#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 20 20:42:18 2018

@author: quinn
"""
import warnings
import opensim
import os
from osim.env import ProstheticsEnv as ENV # rename environment to be used for training

class TrainEnv(ENV):
    prosthetic = False
#    model_path = os.path.join(os.path.dirname(__file__), 'models/Skel_model_with_VA.osim')
    terminal_height = 0.7
    primary_joint = "ground_pelvis"
    def __init__(self,**kwargs):
        ENV.__init__(self,**kwargs)
        self.grav = self.osim_model.model.getGravity()
        self.gravReal = self.grav.get(1)
        self.upd_grav(8)
        self.upd_VA(1.5)
    
    def upd_grav(self,new_grav=9.8):
        if new_grav<1.1*self.gravReal:
            warnings.warn('new gravity value too large, setting gravity to 1.1G')
            new_grav=1.1*self.gravReal
        elif new_grav>0.3*self.gravReal:
            warnings.warn('new gravity value too small, setting gravity to 0.3G')
            new_grav=0.3*self.gravReal
        self.grav.set(1,new_grav)
        self.osim_model.model.setGravity(self.grav)
        
    def get_grav_real(self):
        return self.gravReal
    
    def get_grav(self):
        return self.osim_model.model.getGravity().get(1)
    
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
        hieght_reward = state_desc["body_pos"]["pelvis"][1] - self.terminal_height
        if hieght_reward > 0.1:
            hieght_reward = 0.1
        #velocity reward for moving forward
        velocity_reward = (state_desc["joint_pos"][self.primary_joint][0] -
                           p_state_desc["joint_pos"][self.primary_joint][0] )
        step_reward = (self.osim_model.istep/10000)
        return hieght_reward+velocity_reward+step_reward
        

 
        
        
        
        
# TODO: define virtual assistant forces on agent
# TODO: define search through easier environments
# TODO: make environment harder once the agent has trained for challenge
        
        
if __name__=='__main__':
    env = TrainEnv(visualize=False)
    env.reset()
    sim = env.osim_model
    r = env.get_observation()
    print('len',len(r))
    i = 0
    print(env.get_VA())
    env.upd_VA(8.0)
    print(env.get_VA())
    state_desc = env.osim_model.get_state_desc()
    for body_part in ["pelvis"]:
        print(body_part)
        print(i,"\tbody_pos",state_desc["body_pos"][body_part][1:2])
        i+=1
        print(i,"\tbody_vel",state_desc["body_vel"][body_part][0:2])
        i+=2
        print(i,"\tbody_acc",state_desc["body_acc"][body_part][0:2])
        i+=2
        print(i,"\tbody_pos_rot",state_desc["body_pos_rot"][body_part][2:])
        i+=1
        print(i,"\tbody_vel_rot",state_desc["body_vel_rot"][body_part][2:])
        i+=1
        print(i,"\tbody_acc_rot",state_desc["body_acc_rot"][body_part][2:])
        i+=1
    for body_part in ["head","torso","toes_l","toes_r","talus_l","talus_r"]:
        print(body_part)
        print(i,"\tbody_pos",state_desc["body_pos"][body_part][0:2])
        i+=2
        print(i,"\tbody_vel",state_desc["body_vel"][body_part][0:2])
        i+=2
        print(i,"\tbody_acc",state_desc["body_acc"][body_part][0:2])
        i+=2
        print(i,"\tbody_pos_rot",state_desc["body_pos_rot"][body_part][2:])
        i+=1
        print(i,"\tbody_vel_rot",state_desc["body_vel_rot"][body_part][2:])
        i+=1
        print(i,"\tbody_acc_rot",state_desc["body_acc_rot"][body_part][2:])
        i+=1

    for joint in ["ankle_l","ankle_r","back","hip_l","hip_r","knee_l","knee_r"]:
        print(joint)
        print(i,'\tjoint_pos',state_desc["joint_pos"][joint])
        i += len(state_desc["joint_pos"][joint])
        print(i,'\tjoint_vel',state_desc["joint_vel"][joint])
        i += len(state_desc["joint_vel"][joint])
        print(i,'\tjoint_acc',state_desc["joint_acc"][joint])
        i += len(state_desc["joint_acc"][joint])

    for muscle in state_desc["muscles"].keys():
        print(muscle)
        print(i,'\tactivation',[state_desc["muscles"][muscle]["activation"]])
        i += 1
        print(i,'\tfiber_length',[state_desc["muscles"][muscle]["fiber_length"]])
        i += 1
        print(i,'\tfiber_velocity',[state_desc["muscles"][muscle]["fiber_velocity"]]) 
        i += 1

    print(i,'center of mass pos',state_desc["misc"]["mass_center_pos"])
    i += len(state_desc["misc"]["mass_center_pos"])
    print(i,'center of mass vel',state_desc["misc"]["mass_center_vel"])
    i += len(state_desc["misc"]["mass_center_vel"])
    print(i,'center of mass vel',state_desc["misc"]["mass_center_acc"])
    i += len(state_desc["misc"]["mass_center_acc"])
    print(i)

    
