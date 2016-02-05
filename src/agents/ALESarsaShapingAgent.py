# -*- coding: utf-8 -*-
"""

@author: yocoppen
"""

from rlglue.agent import AgentLoader as AgentLoader

import argparse

from agents.ALESarsaAgent import BasicALESarsaAgent
from agents.ALEAgent import ALEAgent

import numpy as np

class ALESarsaShapingAgent(BasicALESarsaAgent):
    def __init__(self,selected_potential='less_enemies',**kwargs):
        #use full images and color mode
        super(ALESarsaShapingAgent,self).__init__(**kwargs)
        self.name='SARSA-Shaping'
        self.selected_potential = selected_potential

    def potential(self, obs):
        # less enemies is better
        # below shield is good
        frame = self.get_data(obs).reshape((210,160))
        if (self.selected_potential == 'less_enemies'):
            return self.less_enemies(frame)
        elif (self.selected_potential == 'lowest_enemy'):
            return self.lowest_enemy(frame)
        else:
            raise NotImplementedError()

    def num_enemies(self, frame):
        return np.sum(frame == 20)//37 #on average each enemy consists of 37 pixels

    def less_enemies(self, frame):
        max_num_enemies = 37 # according to http://spaceinvaders.wikia.com/wiki/Space_Invaders_(Atari_2600)
        num_enemies = self.num_enemies(frame)
        print 'num enemies: ' + str(num_enemies)
        return max_num_enemies - num_enemies

    #check how low enemies have gotten (game over if they get to ground level)
    def lowest_enemy(self,frame):
        return  260 - np.max(np.where(frame==20)[0]) #max:lower screen has higher row index

    def agent_step(self,reward, observation):
        phi_ns = self.get_phi(observation)
        potential = self.potential(observation)
        F = self.gamma * potential - self.prev_obs
        ALEAgent.agent_step(self, F+reward,observation)
        a_ns = self.step(F+reward,phi_ns)
        #log state data
        self.phi = phi_ns
        self.a = a_ns 
        self.prev_obs = potential

        return self.create_action(self.actions[a_ns])#create RLGLUE action

    def agent_start(self,observation):
        action = super(ALESarsaShapingAgent,self).agent_start(observation)
        self.prev_obs = 0
        return action


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='run Sarsa Agent with reward shaping')
    parser.add_argument('--id', metavar='I', type=int, help='agent id',
                        default=0)
    parser.add_argument('--gamma', metavar='G', type=float, default=0.999,
                    help='discount factor')
    parser.add_argument('--alpha', metavar='A', type=float, default=0.5,
                    help='learning rate')
    parser.add_argument('--lambda_', metavar='L', type=float, default=0.9,
                    help='trace decay')
    parser.add_argument('--eps', metavar='E', type=float, default=0.05,
                    help='exploration rate')
    parser.add_argument('--savepath', metavar='P', type=str, default='.',
                    help='save path')  
    parser.add_argument('--potential', metavar='F', type=str, default='less_enemies',
                    help='potentials to use: less_enemies or lowest_enemy')
    parser.add_argument('--actions', metavar='C',type=int, default=None, 
                        nargs='*',help='list of allowed actions')

    args = parser.parse_args()
    
    act = None
    if not (args.actions is None):
        act = np.array(args.actions)

    if args.potential == 'less_enemies':
        AgentLoader.loadAgent(ALESarsaShapingAgent(agent_id=args.id,
            alpha =args.alpha,
            lambda_=args.lambda_,
            eps =args.eps,
            gamma=args.gamma, 
            save_path=args.savepath,
            actions = act,
            selected_potential = args.potential))
    elif args.potential == 'lowest_enemy':
        AgentLoader.loadAgent(ALESarsaShapingAgent(agent_id=args.id,
                                     alpha =args.alpha,
                                     lambda_=args.lambda_,
                                     eps =args.eps,
                                     gamma=args.gamma, 
                                     save_path=args.savepath,
                                     actions = act,
                                     selected_potential = args.potential))
    else:
        print 'unknown potential type'
