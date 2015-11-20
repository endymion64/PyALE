# -*- coding: utf-8 -*-
"""
@author: yocoppen
"""

from rlglue.agent import AgentLoader as AgentLoader



import argparse

from util.ALEFeatures import BasicALEFeatures,RAMALEFeatures
from agents.ALEAgent import ALEAgent


import numpy as np

class ALEERSarsaAgent(ALEAgent):
    
    def __init__(self,alpha=0.1,lambda_=0.9,gamma=.999,eps=0.05,
                 agent_id=0,save_path='.',actions=None, db_size=1000,
                 trajectory_length=1, replays=1):
        #use full images and color mode
        super(ALEERSarsaAgent,self).__init__(actions,agent_id,save_path)
        
        self.eps = eps
        self.name='ER-SARSA'
        self.alpha0 = alpha
        self.alpha = alpha
        self.lambda_ = lambda_
        self.gamma = gamma
        self.db_size = db_size
        self.trajectory_length = trajectory_length
        self.replays = replays

    def agent_start(self,observation):
        super(ALEERSarsaAgent,self).agent_start(observation)
        #reset trace
        self.trace = np.zeros_like(self.theta)
        #action selection
        phi = self.get_phi(observation)
        vals = self.get_all_values(phi,self.sparse)
        a = self.select_action(vals)
        #store state and action
        self.phi = phi
        self.a = a
        return self.create_action(self.actions[a])
        
    
    def agent_init(self,taskSpec):
        super(ALEERSarsaAgent,self).agent_init(taskSpec)
        self.state_projector =  self.create_projector()
        self.theta = np.zeros((self.state_projector.num_features(),
                                self.num_actions()))
        self.sparse = True
        # Sample database
        self.k = 0
        self.l = 1 
        self.db = np.zeros((self.sample_size,4))
      
    #these methods determine how atari observations are processed
    def create_projector(self):
        raise NotImplementedError()
        
    def get_data(self,obs):
        raise NotImplementedError()

    #convert observation to feature vector
    def get_phi(self,obs):
        im = self.get_data(obs)
        if self.sparse:
            #returns only active tiles indices
            return self.state_projector.phi_idx(im)
        else:
            #returns full binary features vector
            return self.state_projector.phi(im)

        
    def get_value(self,phi,a,sparse=False):
        if sparse:
            return np.sum(self.theta[phi,a])
        else:
            return np.dot(phi,self.theta[:,a])
            
    def get_all_values(self,phi,sparse=False):
        if sparse:
            return np.sum(self.theta[phi,:],axis=0)
        else:
            return np.dot(phi,self.theta)
            
    def select_action(self,values): 
        #egreedy
        acts = np.arange(values.size)
        if np.random.rand()< self.eps:
            return np.random.choice(acts)
        else:
            #randomly select action with maximum value
            max_acts = acts[(values==np.max(values))]
            #print np.max(values)
            return np.random.choice(max_acts)
            
    def update_trace(self,phi,a):
        self.trace *= self.gamma*self.lambda_
        if self.sparse:
            #phi consists of nonzero feature indices
            self.trace[self.phi,a] += 1.
        else:
            #phi is full vector of feature values
            self.trace[:,a] += self.phi
        
       # self.trace = np.clip(self.trace,0.,5.)

    def store_sample(self, phi, a, phi_ns, reward):
        i = self.k % self.db_size
        self.db[i,:] = np.array([phi, a, phi_ns, reward])
        return
        
        
    def step(self,reward,phi_ns = None):
        a_ns = None
        if not (phi_ns is None):
            ns_values = self.get_all_values(phi_ns,self.sparse)
            a_ns = self.select_action(ns_values)

        n_rew = self.normalize_reward(reward)
        self.store_sample(self.phi,self.a,phi_ns,n_rew)
        self.k += 1
        return a_ns  #a_ns is action index (not action value)

    def agent_step(self,reward, observation):
        super(ALEERSarsaAgent,self).agent_step(reward, observation)
        phi_ns = self.get_phi(observation)
        a_ns = self.step(reward,phi_ns)
        #log state data
        self.phi = phi_ns
        self.a = a_ns 
        
        return self.create_action(self.actions[a_ns])#create RLGLUE action
        
    def learn_samples(self):
        K = self.trajectory_length*self.l*self.replays
        samples_in_db = None
        if self.k < self.db_size:
            samples_in_db = self.k
        else:
            samples_in_db = self.db_size
        for i in np.arange(K):                 
            ind = np.random.randint(samples_in_db)
            sample = self.db[ind,:]
            phi = sample[0]
            a = sample[1]
            phi_ns = sample[2]
            n_rew = sample[3]
            # Perform SARSA update
            self.update_trace(phi,a)
            delta = n_rew - self.get_value(phi,a,self.sparse)
            if not (phi_ns is None):
                ns_values = self.get_all_values(phi_ns,self.sparse)
                a_ns = self.select_action(ns_values)
                delta += self.gamma*ns_values[a_ns]
            #normalize alpha with nr of active features
            alpha = self.alpha / float(np.sum(self.phi!=0.))
            self.theta+= alpha*delta*self.trace
        self.l += 1

             
    def agent_end(self,reward):
        super(ALEERSarsaAgent,self).agent_end(reward)
        self.step(reward)
        self.learn_samples()
        
        
class BasicALEERSarsaAgent(ALEERSarsaAgent):
    def __init__(self,bg_file='../data/space_invaders/background.pkl',**kwargs):
        super(BasicALEERSarsaAgent,self).__init__(**kwargs)
        self.background = bg_file
        
    def create_projector(self):
        return BasicALEFeatures(num_tiles=np.array([14,16]),
            background_file =  self.background,secam=True )
 
    def get_data(self,obs):
        return self.get_frame_data(obs)

    
class RAMALEERSarsaAgent(ALEERSarsaAgent):
    def create_projector(self):
        return RAMALEFeatures()
        
    def get_data(self,obs):
        return self.get_ram_data(obs)

        
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='run ER-Sarsa Agent')
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
    parser.add_argument('--db_size', metavar='D', type=int, default=1000,
                    help='sample database size')
    parser.add_argument('--trajectory_length', metavar='T', type=int, default=1,
                    help='trajectory length')
    parser.add_argument('--replays', metavar='R', type=int, default=2,
                    help='replay factor')
    parser.add_argument('--savepath', metavar='P', type=str, default='.',
                    help='save path')  
    parser.add_argument('--features', metavar='F', type=str, default='BASIC',
                    help='features to use: RAM or BASIC')
                    
    parser.add_argument('--actions', metavar='C',type=int, default=None, 
                        nargs='*',help='list of allowed actions')

    args = parser.parse_args()
    
    act = None
    if not (args.actions is None):
        act = np.array(args.actions)

    if args.features == 'RAM':
        AgentLoader.loadAgent(RAMALEERSarsaAgent(agent_id=args.id,
                                     alpha =args.alpha,
                                     lambda_=args.lambda_,
                                     eps =args.eps,
                                     gamma=args.gamma, 
                                     save_path=args.savepath,
                                     actions = act,
                                     db_size= args.db_size,
                                     trajectory_length=args.trajectory_length,
                                     replays=args.replays))
    elif args.features == 'BASIC':
        AgentLoader.loadAgent(BasicALEERSarsaAgent(agent_id=args.id,
                                     alpha =args.alpha,
                                     lambda_=args.lambda_,
                                     eps =args.eps,
                                     gamma=args.gamma, 
                                     save_path=args.savepath,
                                     actions = act,
                                     db_size= args.db_size,
                                     trajectory_length=args.trajectory_length,
                                     replays=args.replays))
    else:
        print 'unknown feature type'