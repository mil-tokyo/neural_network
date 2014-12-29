import collections
import itertools
import environment
import numpy as np

class QLearning:
    def __init__(self,env,alpha0,gamma0,epsilon):
        self.env=env
        self.q=collections.defaultdict(itertools.repeat(1000).next)
        self.alpha0=alpha0
        self.gamma0=gamma0
        self.epsilon=epsilon

    def learn(self,num):
        state=self.env.states[self.env.initname]
        for it in range(1,num+1):
            sname=state.name
            alpha=self.alpha0#/(it**0.75)
            gamma=self.gamma0#/(it**0.75)
            action=self.egreedy(state,self.epsilon)
            (nextname,reward)=self.env.movestates(sname,action)
            nextstate=self.env.states[nextname]
            self.q[(sname,action)]=(1-alpha)*self.q[(sname,action)]+alpha*(reward+gamma*self.maxq(nextstate)[0])
            state=nextstate

    def egreedy(self,state,epsilon):
        actions=state.actions()
        anum=len(actions)
        qs = [(self.q[(state.name,action)],action) for action in actions]
        if(np.random.uniform(0,1,1)<epsilon):
            return actions[np.where(np.random.multinomial(1,[1.0/anum]*anum)==1)[0][0]]
            #return np.random.choice(actions,1)[0]
        else:
            return self.maxq(state)[1]


    def maxq(self,state):
        qs = [(self.q[(state.name,action)],action) for action in state.actions()]
        return max(qs)
