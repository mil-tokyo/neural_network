import numpy as np

class Randselect:
    def __init__(self,lst):
        self.probs=[fst for (fst,snd) in lst]
        self.returns=[snd for (fst,snd) in lst]

    def getresult(self):
        return self.returns[np.where(np.random.multinomial(1,self.probs)==1)[0][0]]
        #return np.random.choice(returns,1,p=probs)[0]


class State:
    def __init__(self,name,actiondict,rewarddict):
        self.name=name
        self.actiondict=actiondict
        self.rewarddict=rewarddict

    def nextstate(self,action):
        ne=self.actiondict[action].getresult()
        re=self.rewarddict[(self.name,action,ne)].getresult()
        return (ne,re)

    def addaction(self,aname,action):
        self.actiondict[aname]=action

    def addrewards(self,aname,nname,reward):
        self.rewarddict[(self.name,aname,nname)]=reward

    def actions(self):
        return self.actiondict.keys()


class Env:
    def __init__(self,initname):
        self.states={}
        self.initname=initname

    def addstate(self,sname,state):
        self.states[sname]=state

    def addaction(self,sname,aname,action):
        self.states[sname].addaction(aname,action)

    def addrewards(self,sname,aname,nname,reward):
        self.states[sname].addrewards(aname,nname,reward)

    def movestates(self,sname,action):
        return self.states[sname].nextstate(action)
