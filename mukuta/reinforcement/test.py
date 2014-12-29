import qlearning
import environment

env = environment.Env('start')
env.addstate('start',environment.State('start',{},{}))
env.addaction('start','research',environment.Randselect([(0.7,'result'),(0.3,'start')]))
env.addaction('start','play',environment.Randselect([(1,'start')]))
env.addrewards('start','research','result',environment.Randselect([(1,-10)]))
env.addrewards('start','research','start',environment.Randselect([(1,-10)]))
env.addrewards('start','play','start',environment.Randselect([(1,2)]))

env.addstate('result',environment.State('result',{},{}))
env.addaction('result','research',environment.Randselect([(0.7,'thesis'),(0.3,'result')]))
env.addaction('result','play',environment.Randselect([(0.7,'start'),(0.3,'result')]))
env.addrewards('result','research','thesis',environment.Randselect([(1,-10)]))
env.addrewards('result','research','result',environment.Randselect([(1,-10)]))
env.addrewards('result','play','result',environment.Randselect([(1,2)]))
env.addrewards('result','play','start',environment.Randselect([(1,2)]))

env.addstate('thesis',environment.State('thesis',{},{}))
env.addaction('thesis','research',environment.Randselect([(0.7,'job'),(0.3,'thesis')]))
env.addaction('thesis','play',environment.Randselect([(0.7,'result'),(0.3,'thesis')]))
env.addrewards('thesis','research','job',environment.Randselect([(1,-10)]))
env.addrewards('thesis','research','thesis',environment.Randselect([(1,-10)]))
env.addrewards('thesis','play','thesis',environment.Randselect([(1,2)]))
env.addrewards('thesis','play','result',environment.Randselect([(1,2)]))

env.addstate('job',environment.State('job',{},{}))
env.addaction('job','research',environment.Randselect([(1,'job')]))
env.addaction('job','play',environment.Randselect([(0.7,'thesis'),(0.3,'job')]))
env.addrewards('job','research','job',environment.Randselect([(1,100)]))
env.addrewards('job','play','job',environment.Randselect([(1,2)]))
env.addrewards('job','play','thesis',environment.Randselect([(1,2)]))

q = qlearning.QLearning(env,0.1,0.9,0.3)
