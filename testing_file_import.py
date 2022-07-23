#import
from gym_fetch_RT import RT_run
model = None
model, states, actions, params = RT_run.main(model)
assert len(states) == len(actions)
#sys.path.insert(1, './gym_fetch_RT/baselines')
#gym_fetch_RT.baselines.RT_run.main()
#from baselines import RT_main
#baselines.RT_run.main()