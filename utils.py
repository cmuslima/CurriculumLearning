import numpy as np
#change from 25 to 20
import config
discount = .99
def train(agent, env, episodes, teacher_action):
    for i in range(episodes):
        state = agent.reset(start_state = teacher_action)
        score = 0
        num_time_steps = 0

        while(True):

            action_movement, action_index = agent.act(state, env)

            next_state, reward, done = agent.step(state, action_index, action_movement,env)
            score += reward
            if num_time_steps>=config.max_time_step:
                done = True
                    
            agent.learning_update(reward, next_state, state, action_index, done)

            if done:
                num_time_steps+=1
                if score == 1:
                    score = config.discount**num_time_steps
                if i == 0: #this is the first training episode
                    first_reward = score
                 
                if i == episodes-1: #this is the last training episode
                    final_reward = score
                break

            state=next_state
            num_time_steps+=1
    if config.debug:
        print(f'first reward {first_reward}, final reward {final_reward}')               
    return first_reward,  final_reward