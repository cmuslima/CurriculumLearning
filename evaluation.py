
import numpy as np
import random



# def normalize(value):
#     min = 1
#     max = 25
#     normalize_value = (value - min)/(max- min)
#     return normalize_value

def evaluate_task(student_agent, env, ss, args): #this will evaluate it for one episode 

    score_per_episode = list()
    #cost_per_episode = list()
    num_steps_per_episode = list()

    num_episodes = 20
    eps = 0
    for i in range(num_episodes):
        start_state = ss #target task
        state = student_agent.reset(start_state)
         
        num_time_steps = 0
        while(True):
            action_movement, action_index = student_agent.e_greedy_action_selection(state, env, eps)
            
            next_state, reward, done = student_agent.step(state, action_index, action_movement, env)
            #total_rewards+=reward
            #print('S prime', next_state)
            if done or num_time_steps>=args.max_time_step:
                num_time_steps+=1
                if reward == 1:
                    score = args.discount**num_time_steps*1
                else:
                    score = 0 
                score_per_episode.append(score)
                #cost_per_episode.append(normalize(num_time_steps))
                num_steps_per_episode.append(num_time_steps) 

                break
            state=next_state
            num_time_steps+=1
    assert len(score_per_episode) == num_episodes
    average_score = np.mean(np.array(score_per_episode))
    #average_cost = np.mean(np.array(cost_per_episode))
    average_time_step = np.mean(np.array(num_steps_per_episode))

    return average_score, average_time_step

def train(student_agent, env, episodes, teacher_action, args):
    for i in range(episodes):
        state = student_agent.reset(start_state = teacher_action)
        score = 0
        num_time_steps = 0

        while(True):

            action_movement, action_index = student_agent.act(state, env)

            next_state, reward, done = student_agent.step(state, action_index, action_movement,env)
            score += reward
            if num_time_steps>=args.max_time_step:
                done = True
                    
            student_agent.learning_update(reward, next_state, state, action_index, done)

            if done:
                num_time_steps+=1
                if score == 1:
                    score = args.discount**num_time_steps
                if i == 0: #this is the first training episode
                    first_reward = score
                 
                if i == episodes-1: #this is the last training episode
                    final_reward = score
                break

            state=next_state
            num_time_steps+=1
    if args.debug:
        print(f'first reward {first_reward}, final reward {final_reward}')               
    return first_reward,  final_reward
  