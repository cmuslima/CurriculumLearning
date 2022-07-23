
import numpy as np
import utils

class train_evaluate_protocol():
    def __init__(self, env, args):
        self.env = env

        if args.env == 'fetch_push':
            self.run, self.build_env = utils.import_modules(args)
        if args.env == 'fetch_reach_3D_outer':
            self.RT_run  = utils.import_modules(args)
        if args.env == 'four_rooms' and args.tabular == False:
            print('loading four rooms modules')
            self.student_train, self.evaluate_task, self.visualize = utils.import_modules(args)
    def train(self, student_agent, student_type, task_name, args, model, env_dict, config_params, dims):

        if args.tabular:
            params = None
            model = None
            if args.Narvekar2017 or args.Narvekar2018:
                score = self.train_tabular_Narvekar(student_agent, task_name, args, student_type)
                obss, q_values, actions = None, None, None
            else:
        
                score, obss, q_values, actions = self.train_tabular(student_agent, task_name, args, student_type)
 
        if args.student_type == "PPO":
            params = None
            model = None
            score, obss, q_values, actions = self.student_train(args, task_name)
            #print(np.shape(obss), np.shape(actions), np.shape(q_values))

        if args.student_type == 'DDPG' and args.env != 'fetch_push':
            print('in this section')
            print('task name', task_name)
            model, obss, actions, params, config_params, dims = self.RT_run.student_training(model, task_name, args, env_dict, False, config_params, dims)
            q_values = None
            score = None

        if args.student_type == 'DDPG' and args.env == 'fetch_push':
            print(f'Training DDPG on fetch push')
            training_complete = False
            model, obss, actions, params, config_params, dims = self.run.student_training(model, config_params, dims, training_complete, env_dict, task_name, args)
            #model, obss, actions, params, config_params, dims = RT_run.student_training(model, task_name, args, env_dict, False, config_params, dims)
            q_values = None
            score = None
        return score, obss, q_values, actions, model, params, config_params, dims 
    def visualize(self, args):

 
        self.visualize(args)
        
        #print('average_score', average_score)
        return 
    def evaluate(self, student_agent, task_name, args, model, env_dict, config_params, dims):

        if args.tabular:
            average_score, _, params = self.evaluate_task_tabular(student_agent, task_name, args)
        if args.student_type == 'DDPG' and args.env != 'fetch_push':
            average_score, _, _  = self.RT_run.student_evalution(args, model, task_name, env_dict, config_params, dims)
            params = None
          
        if args.student_type == 'DDPG' and args.env == 'fetch_push':
            average_score = self.run.evaluation(model, task_name, env_dict, config_params, dims, args)
            params = None        
        if args.student_type == 'PPO':
            #print('here2')
            print('using a PPO student')
            average_score, params = self.evaluate_task(args, task_name)
        
        #print('average_score', average_score)
        return average_score, params

    def source_target_evaluation(self, student_id,student_agent,task_name, target_task, args,  model, env_dict, config_params, dims):
        #print('STUDENT ID', student_id)
        source_task_score, _= self.evaluate(student_agent, task_name, args, model, env_dict, config_params, dims)
        target_task_score, params = self.evaluate(student_agent,target_task,args,  model, env_dict, config_params, dims)
        return source_task_score, target_task_score,params


    #def evaluate_task_FA:
        # if args.student == 'PPO':
        #     do X
        # else:
        #     do Y
        #return average_score, params
    def evaluate_task_tabular(self, student_agent, task_name, args): #this will evaluate it for one episode 
        #print('new evaluation')
        score_per_episode = list()
        num_steps_per_episode = list()

       
        eps = 0

        #print('ss', ss)
        
        for i in range(args.num_evaluation_episodes):
            start_state = task_name #target task_name
            #print('start state', start_state)
            state = student_agent.reset(start_state)
            #print('state', state)
            self.env.reset(start_state)
            #print('termination state', self.env.termination_state)
            action_movement, action_index = student_agent.e_greedy_action_selection(state, self.env, eps)
            num_time_steps = 0
            score=0
            #print('first state', state)
            while(True):
                #print(student_agent.q_matrix[tuple(state)])
                next_state, reward, done = student_agent.step(state, action_index, action_movement, self.env)
                score+=reward
                #print(f'S = {state} A = {action_index} R = {reward} S prime = {next_state}')
                #print(student_agent.q_matrix[(4,13)])
                if done or num_time_steps>=args.max_time_step:
                    num_time_steps+=1
                    if args.env == 'maze' or args.env == 'four_rooms' or args.env == 'expanded_fourrooms' or args.env =='combination_lock': 
                        #print('imm here bitches')
                        if score == 1:
                            score = student_agent.discount**num_time_steps*1
                            
                        else:
                            score = 0 
                        
                        #print('score', score)

                    score_per_episode.append(score)

                    num_steps_per_episode.append(num_time_steps) 

                    break
                state=next_state
                action_movement, action_index = student_agent.e_greedy_action_selection(state, self.env, eps)
                num_time_steps+=1

        assert len(score_per_episode) == args.num_evaluation_episodes
        average_score = np.mean(np.array(score_per_episode))
        average_time_step = np.mean(np.array(num_steps_per_episode))
        #print(average_time_step)
        if args.debug:
            
            comparison = ss == np.array([3,0])
            if comparison.all():
                print('in eval')
                for key in list(student_agent.q_matrix.keys()):
                    print(f'{key} with q value {student_agent.q_matrix[key]} with max action {np.argmax(student_agent.q_matrix[key])}')
        
        #print('buffer at the end of an evaluation', student_agent.state_buffer)
        #q_values = utils.get_q_values(student_agent.state_buffer, student_agent.action_buffer, student_agent)
        #print(f'average_score for task {start_state}= {average_score}')
        return average_score, average_time_step, list(student_agent.q_matrix) #student_agent.state_buffer, q_values, student_agent.action_buffer

    #def train_FA():
        # if args.student == 'PPO':
        #     do X
        # else:
        #     do Y
        #return source_task_score, obss, q_values, actions


    def train_tabular_Narvekar(self, student_agent, task_name, args, student_type):
        #print(student_agent.q_matrix)
        print('stagnation', args.stagnation)
        total_time_steps = 0
        all_scores = []
        for i in range(args.num_training_episodes):
        
            state = student_agent.reset(start_state = task_name)
            self.env.reset(task_name)
            action_movement, action_index = student_agent.act(state, self.env)
            #print(f'first state = {state} first action = {action_index}')
            score = 0
            num_time_steps = 0
            while(True):

        
            
                next_state, reward, done = student_agent.step(state, action_index, action_movement,self.env)
                score += reward
                

                if num_time_steps>=args.max_time_step:
                    #print(num_time_steps)
                    done = True
                        
                next_action_movement, next_action_index = student_agent.act(next_state, self.env)
                #print(f' state = {next_state} action = {next_action_index}')
                if student_type == 'sarsa':
                    #print(f'making sarsa update')
                    if args.student_transfer:
                        student_agent.q_learning_update(state, action_index, reward, next_state, done)
                    else:
                        student_agent.sarsa_update(state, action_index, reward, next_state, next_action_index, done)
                
                if student_type == 'q_learning':
                    #print(f'making q learning update')
                    if args.student_transfer:
                        student_agent.sarsa_update(state, action_index, reward, next_state, next_action_index, done)
                    else:
                        student_agent.q_learning_update(state, action_index, reward, next_state, done)
                if done:
                    num_time_steps+=1
                    total_time_steps+=num_time_steps
                    #print('score', score)
                    if args.env == 'maze' or args.env == 'four_rooms' or args.env =='expanded_fourrooms' or args.env =='combination_lock': 
                        if score == 1:
                            all_scores.append(score)
                            score = student_agent.discount**num_time_steps
                            #print('score', num_time_steps)
                            #print(task_name)                        
                    break
                
                action_index = next_action_index
                action_movement = next_action_movement
                state=next_state
                num_time_steps+=1

            if args.stagnation:
                print('should be here')
                count = 0
                if len(all_scores)>=10:
                    
                    for s in all_scores[-10:]:
                        if s == 1:
                            count+=1
                if count >=10:
                    print('all scores', all_scores)
                    break
            else:
                
            # if score == 1:
            #     print('student has solved ')
            #     break
                if score > 0 :
                    print('score', score)
                    print(f'stopping at student episode {i}')
                    print(f'total cost = {total_time_steps}')

                    break
        
        print(f'stopping at student episode {i}')
        print(f'total cost = {total_time_steps}')
        print(f'score = {score}')
        # print(f'success for a total of {count} rounds')

        return total_time_steps
    def train_tabular(self, student_agent, task_name, args, student_type):
        #print('inside train')
        #print(student_agent.q_matrix)
        student_agent.clear_buffer()
        #print('student agent buffer', student_agent.state_buffer, student_agent.action_buffer)
        average_score = []
        for i in range(args.num_training_episodes):
        
            state = student_agent.reset(start_state = task_name)
            ##print('state', state)
            self.env.reset(task_name)
            #print('start state', self.env.start_state)
            #print('goal state', self.env.termination_state)
            action_movement, action_index = student_agent.act(state, self.env)
            #print(f'first state = {state} first action = {action_index}')
            score = 0
            num_time_steps = 0
            while(True):

        
            
                next_state, reward, done = student_agent.step(state, action_index, action_movement,self.env)
                score += reward
                

                if num_time_steps>=args.max_time_step:
                    #print(num_time_steps)
                    done = True
                        
                next_action_movement, next_action_index = student_agent.act(next_state, self.env)
                #print(f' state = {next_state} action = {next_action_index}')
                if student_type == 'sarsa':
                    #print(f'making sarsa update')
                    if args.student_transfer:
                        student_agent.q_learning_update(state, action_index, reward, next_state, done)
                    else:
                        student_agent.sarsa_update(state, action_index, reward, next_state, next_action_index, done)
                
                if student_type == 'q_learning':
                    #print(f'making q learning update')
                    if args.student_transfer:
                        student_agent.sarsa_update(state, action_index, reward, next_state, next_action_index, done)
                    else:
                        #print('updating with', state, action_index, reward, next_state)
                        student_agent.q_learning_update(state, action_index, reward, next_state, done)
                if done:
                    num_time_steps+=1
                    average_score.append(num_time_steps)
                    #print('successs', score)
                    if args.env == 'maze' or args.env == 'four_rooms' or args.env =='expanded_fourrooms' or args.env =='combination_lock': 
                        if score == 1:
                            score = student_agent.discount**num_time_steps
                            #print('score', num_time_steps)
                            #print(task_name)                        
                    break
                
                action_index = next_action_index
                action_movement = next_action_movement
                state=next_state
                num_time_steps+=1


        
        # if args.debug:
        #     print(f'first reward {first_reward}, final reward {final_reward}')      
        #     print("Num steps on last round ", num_time_steps) 
        # 
        assert len(student_agent.state_buffer) == len(student_agent.action_buffer)
        #print(student_agent.state_buffer[0])
        q_values = utils.get_q_values(student_agent.state_buffer, student_agent.action_buffer, student_agent)
        #print(student_agent.q_matrix)
        #print(q_values)
        # for i in list(student_agent.q_matrix.keys()):
        #     print(student_agent.q_matrix[i], i)
        #print('buffer at the end of an training', student_agent.state_buffer)
        average_score = np.mean(average_score)
        return score, student_agent.state_buffer, q_values, student_agent.action_buffer  # need to get these

    def train_n_step_tabular(self, student_agent, task_name, args):
        student_agent.clear_buffer()
        for i in range(args.num_training_episodes):
            state = student_agent.reset(start_state = task_name)
            action_movement, action_index = student_agent.act(state, self.env)

            actions = [action_index]
            states = [states]
            rewards = [0]

            score = 0
            num_time_steps = 0
            T = np.inf
            while(True):

                if num_time_steps < T:
            
                    next_state, reward, done = student_agent.step(state, action_index, action_movement,self.env)
                    score += reward
 
                    states.append(next_state)
                    rewards.append(reward)                   

                    if num_time_steps>=args.max_time_step:
                        #print(num_time_steps)
                        done = True


                    if done:
                        T = num_time_steps+1
                    else:
                        next_action_movement, next_action_index = student_agent.act(next_state, self.env)
                        actions.append(next_action_index) 
                    
                # state tau being updated
                tau = t - args.n_step + 1
                if tau >= 0:
                    G = 0
                    for j in range(tau + 1, min(tau + args.n_step + 1, T + 1)):
                        G += np.power(student_agent.discount, j - tau - 1) * rewards[j]
                    
                    #print(f' state = {next_state} action = {next_action_index}')
                    if student_type == 'sarsa':
                        #print(f'making sarsa update')
                        if args.student_transfer:
                            student_agent.q_learning_update(state, action_index, reward, next_state, done)
                        else:
                            student_agent.sarsa_update(state, action_index, reward, next_state, next_action_index, done)
                    
                    if student_type == 'q_learning':
                        #print(f'making q learning update')
                        if args.student_transfer:
                            student_agent.sarsa_update(state, action_index, reward, next_state, next_action_index, done)
                        else:
                            student_agent.q_learning_update(state, action_index, reward, next_state, done)
                    if done:
                        num_time_steps+=1
                        if args.env == 'maze' or args.env == 'four_rooms' or args.env =='expanded_fourrooms' or args.env =='combination_lock': 
                            if score == 1:
                                score = student_agent.discount**num_time_steps
                                #print('score', num_time_steps)
                                #print(task_name)                        
                        break
                    
                    action_index = next_action_index
                    action_movement = next_action_movement
                    state=next_state
                    num_time_steps+=1


        
        # if args.debug:
        #     print(f'first reward {first_reward}, final reward {final_reward}')      
        #     print("Num steps on last round ", num_time_steps) 
        # 
        assert len(student_agent.state_buffer) == len(student_agent.action_buffer)
        #print(student_agent.state_buffer[0])
        q_values = utils.get_q_values(student_agent.state_buffer, student_agent.action_buffer, student_agent)
        #print(student_agent.q_matrix)
        #print(q_values)
        # for i in list(student_agent.q_matrix.keys()):
        #     print(student_agent.q_matrix[i], i)
        #print('buffer at the end of an training', student_agent.state_buffer)
        return score, student_agent.state_buffer, q_values, student_agent.action_buffer # need to get these