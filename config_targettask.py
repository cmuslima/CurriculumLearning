
import sys
import argparse
import utils
from run_training_loop import run_train_loop
from run_eval_loop import run_evaluation_loop
#from MP_run import MP_run_training_loop
#from gym_fetch_RT import RT_run
# try:
#     print('here')
#     from gym_fetch_RT import RT_run
#     print('got it')
# except:
#     pass
import plotting_graphs_updated
import average
import teacher_data_plot
#from plotting_graphs import plotting, plot_actions
#from average import average_data, get_data
# #from subplot import plotting
# from target_CL_loop import target_loop
#from testing_fetchpush import testing

def main(args):
    print('\n\n\n\n main function called \n\n\n\n')
    utils.make_dir(args, args.rootdir)

    if args.training:
        run_train_loop(args)
                        #run_evaluation_loop(args)

 
    if args.evaluation:
        run_evaluation_loop(args)

    if args.plotting:
        #data = utils.get_data('student_scores_buffer_policy_simple_LP_0.0001_64_50_all_seeds')
        #print(data)
        average.average_data(args)
        plotting_graphs_updated.plot_single_baseline(args)
 

if __name__ == '__main__':
    print('\n\n\n\n\n\n\n main called \n\n\n\n\n\n\n\n')

    parser = argparse.ArgumentParser()

    parser.add_argument('--rootdir', type=str) 
    parser.add_argument('--env', type=str, default= 'four_rooms')
    parser.add_argument('--max_time_step', type=int, default= 39) #or 40
    parser.add_argument('--SR', type=str, default= 'params')
    parser.add_argument('--reward_function', type=str, default = 'simple_LP')
    parser.add_argument('--alpha', type=float, default= 1)
    parser.add_argument('--teacher_evaluation_seed', type=int, default= 30)
    parser.add_argument('--student_evaluation_seed', type=int, default= 0)

    parser.add_argument('--student_transfer', type=bool, default= False)
    parser.add_argument('--student_lr_transfer', type=bool, default= False)
    parser.add_argument('--student_NN_transfer', type=bool, default= False)
    parser.add_argument('--random_student_seed', type=bool, default= True)
    parser.add_argument('--student_discount', type=float, default= .99)
    parser.add_argument('--student_eps', type=float, default= .01)


    parser.add_argument('--teacher_agent', type=str, default= 'DQN')
    parser.add_argument('--teacher_eps_start', type=float, default= .5)
    parser.add_argument('--teacher_eps_decay', type=float, default= .99)
    parser.add_argument('--teacher_eps_end', type=float, default= .01)

    parser.add_argument('--teacher_buffersize', type=int, default= 200)
    parser.add_argument('--teacher_batchsize', type=int, default= 256)
    parser.add_argument('--teacher_lr', type=float, default= .001)
    parser.add_argument('--teacher_episodes', type=int, default= 200) #500


    parser.add_argument('--one_hot_action_vector', type=bool, default = True)
    parser.add_argument('--easy_initialization', type=bool, default = True)
    parser.add_argument('--reward_log', type=bool, default= True)
    parser.add_argument('--normalize', type=bool, default= False)


    parser.add_argument('--student_episodes', type=int, default= 100)
    parser.add_argument('--num_training_episodes', type=int, default=10)
    parser.add_argument('--num_evaluation_episodes', type=int, default=40)
    parser.add_argument('--student_type', type=str, default = 'q_learning')
    parser.add_argument('--two_buffer', type=bool, default = False)
    parser.add_argument('--multi_controller', type=bool, default = False)
    parser.add_argument('--multi_controller_v2', type=bool, default = False)
    parser.add_argument('--clear_buffer', type=bool, default = False)

    parser.add_argument('--multi_students', type=bool, default = False)
    parser.add_argument('--tabular', type=bool, default = False)
    parser.add_argument('--goal_conditioned', type=bool, default = False)

    parser.add_argument('--stagnation_threshold', type=int, default= 3)
    parser.add_argument('--LP_threshold', type=float, default= .05)
    #parser.add_argument('--percent_change', type=bool, default = False)

    #types of teachers during evaluation:
    parser.add_argument('--random_curriculum', type=bool, default = False)
    parser.add_argument('--target_task_only', type=bool, default = True)
    parser.add_argument('--trained_teacher', type=bool, default = False)
    parser.add_argument('--handcrafted', type=bool, default = False)
    parser.add_argument('--HER', type=bool, default = False)


    parser.add_argument('--debug', type=bool, default= False)
    parser.add_argument('--num_runs_start', type=int, default= 0)
    parser.add_argument('--num_runs_end', type=int, default= 10)
    parser.add_argument('--num_runs', type=int, default= 10)

    parser.add_argument('--num_student_processes', type=int, default= 1)
    parser.add_argument('--MP', type=bool, default = False)
    #baselines:
    parser.add_argument('--Narvekar2018', type=bool, default = False)
    parser.add_argument('--Narvekar2017', type=bool, default = False)
    parser.add_argument('--L2T', type=bool, default = False)




    parser.add_argument('--training', type=bool, default = False)
    parser.add_argument('--evaluation', type=bool, default = True)
    parser.add_argument('--plotting', type=bool, default = False)
    parser.add_argument('--average', type=bool, default = False)
    parser.add_argument('--saving_method', type=str, default = 'exceeds_average')
    parser.add_argument('--folder_name', type=str, default = 'None')

    #network args

    parser.add_argument('--student_input_size', type=int)
    parser.add_argument('--student_output_size', type=int)
    parser.add_argument('--hidden_size_input', type=float)
    parser.add_argument('--hidden_size_output', type=int) 
    parser.add_argument('--teacher_network_hidden_size', type=int) 
    parser.add_argument('--three_layer_network', type=bool) 

    #Training params for PPO student
    parser.add_argument("--algo", type = str, default= 'ppo',
                        help="algorithm to use: a2c | ppo (REQUIRED)")
    # parser.add_argument("--model", default='FourRoomsCL', 
    #                     help="name of the model (default: {ENV}_{ALGO}_{TIME})")
    parser.add_argument("--seed", type=int, default=0,
                        help="random seed (default: 1)")
    parser.add_argument("--log-interval", type=int, default=1,
                        help="number of updates between two logs (default: 1)")
    parser.add_argument("--save_interval", type=int, default=5,
                        help="number of updates between two saves (default: 10, 0 means no saving)")
    parser.add_argument("--procs", type=int, default=1,
                        help="number of processes (default: 16)")
    parser.add_argument("--frames", type=int, default=10**7,
                        help="number of frames of training (default: 1e7)")
    parser.add_argument("--updates", type=int, default=25,
                        help="number of frames of training (default: 1e7)")#was 50
    ## Parameters for main algorithm
    parser.add_argument("--epochs", type=int, default=4,
                        help="number of epochs for PPO (default: 4)")
    parser.add_argument("--student_batch_size", type=int, default=256,
                        help="batch size for PPO (default: 256)")
    parser.add_argument("--frames-per-proc", type=int, default=128,
                        help="number of frames per process before update (default: 5 for A2C and 128 for PPO)")
    parser.add_argument("--discount", type=float, default=0.99,
                        help="discount factor (default: 0.99)")
    parser.add_argument("--student_lr", type=float, default=0.001,
                        help="learning rate (default: 0.001)")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
    parser.add_argument("--entropy-coef", type=float, default=0.01,
                        help="entropy term coefficient (default: 0.01)")
    parser.add_argument("--value-loss-coef", type=float, default=0.5,
                        help="value loss term coefficient (default: 0.5)")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="maximum norm of gradient (default: 0.5)")
    parser.add_argument("--optim-eps", type=float, default=1e-8,
                        help="Adam and RMSprop optimizer epsilon (default: 1e-8)")
    parser.add_argument("--optim-alpha", type=float, default=0.99,
                        help="RMSprop optimizer alpha (default: 0.99)")
    parser.add_argument("--clip-eps", type=float, default=0.2,
                        help="clipping epsilon for PPO (default: 0.2)")
    parser.add_argument("--recurrence", type=int, default=1,
                        help="number of time-steps gradient is backpropagated (default: 1). If > 1, a LSTM is added to the model to have memory.")
    parser.add_argument("--text", action="store_true", default=False,
                        help="add a GRU to the model to handle text input")

    #student Evaluation params

    parser.add_argument("--evaluation_seed", type=int, default=0,
                        help="random seed (default: 0)")
    parser.add_argument("--argmax", action="store_true", default=False,
                        help="action with highest probability is selected")


    # open AI baseline students

    parser.add_argument('--alg', help='Algorithm', type=str, default='her')
    parser.add_argument('--total_timesteps', type=float, default=10000), #100050
    parser.add_argument('--network', help='network type (mlp, cnn, lstm, cnn_lstm, conv_only)', default='mlp')
    parser.add_argument('--gamestate', help='game state to load (so far only used in retro games)', default=None)
    parser.add_argument('--num_env', help='Number of environment copies being run in parallel. When not specified, set to number of cpus for Atari, and to 1 for Mujoco', default=2, type=int)
    parser.add_argument('--reward_scale', help='Reward scale factor. Default: 1.0', default=1.0, type=float)
    parser.add_argument('--save_path', help='Path to save trained model to', default=None, type=str)
    parser.add_argument('--save_video_interval', help='Save video every x steps (0 = disabled)', default=0, type=int)
    parser.add_argument('--save_video_length', help='Length of recorded video. Default: 200', default=200, type=int)
    parser.add_argument('--log_path', help='Directory to save learning curve data.', default=None, type=str)
    parser.add_argument('--play', default=False, action='store_true')
    parser.add_argument('--target_task', type=str, default='FetchPush-v1')


    #plotting arguments
    parser.add_argument('--single_baseline_comp', type=bool, default=True)
    parser.add_argument('--comparing_scores', type=bool, default=True)
    parser.add_argument('--plot_best_data', type=bool, default=True)
    parser.add_argument('--stagnation', type=bool, default=True)



    args = parser.parse_args()
    if args.env == 'maze': #once I update the env code such that I only have one code for all environments, this will be more useful
        args.rows = 11
        args.columns = 16
        args.student_num_actions = 4
        args.max_time_step = 39
        args.num_evaluation_episodes = 30
        args.num_training_episodes = 10
        args.student_lr = .5
        args.student_input_size = 2
        args.student_output_size = 4
        args.hidden_size_input = 32
        args.hidden_size_output = 32
        args.teacher_network_hidden_size = 64
        args.tabular = True
        args.teacher_episodes = 300
        args.student_episodes = 100*150
        args.three_layer_network = False
        args.num_tiles = 8
        args.stagnation = True
        if args.Narvekar2017:
            args.reward_function ='Narvekar2017'
            args.SR = 'params'
            args.num_training_episodes = 500
            args.student_episodes = 100

        if args.Narvekar2018:
            args.reward_function ='Narvekar2018'
            args.SR = 'tile_coded_params'
            args.num_training_episodes = 500
            args.student_episodes = 100

        if args.L2T:
            args.reward_function ='L2T'
            args.SR = 'L2T'       
    elif args.env == 'expanded_fourrooms': #once I update the env code such that I only have one code for all environments, this will be more useful
        args.rows = 7
        args.columns = 10
        args.student_num_actions = 4
        args.max_time_step = 20
        args.num_evaluation_episodes = 30
        args.num_training_episodes = 2
        args.student_lr = .5
        args.student_input_size = 2
        args.student_output_size = 4
        args.hidden_size_input = 32
        args.hidden_size_output = 32
        args.teacher_network_hidden_size = 64
        args.tabular = True
        args.teacher_episodes = 300
        args.student_episodes = 50
        args.three_layer_network = False
        args.num_tiles = 8
        args.stagnation = True
      
    elif args.env == 'cliff_world': #once I update the env code such that I only have one code for all environments, this will be more useful
        args.rows = 4
        args.columns = 12
        args.student_num_actions = 4
        args.max_time_step = 1000
        args.num_evaluation_episodes = 30
        args.num_training_episodes = 5
        args.student_lr = .5
        args.student_eps = .1
        args.student_input_size = 2
        args.student_output_size = 4
        args.hidden_size_input = 32
        args.hidden_size_output = 32
        args.teacher_network_hidden_size = 64
        args.tabular = True
        args.teacher_episodes = 400
        args.student_episodes = 150
        args.three_layer_network = False
        args.normalize = True
        print('args.env', args.env)
    elif args.env == 'combination_lock': #once I update the env code such that I only have one code for all environments, this will be more useful
        args.rows = 1
        args.columns = 23
        args.student_num_actions = 2
        args.max_time_step = 300 #it was 100 for 20,30, and 50.. now for 100 we're gonna up it to 200
        args.num_evaluation_episodes = 30
        args.num_training_episodes = 5
        args.student_lr = .5
        args.student_eps = .01
        args.student_input_size = 2
        args.student_output_size = 2
        args.hidden_size_input = 32
        args.hidden_size_output = 32
        args.teacher_network_hidden_size = 64
        args.tabular = True
        args.teacher_episodes = 200
        args.student_episodes = 100
        args.three_layer_network = False
        args.normalize = False
    elif args.env == 'four_rooms':

        args.tabular = False
        args.student_input_size = 243
        args.student_output_size = 3
        args.hidden_size_input = 64
        args.hidden_size_output = 64
        args.teacher_network_hidden_size = 128
        args.student_num_actions = 3
        args.max_time_step = 40
        args.num_evaluation_episodes = 40
        args.num_training_episodes = 25
        args.teacher_episodes = 100
        args.student_episodes = 50*20
        args.student_type = 'PPO'
        args.saving_method = 'exceeds_average'
        args.memory = args.recurrence > 1
        args.target_task = f'MiniGrid-Simple-4rooms-0-v0'
        #utils.make_dir(args, f'{args.rootdir}/models')
        args.model_folder_path = f'{args.SR}_{args.teacher_batchsize}_{args.teacher_lr}_{args.reward_function}_{args.teacher_buffersize}_{args.num_runs_start}'
        args.three_layer_network = False
        
        
        if args.L2T:
            args.reward_function ='L2T'
            args.SR = 'L2T'  
    print(args.student_type)
    if args.env == 'fetch_reach_2D' or args.env == 'fetch_reach_2D_outer' or args.env == 'fetch_reach_3D' or args.env == 'fetch_reach_3D_outer': #once I update the env code such that I only have one code for all environments, this will be more useful
        args.num_evaluation_episodes = 80
        args.student_input_size = 10
        args.student_output_size = 4
        args.hidden_size_input = 64
        args.hidden_size_output = 64
        args.teacher_network_hidden_size = 128
        args.teacher_episodes = 50
        args.student_episodes = 50 #change this back
        args.saving_method = 'exceeds_average'
        args.student_type = 'DDPG'
        args.three_layer_network = False
        args.tabular = False
        args.num_env = 1

    if args.env == 'fetch_push':
        args.num_evaluation_episodes = 80
        args.student_input_size = 25
        args.student_output_size = 4
        args.hidden_size_input = 64
        args.hidden_size_output = 64
        args.teacher_network_hidden_size = 128
        args.teacher_episodes = 75
        args.student_episodes = 50
        args.saving_method = 'exceeds_average'
        args.student_type = 'DDPG'
        args.three_layer_network = False
        args.num_env = 2



    if args.training:

        learning_rates = [.001] #need to do .001, 100, all batch sizes.  then .005 with everything for cost 
        buffer_sizes = [200]
        batch_sizes = [128] 
        SR = ['buffer_action'] #'L2T' #'buffer_policy', 'buffer_q_table', 'params' need to do these with L2T
        rf = ['0_target_task_score', '0_target_task_score', 'cost', 'L2T', 'simple_LP', 'LP'] #'cost', 'simple_LP'
        for state_rep in SR:
            # if 'buffer' not in state_rep:
            #     buffer_sizes = [1] #this is because we don't need to do any loops over buffer size for the not behavior embedded state reps
            for reward in rf:
                for lr in learning_rates:
                    for buffer in buffer_sizes:
                        for batch in batch_sizes:     
                            assert args.total_timesteps == 10000
                            args.teacher_batchsize = batch
                            args.teacher_lr = lr
                            args.teacher_buffersize = buffer
                            args.SR = state_rep
                            args.reward_function = reward
                            args.model_folder_path = f'{args.SR}_{args.teacher_batchsize}_{args.teacher_lr}_{args.reward_function}_{args.teacher_buffersize}_{args.num_runs_start}'

                            args.rootdir = utils.get_rootdir(args, args.SR)
                            print(args.rootdir)
                            utils.make_dir(args, args.rootdir)
                            print('\n\n\n\n\n config \n\n\n\n\n')
                            print("Running experiment with a single student", args.student_type)
                            print(f'Root dir = {args.rootdir}')
                            if args.MP:
                                MP_run_training_loop(args)
                            else:
                                run_train_loop(args)
 
    if args.evaluation:

        learning_rates = [.001]
        buffer_sizes = [100]
        batch_sizes = [128] 
        SR = ['buffer_q_table'] #'L2T' #'buffer_policy', 'buffer_q_table', 'params' need to do these with L2T
        rf = ['simple_LP'] #'cost', 'simple_LP'
        student_lr = [.5]
        for state_rep in SR:
            # if 'buffer' not in state_rep:
            #     buffer_sizes = [1] #this is because we don't need to do any loops over buffer size for the not behavior embedded state reps
            for reward in rf:
                for lr in learning_rates:
                    for buffer in buffer_sizes:
                        for batch in batch_sizes:     
                            for student_l in student_lr:
                                #args.student_lr = student_l
                                args.teacher_batchsize = batch
                                args.teacher_lr = lr
                                args.teacher_buffersize = buffer
                                args.SR = state_rep
                                args.reward_function = reward
                                args.model_folder_path = f'{args.SR}_{args.teacher_batchsize}_{args.teacher_lr}_{args.reward_function}_{args.teacher_buffersize}_{args.num_runs_start}'

                                args.rootdir = utils.get_rootdir(args, args.SR)
                                utils.make_dir(args, args.rootdir)
                                print('\n\n\n\n\n config \n\n\n\n\n')
                                print("Running experiment with a single student", args.student_type)
                                print(f'Root dir = {args.rootdir}')
                                run_evaluation_loop(args)
 

    if args.plotting:
        learning_rates = [.001]
        buffer_sizes = [75]
        batch_sizes = [128] 
        SR = ['buffer_q_table'] #'L2T' #'buffer_policy', 'buffer_q_table', 'params' need to do these with L2T
        rf = ['simple_LP'] 
        student_lrs = [.5]
        for state_rep in SR:
            # if 'buffer' not in state_rep:
            #     buffer_sizes = [1] #this is because we don't need to do any loops over buffer size for the not behavior embedded state reps
            for reward in rf:
                for lr in learning_rates:
                    for buffer in buffer_sizes:
                        for batch in batch_sizes:     
                            for student_lr in student_lrs:
                                args.student_lr = student_lr  
                                args.teacher_batchsize = batch
                                args.teacher_lr = lr
                                args.teacher_buffersize = buffer
                                args.SR = state_rep
                                args.reward_function = reward
                                args.model_folder_path = f'{args.SR}_{args.teacher_batchsize}_{args.teacher_lr}_{args.reward_function}_{args.teacher_buffersize}_{args.num_runs_start}'
                                
                                args.rootdir = utils.get_rootdir(args, args.SR)
                               # average.average_data(args)
                            #plotting_graphs_updated.determine_normal_dis(args)
        #plotting_graphs_updated.plot_actions(args)
        #average.quick_plot(args)
        plotting_graphs_updated.plot_single_baseline(args)
        #teacher_data_plot.plot_single_baseline(args)
        #plotting_graphs_updated.p_testing_area_under_curve(args)
        #average.average_data(args)
        # print(area_under_curve)
       #plotting_graphs_updated.t_testing(args)
        #plotting_graphs_updated.plot_actions(args)
        # import utils
        # import numpy as np
        # area_under_curve_list = dict()
        # learning_rates = [.001]
        # buffer_sizes = [200]
        # batch_sizes = [128] 
        # #SR = ['buffer_action'] #'L2T' #'buffer_policy', 'buffer_q_table', 'params' need to do these with L2T
        # #rf = ['simple_LP', 'L2T', 'cost'] #'cost', 'simple_LP'
        # SR = ['buffer_action', 'params', 'L2T'] #, 'params', 'tile_coded_params', 'buffer_action'
        # rf = ['simple_LP']
        # for state_rep in SR:
        #     # if 'buffer' not in state_rep:
        #     #     buffer_sizes = [1] #this is because we don't need to do any loops over buffer size for the not behavior embedded state reps
        #     for reward in rf:
        #         for lr in learning_rates:
        #             for buffer in buffer_sizes:
        #                 for batch in batch_sizes:     
        #                     assert args.total_timesteps == 10001  
        #                     args.teacher_batchsize = batch
        #                     args.teacher_lr = lr
        #                     args.teacher_buffersize = buffer
        #                     args.SR = state_rep
        #                     args.reward_function = reward
        #                     args.model_folder_path = f'{args.SR}_{args.teacher_batchsize}_{args.teacher_lr}_{args.reward_function}_{args.teacher_buffersize}_{args.num_runs_start}'
                            
        #                     args.rootdir = utils.get_rootdir(args, args.SR)
        #                     if 'buffer' in state_rep:
        #                         data = utils.get_data(f'{args.rootdir}/evaluation-data/raw_averaged_returns_{state_rep}_{reward}_{lr}_{args.teacher_batchsize}_{buffer}_')
        #                         #data = get_data(f'{args.rootdir}/teacher-data/teacher_return_list_{SR}_{reward}_{lr}_{batchsize}_{buffer}_{i}')
        #                         mean = data[0]
        #                         std = data[1]
        #                         averaged_std = np.mean(std)
        #                         AOC = plotting_graphs_updated.calculate_area_under_curve(mean)
        #                     else:
        #                         #print(f'{args.rootdir}/evaluation-data/student_score_{SR}_{reward}_{lr}_{batchsize}_{i}')
                                
        #                         data = utils.get_data(f'{args.rootdir}/evaluation-data/raw_averaged_returns_{state_rep}_{reward}_{lr}_{args.teacher_batchsize}_')
        #                         mean = data[0]
        #                         std = data[1]
        #                         averaged_std = np.mean(std)
        #                         AOC = plotting_graphs_updated.calculate_area_under_curve(mean)

                            
        #                     name = f'{state_rep} + {reward}'
        #                     area_under_curve_list[name] = [AOC, averaged_std]
        

        # print(area_under_curve_list)
