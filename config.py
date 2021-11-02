import sys
import argparse
from run_teacher_training_loop import run
from evaluate_teacher import run_evaluate_teacher
parser = argparse.ArgumentParser()

#Put the env name as the roordir
parser.add_argument('--rootdir', type=str, default= './four_rooms2')
#put the state rep as the experiment folder
parser.add_argument('--experiment_folder', type=str, default= 'action_return')
parser.add_argument('--env', type=str, default= 'four_rooms')
parser.add_argument('--SR', type=str, default= 'action_return')
parser.add_argument('--optimal_target_threshold', type=bool, default= True)
parser.add_argument('--reward_log', type=bool, default= True)
parser.add_argument('--teacher_episodes', type=int, default= 15) #100
parser.add_argument('--student_episodes', type=int, default= 3)
parser.add_argument('--discount', type=float, default= .99)
parser.add_argument('--num_training_episodes', type=int, default= 1)
parser.add_argument('--debug', type=bool, default= False)
parser.add_argument('--runs', type=int, default= 1)
parser.add_argument('--lr', type=float, default= .001)
parser.add_argument('--alpha', type=float, default= 1.25)
parser.add_argument('--batchsize', type=int, default= 64)
parser.add_argument('--max_time_step', type=int, default= 24) #or 40
parser.add_argument('--random_curriculum', type=bool, default = False)

parser.add_argument('--random_curr', type=bool, default = False)
parser.add_argument('--target_only', type=bool, default = False)
parser.add_argument('--reward_function', type=str, default = 'cost')
parser.add_argument('--one_hot_action_vector', type=bool, default = True)
parser.add_argument('--easy_initialization', type=bool, default = True)
parser.add_argument('--training', type=bool, default = True)
parser.add_argument('--evaluation', type=bool, default = True)


args = parser.parse_args()

if args.training:
    run(args)

if args.evaluation:
    run_evaluate_teacher(args)

