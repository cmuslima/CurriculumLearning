import sys
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--rootdir', type=str, default= '/Users/cmuslimani/Projects/Curriculum_MDP/Tabular/debug/four_rooms')
parser.add_argument('--experiment_folder', type=str, default= 'new-reward-function/log-based/optimal/random-initalization/fixed-student-seed/action_return')
parser.add_argument('--env', type=str, default= 'four_rooms')
parser.add_argument('--SR', type=str, default= 'action_return')
parser.add_argument('--optimal_target_threshold', type=bool, default= True)
parser.add_argument('--reward_log', type=bool, default= True)
parser.add_argument('--teacher_episodes', type=int, default= 500) #100
parser.add_argument('--student_episodes', type=int, default= 200)
parser.add_argument('--discount', type=float, default= .99)
parser.add_argument('--num_training_episodes', type=int, default= 1)
parser.add_argument('--debug', type=bool, default= False)
parser.add_argument('--runs', type=int, default= 15)

parser.add_argument('--random_curr', type=bool, default = False)
parser.add_argument('--target_only', type=bool, default = False)
parser.add_argument('--reward_function', type=str, default = 'cost')



args = parser.parse_args()


env = args.env
if env == 'maze':
    max_time_step = 40
else:
    max_time_step = 24
SR = args.SR

teacher_episodes = args.teacher_episodes
student_episodes = args.student_episodes
optimal_target_threshold = args.optimal_target_threshold
num_training_episodes = args.num_training_episodes
runs = args.runs
debug = args.debug
reward_log = args.reward_log
target_only = args.target_only
random_curriculum = args.random_curr
rootdir = args.rootdir
experiment_folder = args.experiment_folder
reward_function = args.reward_function
discount = args.discount
print('Updated Config to:')
print(f'root dir {rootdir}, experiment folder {experiment_folder}')
print(f'Environment {env}, State representation {SR}, number of teacher episodes {teacher_episodes}')
print(f'number of student episodes  {student_episodes}, number runs {runs}')

print('random curr?', random_curriculum)
print('target only', target_only)
print('reward function', reward_function)
print('max time steps for student', max_time_step)
print('num training episodes', num_training_episodes)
print('reward log', reward_log)