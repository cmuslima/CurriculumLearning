
#this is about the student env
from env import lavaworld, four_rooms, expanded_four_rooms, maze
from env2 import basic_grids,cliff_world, combination_lock

def make_env(args):
    if args.tabular:
        student_input_size = args.rows*args.columns
        print('args.env', args.env)
        
        env = basic_grids(args.env, args.columns, args.rows)
        
        if args.env == 'cliff_world':
            print('here')
            env = cliff_world()
        if args.env == 'combination_lock':
            print('here')
            env = combination_lock(args.columns)
        
        student_input_size = 2
        if args.SR == 'params' or args.SR == 'params_student_type':
            student_input_size = args.rows*args.columns

        if args.SR == 'L2T':
            student_input_size =  3

        if args.SR == 'loss_mismatch':
            student_input_size =  5
        if args.SR == 'tile_coded_params':
            student_input_size = args.rows*args.columns*(args.rows*args.columns)

        return env, student_input_size
    else:
        if args.student_type == 'PPO':
            student_input_size = 243
            if args.SR == 'L2T':
                student_input_size =  3
            if args.SR == 'params':
                student_input_size = 43700

            if args.SR == 'loss_mismatch':
                student_input_size =  5

        elif args.student_type == 'DDPG' and args.env == 'fetch_reach_2D_outer':
            student_input_size = 10
            if args.SR == 'params':
                student_input_size = 545290
            if args.SR == 'L2T':
                student_input_size =  3
            if args.SR == 'loss_mismatch':
                student_input_size =  5

        elif args.student_type == 'DDPG' and args.env == 'fetch_reach_3D_outer':
            student_input_size = 25
            if args.SR == 'params':
                student_input_size = 545290
            if args.SR == 'L2T':
                student_input_size =  3
            if args.SR == 'loss_mismatch':
                student_input_size =  5

                
        elif args.student_type == 'DDPG' and args.env == 'fetch_push':
            student_input_size = 25
            if args.SR == 'params':
                student_input_size = 560650 
            
            if args.SR == 'L2T':
                student_input_size =  3
            if args.SR == 'loss_mismatch':
                student_input_size =  5

        print('using student input size', student_input_size)
        return None, student_input_size



