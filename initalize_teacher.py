    
from ppo import PPOAlgo
from teacher_ppo_model import ACModel
from rl_starter_files_master.student_utils import device, get_model_dir, get_status, save_status


#from gym_fetch_RT import RT_run
try:
    from gym_fetch_RT import RT_run
except:
    pass
import random

def init_teacher( args, teacher_state_size, teacher_action_size, env, seed, student_seed = None, file=None):
    teacher_model = None
    if args.trained_teacher == False and args.evaluation:
        seed = student_seed
    print('inside init teacher')
    if args.teacher_agent == 'DQN':
        
        print('am I inside the DQN call')
        if args.SR == 'buffer_policy' or args.SR == 'buffer_q_table' or args.SR == 'buffer_action' or args.SR == 'buffer_max_policy' or args.SR == 'buffer_max_q_table' or args.SR == 'buffer_max_action':
            #print('am I in the right place')
            from buffer_teacher_agent import DQNAgent
            
            teacher_agent = DQNAgent(state_size=teacher_state_size, action_size = teacher_action_size, seed=seed, args= args) 
            #print('here')
            if args.evaluation and args.trained_teacher:
                teacher_agent.qnetwork_local.load_state_dict(torch.load(file))
        else:
            #print('am I ever here')
            from teacher_agent import DQNAgent
            #print('seed here', seed)
            teacher_agent = DQNAgent(state_size=teacher_state_size, action_size = teacher_action_size, seed=seed, args = args) 

            if args.evaluation and args.trained_teacher:
                teacher_agent.qnetwork_local.load_state_dict(torch.load(file))

    #example of what I will include once I have different kinds of teacher agents.  
    if args.teacher_agent == 'PPO':
        print('am I inside the PPO call')
        model_name = 'test_teacher_ppo'
        model_dir = get_model_dir(model_name, args)
        try:
            status = get_status(model_dir)
        except OSError:
            status = {"num_frames": 0, "update": 0}

        # Load model
        print('about to load the model')
        teacher_model = ACModel(teacher_state_size, teacher_action_size, args)
        if "model_state" in status:
            teacher_model.load_state_dict(status["model_state"])
        teacher_model.to(device)
        print('finished loading the model')

        teacher_agent = PPOAlgo(args, env, teacher_model, device, args.interaction_steps, args.teacher_discount, args.teacher_lr, args.teacher_gae_lambda,
                            args.teacher_entropy_coef, args.teacher_value_loss_coef, args.teacher_max_grad_norm, args.teacher_recurrence,
                            args.teacher_optim_eps, args.teacher_clip_eps, args.teacher_epochs, args.teacher_batchsize, None)
                #print('here')
        if args.evaluation and args.trained_teacher:
            teacher_agent.qnetwork_local.load_state_dict(torch.load(file))
        print('i got my PPO teacher')
    

    
    return teacher_agent, teacher_model