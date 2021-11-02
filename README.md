# CurriculumLearning



run_teacher_training_loop file contains the main function (CL_Loop) which trains the teacher agent.

evaluate_teacher file contains the main function (eval_loop) which evaluates the teacher.

config.py file is where you specify all the parameters + hyperparameters, then this file will execute the training and evaluation loop for the teacher. 


How the directory structure is set up:

I seperate the data into directories based on the env then the state represenation.
Example:
./Env/StateRep/

I included the Four Rooms folder and the 3 folders for the action_return, policy_table and q_matrix state representations.
Within each state rep subdirectory, there are 3 additional directories: teacher-checkpoints, teacher-returns, and evaluation-data. 

I already included these 3 subdirectories in the action_return directory as an example. 



