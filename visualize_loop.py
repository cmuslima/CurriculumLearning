
import numpy as np
from init_teacher import teacher_utils
import init_student




def visualize(args):
   
    teacher_help_fns = teacher_utils(args)
    teacher_agent, student_env = teacher_help_fns.visual_student(args) #args.teacher_evaluation_seed will be a value outside of 0-5
  