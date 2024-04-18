import grid2op
from grid2op.Reward import L2RPNReward
from l2rpn_baselines.utils import TrainingParam, NNParam
from l2rpn_baselines.DuelQSimple import train


for i in range(3):
    
    env = grid2op.make("rte_case14_realistic",
                       reward_class=L2RPNReward)
    
    tp = TrainingParam(update_tensorboard_freq=20)
    
    li_attr_obs_X = ["day_of_week", "hour_of_day", "minute_of_hour", "prod_p", "prod_v", "load_p", "load_q",
                     "actual_dispatch", "target_dispatch", "topo_vect", "time_before_cooldown_line",
                     "time_before_cooldown_sub", "rho", "timestep_overflow", "line_status"]
    
    li_attr_obs_X = ["rho"]
    
    observation_size = NNParam.get_obs_size(env, li_attr_obs_X)
    size_multiplier = 16
    sizes = [size_multiplier * 2, size_multiplier * 1, size_multiplier * 2]  
    
    kwargs_archi = {'observation_size': observation_size,
                    'sizes': sizes,
                    'activs': ["relu" for _ in sizes],  
                    "list_attr_obs": li_attr_obs_X}
    
    
    if i == 0: 
        kwargs_converters = {"all_actions": None,
                              "change_line_status": True,
                              "change_bus_vect": False,
                              }
        nm_ = "action_PLS_only"
    elif i == 1: 
        kwargs_converters = {"all_actions": None,
                              "set_line_status": False,
                              "change_bus_vect": True,
                              }
        nm_ = "action_TG_only"
    elif i == 2: 
        kwargs_converters = {"all_actions": None,
                              "set_line_status": True,
                              "change_bus_vect": True,
                              }
        nm_ = "action_both"        
        
    
    try:
        train(env,
              name=nm_,
              iterations=1000,
              save_path="test_results/saved_model",
              load_path=None,
              logs_dir="test_results/logs",
              training_param=tp,
              kwargs_converters=kwargs_converters,
              kwargs_archi=kwargs_archi)
    finally:
        env.close()