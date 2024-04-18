import grid2op
from grid2op.Reward import L2RPNReward
from l2rpn_baselines.utils import TrainingParam, NNParam
from l2rpn_baselines.SACOld import train


for i in range(8):

    env = grid2op.make("l2rpn_case14_sandbox",
                       reward_class=L2RPNReward)
    
    tp = TrainingParam(update_tensorboard_freq=100)

    # li_attr_obs_X = ["day_of_week", "hour_of_day", "minute_of_hour", "prod_p", "prod_v", "load_p", "load_q",
    #                  "actual_dispatch", "target_dispatch", "topo_vect", "time_before_cooldown_line",
    #                  "time_before_cooldown_sub", "rho", "timestep_overflow", "line_status"]
    
    li_attr_obs_X = ["rho"]
    
    observation_size = NNParam.get_obs_size(env, li_attr_obs_X)
    
    sizes_q = [64,32,64]  
    sizes_v = [32,32]  
    sizes_pol = [64,32,64]  
    
    kwargs_archi = {'observation_size': observation_size,
                    'sizes': sizes_q,
                    'activs': ["relu" for _ in range(len(sizes_q))],
                    "list_attr_obs": li_attr_obs_X,
                    "sizes_value": sizes_v,
                    "activs_value": ["relu" for _ in range(len(sizes_v))],
                    "sizes_policy": sizes_pol,
                    "activs_policy": ["relu" for _ in range(len(sizes_pol))]
                    }
    
    if i == 0: 
        kwargs_converters = {"all_actions": None,
                              "change_line_status": True,
                              "change_bus_vect": False,
                              "redispatch": False,
                              "storage_power": False,
                              "curtailment": False
                              }
        nm_ = "action_PLS_only"
    elif i == 1: 
        kwargs_converters = {"all_actions": None,
                              "change_line_status": False,
                              "change_bus_vect": True,
                              "redispatch": False,
                              "storage_power": False,
                              "curtailment": False
                              }
        nm_ = "action_TG_only"
    elif i == 2: 
        kwargs_converters = {"all_actions": None,
                              "change_line_status": False,
                              "change_bus_vect": False,
                              "redispatch": True,
                              "storage_power": False,
                              "curtailment": False
                              }
        nm_ = "action_redisp_only"
    elif i == 3: 
        kwargs_converters = {"all_actions": None,
                              "change_line_status": False,
                              "change_bus_vect": False,
                              "redispatch": False,
                              "storage_power": True,
                              "curtailment": False
                              }
        nm_ = "action_stor_only"
    elif i == 4: 
        kwargs_converters = {"all_actions": None,
                              "change_line_status": False,
                              "change_bus_vect": False,
                              "redispatch": False,
                              "storage_power": False,
                              "curtailment": True
                              }
        nm_ = "action_curt_only"    
    elif i == 5: 
        kwargs_converters = {"all_actions": None,
                              "change_line_status": True,
                              "change_bus_vect": True,
                              "redispatch": False,
                              "storage_power": False,
                              "curtailment": False
                              }
        nm_ = "action_PLS_TG"    
    elif i == 6: 
        kwargs_converters = {"all_actions": None,
                              "change_line_status": False,
                              "change_bus_vect": False,
                              "redispatch": True,
                              "storage_power": True,
                              "curtailment": True
                              }
        nm_ = "action_redisp_stor_curt"  
    elif i == 7: 
        kwargs_converters = {"all_actions": None,
                              "change_line_status": True,
                              "change_bus_vect": True,
                              "redispatch": True,
                              "storage_power": True,
                              "curtailment": True
                              }
        nm_ = "action_all"  
        
    
    try:
        train(env,
              name=nm_,
              iterations=200000,
              save_path="test_results/saved_model",
              load_path=None,
              logs_dir="test_results/logs",
              training_param=tp,
              kwargs_converters=kwargs_converters,
              kwargs_archi=kwargs_archi)
    finally:
        env.close()