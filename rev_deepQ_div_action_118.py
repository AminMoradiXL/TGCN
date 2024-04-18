import grid2op
from grid2op.Reward import L2RPNReward
from l2rpn_baselines.utils import TrainingParam, NNParam
from l2rpn_baselines.DeepQSimple import train


for i in range(8):
    
    env = grid2op.make("l2rpn_wcci_2022",
                        reward_class=L2RPNReward)

    tp = TrainingParam(update_tensorboard_freq=100)


    li_attr_obs_X = ["rho"]

    observation_size = NNParam.get_obs_size(env, li_attr_obs_X)

    size_multiplier = 32
    sizes = [size_multiplier * 2, size_multiplier * 1, size_multiplier * 2]  
    
    kwargs_archi = {'observation_size': observation_size,
                    'sizes': sizes,
                    'activs': ["relu" for _ in sizes],  # all relu activation function
                    "list_attr_obs": li_attr_obs_X}
    
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
              iterations= 200000,
              save_path="test_results/saved_model",
              load_path=None,
              logs_dir="test_results/logs",
              training_param=tp,
              kwargs_converters=kwargs_converters,
              kwargs_archi=kwargs_archi)
    finally:
        env.close()
    



# set_line_status if the action tries to set the status of some powerlines. 

# change_line_status: if the action tries to change the status of some powerlines. 

# change_bus_vect: if the action tries to change the topology of some substations.

# set_bus_vect: if the action tries to set the topology of some substations. 

# redispatch the redispatching action (if any). It gives, for each generator (all generator, not just the dispatchable one) the amount of power redispatched in this action.

# storage_power: the setpoint for production / consumption for all storage units

# curtailment: the curtailment performed on all generator    

# load_p: if the action modifies the active loads.

# load_q: if the action modifies the reactive loads.

# prod_p: if the action modifies the active productions of generators.

# prod_v: if the action modifies the voltage setpoint of generators.