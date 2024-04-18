import grid2op
from l2rpn_baselines.utils import TrainingParam, NNParam
from l2rpn_baselines.DeepQSimple import train
from grid2op.Action import PowerlineSetAction
from grid2op.Opponent import RandomLineOpponent, BaseActionBudget

env_name = "l2rpn_case14_sandbox"

env_with_opponent = grid2op.make(env_name,
                                 opponent_attack_cooldown=12*6,
                                 opponent_attack_duration=12*1,
                                 opponent_budget_per_ts=0.5,
                                 opponent_init_budget=0.,
                                 opponent_action_class=PowerlineSetAction,
                                 opponent_class=RandomLineOpponent,
                                 opponent_budget_class=BaseActionBudget,
                                 kwargs_opponent={"lines_attacked":
                                      ["1_3_3", "1_4_4", "3_6_15", "9_10_12", "11_12_13", "12_13_14"]}
                                 )
    
    
tp = TrainingParam(update_tensorboard_freq=10)


li_attr_obs_X = ["rho"]

observation_size = NNParam.get_obs_size(env_with_opponent, li_attr_obs_X)

i = 4
size_multiplier = 4 * (i)
sizes = [size_multiplier * 2, size_multiplier * 1, size_multiplier * 2]  

kwargs_archi = {'observation_size': observation_size,
                'sizes': sizes,
                'activs': ["relu" for _ in sizes],  # all relu activation function
                "list_attr_obs": li_attr_obs_X}


kwargs_converters = {"all_actions": None,
                      "change_line_status": True,
                      "change_bus_vect": False,
                      "redispatch": False,
                      "storage_power": False,
                      "curtailment": False
                      }

iterations = 10000000
nm_ = f"attacked_grid_14_{iterations}"

try:
    train(env_with_opponent,
          name=nm_,
          iterations= iterations,
          save_path= "test_results/saved_model",
          load_path=None,
          logs_dir= "test_results/logs",
          training_param=tp,
          kwargs_converters=kwargs_converters,
          kwargs_archi=kwargs_archi)
finally:
    env_with_opponent.close()

    
    