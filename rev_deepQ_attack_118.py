import grid2op
from l2rpn_baselines.utils import TrainingParam, NNParam
from l2rpn_baselines.DeepQSimple import train
from grid2op.Action import PowerlineSetAction
from grid2op.Opponent import RandomLineOpponent, BaseActionBudget

env_name = "l2rpn_wcci_2022"

env_with_opponent = grid2op.make(env_name,
                                 opponent_attack_cooldown=12*24,
                                 opponent_attack_duration=12*3,
                                 opponent_budget_per_ts=0.5,
                                 opponent_init_budget=0.,
                                 opponent_action_class=PowerlineSetAction,
                                 opponent_class=RandomLineOpponent,
                                 opponent_budget_class=BaseActionBudget,
                                 kwargs_opponent={"lines_attacked":
                                                     ["26_31_106",
                                                      "21_22_93",
                                                      "17_18_88",
                                                      "4_10_162",
                                                      "12_14_68",
                                                      "14_32_108",
                                                      "62_58_180",
                                                      "62_63_160",
                                                      "48_50_136",
                                                      "48_53_141",
                                                      "41_48_131",
                                                      "39_41_121",
                                                      "43_44_125",
                                                      "44_45_126",
                                                      "34_35_110",
                                                      "54_58_154",
                                                      "74_117_81",
                                                      "80_79_175",
                                                      "93_95_43",
                                                      "88_91_33",
                                                      "91_92_37",
                                                      "99_105_62",
                                                      "102_104_61"]}
                                 )
    
    
tp = TrainingParam(update_tensorboard_freq=100)


li_attr_obs_X = ["rho"]

observation_size = NNParam.get_obs_size(env_with_opponent, li_attr_obs_X)


size_multiplier = 8
sizes = [size_multiplier * 8, size_multiplier * 4, size_multiplier * 8]  

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

iterations = 1000000
nm_ = f"attacked_pls_118"

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

    
    