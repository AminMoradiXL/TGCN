import grid2op
from l2rpn_baselines.DeepQSimple import evaluate
from grid2op.Action import PowerlineSetAction
from grid2op.Opponent import RandomLineOpponent, BaseActionBudget
from grid2op.Runner import Runner

nm_ = "attacked_redisp_118"

# nm_ = "attacked_grid_118_10000000"

env_name = "l2rpn_wcci_2022"

nb_episode = 1000

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
    

trained_agent, trained_res = evaluate(env_with_opponent,
          name=nm_,
          load_path="test_results/saved_model",
          logs_path="test_results/eval_logs/redisp_118/",
          nb_episode=nb_episode,
          nb_process=1,
          max_steps=-1,
          verbose=True,
          save_gif=False)

# print("Evaluation summary for trained agent:")
# for _, chron_name, cum_reward, nb_time_step, max_ts in trained_res:
#     msg_tmp = "chronics at: {}".format(chron_name)
#     msg_tmp += "\ttotal score: {:.6f}".format(cum_reward)
#     msg_tmp += "\ttime steps: {:.0f}/{:.0f}".format(nb_time_step, max_ts)
#     print(msg_tmp)
    

# # Do nothing agent 

# runner_params = env_with_opponent.get_params_for_runner()
# runner = Runner(**runner_params)

# res = runner.run(nb_episode=nb_episode,
#                 nb_process=1
#                 )

# print("Evaluation summary for DN agent:")
# for _, chron_name, cum_reward, nb_time_step, max_ts in res:
#     msg_tmp = "chronics at: {}".format(chron_name)
#     msg_tmp += "\ttotal score: {:.6f}".format(cum_reward)
#     msg_tmp += "\ttime steps: {:.0f}/{:.0f}".format(nb_time_step, max_ts)
#     print(msg_tmp)
    
