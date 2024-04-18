import grid2op
from grid2op.Reward import L2RPNSandBoxScore, L2RPNReward
from l2rpn_baselines.DeepQSimple import evaluate

nm_ = "mydeepq"

env = grid2op.make("l2rpn_wcci_2022",
           reward_class=L2RPNSandBoxScore,
           other_rewards={
               "reward": L2RPNReward
           })


evaluate(env,
          name=nm_,
          load_path=f"test_results/saved_model",
          logs_path=f"test_results/eval_logs",
          nb_episode=10,
          nb_process=1,
          max_steps=-1,
          verbose=True,
          save_gif=True)