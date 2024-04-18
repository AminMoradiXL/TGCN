from tensorboard import program

log_directory = 'saved_results/training/attack_118_div_agents_1M/logs'


if __name__ == "__main__":
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', log_directory])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")
    

# "Mean_alive_30", description="Average number of steps (per episode) made over the last 30 completed episodes"

# "Mean_reward_30", description="Average (final) reward obtained over the last 30 completed episodes"

# "loss", description="Training loss (for the last training batch)"

# "last_alive", description="Final number of steps for the last complete episode"

# "last_reward", description="Final reward over the last complete episode"

# "mean_reward", description="Average reward over the whole episodes played"

# "mean_alive", description="Average time alive over the whole episodes played"

# "mean_reward_100", description="Average number of steps (per episode) made over the last 100 completed episodes"

# "mean_alive_100", description="Average (final) reward obtained over the last 100 completed episodes"

# "nb_different_action_taken", description="Number of different actions played the last {} steps"
    
# "nb_illegal_act", description="Number of illegal actions played the last {} steps"

# "nb_ambiguous_act", description="Number of ambiguous actions played the last {} steps"

# "nb_total_success", description="Number of times the episode was completed entirely (no game over)"

# "z_lr", description="Current learning rate"

# "z_epsilon", description="Current epsilon (from the epsilon greedy)"

# "z_max_iter", description="Maximum number of time steps before deciding a scenario is over (=win)"

# "z_total_episode", description="Total number of episode played (number of \"reset\")"

# "zz_freq_inj", description="Frequency of \"injection\" actions type played over the last {} actions"

# "zz_freq_voltage", description="Frequency of \"voltage\" actions type played over the last {} actions"

# "z_freq_topo", description="Frequency of \"topo\" actions type played over the last {} actions"

# "z_freq_line_status", description="Frequency of \"line status\" actions type played over the last {} actions"

# "z_freq_redisp", description="Frequency of \"redispatching\" actions type played over the last {} actions"

# "z_freq_do_nothing", description="Frequency of \"do nothing\" actions type played over the last {} actions"

# "z_freq_storage", description="Frequency of \"storage\" actions type played over the last {} actions"

# "z_freq_curtail", description="Frequency of \"curtailment\" actions type played over the last {} actions"