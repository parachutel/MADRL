export PYTHONPATH=$(pwd):$(pwd)/rltools:$(pwd)/rllab:$(pwd)/madrl_environments:$PYTHONPATH
python3 airspace_animation.py \
    --policy trpo_20_agent_direct_ALL_ENV_Tmax_300_PEN_HEAVY_True_rew_nmac_-150_rew_arr_2/itr_2064_direct_real_good.pkl \
    --constant_n_agents 0 \
    --takeoff_rate 50 \
    --sim_time_steps 1000 \
    --equally_spaced_circle 0 \
    --mode square \
    --n_agents 30

# trpo_20_agent_direct_ALL_ENV_Tmax_300_PEN_HEAVY_True_rew_nmac_-150_rew_arr_2/itr_2064_direct_real_good.pkl
# trpo_full_curr_3_passes_ALL_ENV_Tmax_300_PEN_HEAVY_True_rew_nmac_-150_rew_arr_2/itr_299.pkl
# trpo_shift_ALL_ENV_Tmax_300_PEN_HEAVY_True_rew_nmac_-150_rew_arr_2/itr_7833.pkl
# trpo_full_curr_3_passes_ALL_ENV_Tmax_300_PEN_HEAVY_True_rew_nmac_-150_rew_arr_2_run_2/itr_299.pkl