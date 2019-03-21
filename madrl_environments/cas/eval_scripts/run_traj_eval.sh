export PYTHONPATH=$(pwd)/../../..:$(pwd)/../../../rltools:$(pwd)/../../../rllab:$(pwd)/../../../madrl_environments:$PYTHONPATH
python3 traj_eval.py \
    --policy trpo_full_curr_3_passes_ALL_ENV_Tmax_300_PEN_HEAVY_True_rew_nmac_-150_rew_arr_2_run_2/itr_299.pkl \
    --equally_spaced_circle False \
    --ylabel True \
    --n_eval 20 \
    --n_agents 5 \
    --render False

python3 traj_eval.py \
    --policy trpo_full_curr_3_passes_ALL_ENV_Tmax_300_PEN_HEAVY_True_rew_nmac_-150_rew_arr_2_run_2/itr_299.pkl \
    --equally_spaced_circle False \
    --ylabel True \
    --n_eval 20 \
    --n_agents 10 \
    --render False

python3 traj_eval.py \
    --policy trpo_full_curr_3_passes_ALL_ENV_Tmax_300_PEN_HEAVY_True_rew_nmac_-150_rew_arr_2_run_2/itr_299.pkl \
    --equally_spaced_circle False \
    --ylabel True \
    --n_eval 20 \
    --n_agents 20 \
    --render False

python3 traj_eval.py \
    --policy trpo_full_curr_3_passes_ALL_ENV_Tmax_300_PEN_HEAVY_True_rew_nmac_-150_rew_arr_2_run_2/itr_299.pkl \
    --equally_spaced_circle False \
    --ylabel True \
    --n_eval 20 \
    --n_agents 30 \
    --render False

python3 traj_eval.py \
    --policy trpo_full_curr_3_passes_ALL_ENV_Tmax_300_PEN_HEAVY_True_rew_nmac_-150_rew_arr_2_run_2/itr_299.pkl \
    --equally_spaced_circle False \
    --ylabel True \
    --n_eval 20 \
    --n_agents 40 \
    --render False

# trpo_20_agent_direct_ALL_ENV_Tmax_300_PEN_HEAVY_True_rew_nmac_-150_rew_arr_2/itr_2064_direct_real_good.pkl
# trpo_full_curr_3_passes_ALL_ENV_Tmax_300_PEN_HEAVY_True_rew_nmac_-150_rew_arr_2/itr_299.pkl
# trpo_shift_ALL_ENV_Tmax_300_PEN_HEAVY_True_rew_nmac_-150_rew_arr_2/itr_7833.pkl
# trpo_full_curr_3_passes_ALL_ENV_Tmax_300_PEN_HEAVY_True_rew_nmac_-150_rew_arr_2_run_2/itr_299.pkl