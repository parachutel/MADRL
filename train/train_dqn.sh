export PYTHONPATH=$(pwd)/..:$(pwd)/../rltools:$(pwd)/../rllab:$PYTHONPATH
python3  -W ignore ../runners/train_dqn_disc_multi_aircraft.py \
    --n_iter 4000000 \
    --target_update_step 5000 \
    --batch_size 64 \
    --discount 0.99 \
    --max_experience_size 20000 \
    --traj_sim_len 300 \
    --n_eval_traj 2 \
    --n_agents  15 \
    --save_freq 5000 \
    --log_file_name dqn_train_log_n_15
