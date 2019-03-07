export PYTHONPATH=$(pwd)/..:$(pwd)/../rltools:$(pwd)/../rllab:$PYTHONPATH
python3 ../runners/run_multiaircraft.py rllab \
	--algo thddpg \
    --discount 0.99 \
    --qf_learning_rate 1e-3 \
    --policy_learning_rate 1e-4 \
    --batch_size 32 \
    --qf_hidden 400,300 \
    --policy_hidden 400,300 \
    --max_path_length 1000 \
    --control decentralized \
    --sampler simple \
    --n_agents 15 \
    --rew_arrival 15.0 \
    --rew_closing 0.05 \
    --rew_nmac -15.0 \
    --rew_large_turnrate -0.1 \