export PYTHONPATH=$(pwd):$(pwd)/rltools:$(pwd)/rllab:$PYTHONPATH
python3 runners/run_multiaircraft.py rllab \
	--algo tftrpo \
    --control decentralized \
    --sampler simple \
    --policy_hidden 100,50,50 \
    --n_iter 35 \
    --n_agents 15 \
    --batch_size 30000 \
    --max_path_length 5000 \
    --rew_arrival 15.0 \
    --rew_closing 0.05 \
    --rew_nmac -15.0 \
    --rew_large_turnrate -0.1 \
    --curriculum lessons/cas/env.yaml