export PYTHONPATH=$(pwd):$(pwd)/rltools:$(pwd)/rllab:$PYTHONPATH
python3 runners/run_multiaircraft.py rllab \
    --control decentralized \
    --policy_hidden 100,50,25 \
    --n_iter 200 \
    --n_agents 5 \
    --batch_size 24000 \
    --curriculum lessons/cas/env.yaml