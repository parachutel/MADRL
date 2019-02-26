export PYTHONPATH=$(pwd):$(pwd)/rltools:$(pwd)/rllab:$PYTHONPATH
python3 runners/run_multiwalker.py rllab \
    --control decentralized \
    --policy_hidden 100,50,25 \
    --n_iter 200 \
    --n_walkers 2 \
    --batch_size 24000 \
    --curriculum lessons/multiwalker/env.yaml