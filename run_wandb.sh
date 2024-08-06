#!/bin/bash

source ./venv/bin/activate

declare -a pids=()
python3 tdmpc2/train.py task="dog-run" steps=7000000 &
pids+=($!)
sleep 10
python3 tdmpc2/train.py task="dog-run" steps=7000000 &
pids+=($!)
sleep 10
python3 tdmpc2/train.py task="dog-run" steps=7000000 &
pids+=($!)
sleep 10
python3 tdmpc2/train.py task="dog-run" steps=7000000 &
pids+=($!)
sleep 10

for pid in ${pids[*]}; do
    wait $pid
done
