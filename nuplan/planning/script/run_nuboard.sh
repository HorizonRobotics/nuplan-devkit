#! /usr/bin/env bash
SIMULATION_DIR=$1

python nuplan/planning/script/run_nuboard.py \
    scenario_builder=nuplan_mini \
    simulation_path=$SIMULATION_DIR 