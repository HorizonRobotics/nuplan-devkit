#! /usr/bin/env bash

# Regular simulation
MODEL_PATH="/mnt/nas20/jingyu.qian/nuplan_training_save_dir/training_open_loop_1117/2022.11.17.23.04.25/best_model/epoch\=38-step\=23906.ckpt"
EXPERIMENT="simulate_closed_loop_nonreactive_$(date "+%m%d")"
SAVE_DIR="/mnt/nas20/jingyu/nuplan_simulation_save_dir"

python nuplan/planning/script/run_simulation.py \
    experiment_name=$EXPERIMENT \
    group=$SAVE_DIR \
    +simulation='closed_loop_nonreactive_agents' \
    planner=ml_planner \
    model=raster_model \
    planner.ml_planner.model_config='${model}' \
    planner.ml_planner.checkpoint_path=$MODEL_PATH \
    scenario_builder=nuplan_mini \
    scenario_filter=simulation_test_split \
    enable_simulation_progress_bar=true \
    exit_on_failure=true \
    scenario_filter.num_scenarios_per_type=10
