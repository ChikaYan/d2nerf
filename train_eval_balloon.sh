export DATASET_PATH=data/vrig_balloon
export EXPERIMENT_PATH=logs/vrig_balloon
export CONFIG_PATH=configs/rl/001.gin

python train.py \
    --base_folder $EXPERIMENT_PATH \
    --gin_bindings="data_dir='$DATASET_PATH'" \
    --gin_bindings="TrainConfig.print_every = 1" \
    --gin_configs $CONFIG_PATH

# python eval.py \
#     --base_folder $EXPERIMENT_PATH \
#     --gin_bindings="data_dir='$DATASET_PATH'" \
#     --gin_configs $CONFIG_PATH

