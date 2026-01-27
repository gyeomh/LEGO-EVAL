#!/bin/bash

BASE_PATH="" # path to where the scenes are stored
SAVE_DIR="" # path to where the processed scenes will be saved
OBJAVERSE_ASSET_DIR="" # path to where the objaverse assets are stored
LEGO_BENCH_DATA_DIR="" # path to LEGO_Bench/data/full_data.json
THOR_COMMIT_ID="3213d486cd09bcbafce33561997355983bdf8d1a" # Thor commit id

XORG_SCREENS=(":4" ":5") # adjust this to the screen numbers you want to use

NUM_WORKERS=${#XORG_SCREENS[@]}

SCENES=($(ls $BASE_PATH/*.json))
NUM_SCENES=${#SCENES[@]}

echo "Total scenes: $NUM_SCENES"
echo "Workers: $NUM_WORKERS"

for ((i=0; i<NUM_WORKERS; i++)); do
    SCREEN=${XORG_SCREENS[$i]}
    TMP_DIR=$(mktemp -d)

    echo "Worker $i → XORG $SCREEN"

    # scene 분할
    for ((j=i; j<NUM_SCENES; j+=NUM_WORKERS)); do
        ln -s "${SCENES[$j]}" "$TMP_DIR/$(basename "${SCENES[$j]}")"
    done

    DISPLAY=$SCREEN \
    python preprocess/preprocess_scenes.py \
        --base_path "$TMP_DIR" \
        --save_dir "$SAVE_DIR" \
        --objaverse_dir "$OBJAVERSE_ASSET_DIR" \
        --thor_id "$THOR_COMMIT_ID" \
        --xorg_screens "$SCREEN" \
        --lego_bench_data_dir "$LEGO_BENCH_DATA_DIR" \
        --lego_bench \
        > "$SAVE_DIR/log_worker_$i.txt" 2>&1 &

done

wait
echo "All workers finished."
