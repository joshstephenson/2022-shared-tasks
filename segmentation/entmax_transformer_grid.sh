#!/usr/bin/env bash

if [ -z "$1" ]; then
    echo "Please provide a language name {ces|eng|fra|hun|ita|lat|mon|rus|spa}"
    exit 1
else
    LANG="$1"; shift
    DATA_BIN="2022SegmentationST/data/${LANG}.word"
    NAME="${LANG}"
    echo "NAME: ${NAME}"
    GOLD_PATH="2022SegmentationST/data/${LANG}.word.test.gold.tsv"
    echo "GOLD PATH: ${GOLD_PATH}"
    GRID_LOC="data/models/${LANG}"
    if [ ! -d "$GRID_LOC" ]; then
        mkdir -p "$GRID_LOC"
    fi
    echo "GRID_LOC: ${GRID_LOC}"
fi

ENTMAX_ALPHA=1.5
LR=0.001
BEAM=5

grid() {
    local -r EMB="$1"; shift
    local -r HID="$1"; shift
    local -r LAYERS="$1" ; shift
    local -r HEADS="$1" ; shift
    for WARMUP in 4000 8000 ; do
        for DROPOUT in 0.1 0.3 ; do
            BATCHES=$@
            if [ -z "$BATCHES" ]; then
                BATCHES=8192
                echo "No batch size supplied. Setting to 8192."
            fi
            for BATCH in $BATCHES ; do
                MODEL_DIR="${GRID_LOC}/${NAME}-entmax-minloss-${EMB}-${HID}-${LAYERS}-${HEADS}-${BATCH}-${ENTMAX_ALPHA}-${LR}-${WARMUP}-${DROPOUT}"
                if [ ! -f "${MODEL_DIR}/dev-5.results" ]
                then
                    bash fairseq_train_entmax_transformer.sh $DATA_BIN $NAME $EMB $HID $LAYERS $HEADS $BATCH $ENTMAX_ALPHA $LR $WARMUP $DROPOUT $GRID_LOC
                    if [ $? -ne 0 ]; then
                        echo "fairseq_train_entmax_transformer.sh failed"
                        exit 1
                    fi
                    bash fairseq_segment.sh $DATA_BIN $MODEL_DIR $ENTMAX_ALPHA $BEAM $GOLD_PATH
                    if [ $? -ne 0 ]; then
                        echo "fairseq_segment.sh failed."
                        exit 1
                    else
                        echo "Trained data output to: ${MODEL_DIR}"
                    fi
                else
                    echo "skipping ${MODEL_DIR}"
                fi
            done
        done
    done
}


grid 256 1024 6 8 $@
grid 512 2048 6 8 $@
