#!/usr/bin/env bash

if [ -z "$1" ]; then
    echo "Please provide a language name {ces|eng|fra|hun|ita|lat|mon|rus|spa}"
    exit 1
else
    readonly LANG="$1"; shift
    readonly DATA_PATH="2022SegmentationST/data/${LANG}.word"
    readonly OUT_PATH="data/preprocessed/${LANG}"
    if [ ! -d "$OUT_PATH" ]; then
        mkdir -p "$OUT_PATH"
    fi
    readonly VOCAB=8000 #(underexplored)
    echo "DATA_PATH: ${DATA_PATH}"
    echo "OUT_PATH: ${OUT_PATH}"
    echo "VOCAB: ${VOCAB}"
fi


bin() {
    tail -n +4 "${OUT_PATH}/src.vocab" | cut -f 1 | sed "s/$/ 100/g" > "${OUT_PATH}/src.fairseq.vocab"
    tail -n +4 "${OUT_PATH}/tgt.vocab" | cut -f 1 | sed "s/$/ 100/g" > "${OUT_PATH}/tgt.fairseq.vocab"
    # todo: fix testpref when it is available
    fairseq-preprocess \
        --source-lang="src" \
        --target-lang="tgt" \
        --trainpref="${OUT_PATH}/train" \
        --validpref="${OUT_PATH}/dev" \
        --testpref="${OUT_PATH}/test" \
        --tokenizer=space \
        --thresholdsrc=1 \
        --thresholdtgt=1 \
        --srcdict "${OUT_PATH}/src.fairseq.vocab" \
        --tgtdict "${OUT_PATH}/tgt.fairseq.vocab" \
        --destdir="${OUT_PATH}"
}

python scripts/tokenize.py "${DATA_PATH}.train.tsv" --src-tok-type spm --tgt-tok-type spm --vocab-size $VOCAB --out-dir $OUT_PATH --split train $@
if [ $? -ne 0 ]; then echo "Tokenizing train failed" && exit 1 ; fi
python scripts/tokenize.py "${DATA_PATH}.dev.tsv" --src-tok-type spm --tgt-tok-type spm --vocab-size $VOCAB --existing-src-spm "${OUT_PATH}/src" --existing-tgt-spm "${OUT_PATH}/tgt" --out-dir $OUT_PATH --split dev --shared-data
if [ $? -ne 0 ]; then echo "Tokenizing dev failed" && exit 1 ; fi
python scripts/tokenize.py "${DATA_PATH}.test.gold.tsv" --src-tok-type spm --tgt-tok-type spm --vocab-size $VOCAB --existing-src-spm "${OUT_PATH}/src" --existing-tgt-spm "${OUT_PATH}/tgt" --out-dir $OUT_PATH --split test --shared-data
if [ $? -ne 0 ]; then echo "Tokenizing test.gold failed" && exit 1 ; fi
bin

