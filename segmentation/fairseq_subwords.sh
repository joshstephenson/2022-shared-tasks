#!/usr/bin/env bash

if [ -z "$1" ]; then
    echo "Please provide a language name {ces|eng|fra|hun|ita|lat|mon|rus|spa}"
    exit 1
else
    readonly LANG="$1";
    readonly MODEL_NAME="$2"
    if [ -z "$2" ]; then
        echo "Please provide a model name (hun-14-03-2024-entmax-minlev-256-1024-6-8-8192-1.5-0.001-4000-0.1/)"
        exit 1
    fi
    shift 2
    readonly DATA_BIN="data/preprocessed/${LANG}"
    readonly MODEL_PATH="data/models/${LANG}/${MODEL_NAME}"
fi
readonly ENTMAX_ALPHA=1.5
readonly BEAM=5

decode() {
    local -r CP="$1"; shift
    local -r MODE="$1"; shift
    # Fairseq insists on calling the dev-set "valid"; hack around this.
    local -r FAIRSEQ_MODE="${MODE/dev/valid}"
    CHECKPOINT="${CP}/checkpoint_best.pt"
    OUT="${CP}/${MODE}-${BEAM}.subwords.out"
    PRED="${CP}/${MODE}-${BEAM}.subwords.pred"
    # Makes raw predictions.
    fairseq-generate \
        "${DATA_BIN}" \
        --source-lang="src" \
        --target-lang="tgt" \
        --path="${CHECKPOINT}" \
        --gen-subset="${FAIRSEQ_MODE}" \
        --beam="${BEAM}" \
        --alpha="${ENTMAX_ALPHA}" \
	--batch-size 256 \
        > "${OUT}"
    # Extracts the predictions into a TSV file.
    cat "${OUT}" | grep -P '^H-'  | cut -c 3- | sort -n -k 1 | awk -F "\t" '{print $NF}' > $PRED
}

decode $MODEL_PATH dev
