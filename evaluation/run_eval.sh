set -e

UNIFORM_ROOT=${1:?uniform_root required}
RESULTS_JSON=${2:?results_json required}
LABEL=${3:-ours}
DATASETS=${4:-ho3d,ycbineoat,proxy}

OUTPUT_ROOT=output/metrics
EXTRA_OUTPUT_ROOT=output/metrics_extra

IFS=',' read -ra dataset_list <<< "$DATASETS"
for dataset in "${dataset_list[@]}"; do
    filter_json="evaluation/benchmarks/${dataset}/w1_f49.json"
    echo "Evaluating $LABEL on $dataset..."

    # Paper metrics: ATE, ARE, RPE, drift, 2D error
    python -m evaluation.evaluate_slam \
        --uniform_root "$UNIFORM_ROOT" \
        --filter_json "$filter_json" \
        --results_json "$RESULTS_JSON" \
        --output "${OUTPUT_ROOT}/${dataset}_${LABEL}.json" \
        --label "$LABEL"

    # Additional per-frame metrics: ADD(-S), MSSD, MSPD, VUS
    python -m evaluation.evaluate \
        --uniform_root "$UNIFORM_ROOT" \
        --filter_json "$filter_json" \
        --results_json "$RESULTS_JSON" \
        --output "${EXTRA_OUTPUT_ROOT}/${dataset}_${LABEL}.json" \
        --label "$LABEL"
done

python -m evaluation.make_table --metrics_dir "$OUTPUT_ROOT"
