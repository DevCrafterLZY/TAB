python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_multi_config.json" --data-name-list "SKAB_valve1_2.csv" --model-name "merlion.IsolationForest" --model-hyper-params '{"max_n_samples": 256, "n_estimators": 50}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "label/IsolationForest"