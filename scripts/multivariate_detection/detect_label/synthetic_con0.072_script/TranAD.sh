python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_multi_config.json" --data-name-list "synthetic_con0.072.csv" --model-name "self_impl.TranAD" --model-hyper-params '{"anomaly_ratio": 25.0}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "label/TranAD"