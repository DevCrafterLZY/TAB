python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_multi_config.json" --data-name-list "SKAB_other_7.csv" --model-name "self_impl.TranAD" --model-hyper-params '{"batch_size": 32, "anomaly_ratio": 3.0}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "label/TranAD"