python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_multi_config.json" --data-name-list "SKAB_valve1_4.csv" --model-name "self_impl.TranAD" --model-hyper-params '{"batch_size": 32, "anomaly_ratio": 10.0}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "label/TranAD"