python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_multi_config.json" --data-name-list "PSM.csv" --model-name "self_impl.TranAD" --model-hyper-params '{"batch_size": 128, "lr": 0.0001, "n_window": 8, "num_epochs": 5, "patience": 3, "anomaly_ratio": 10.0}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "label/TranAD"