python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_multi_config.json" --data-name-list "SWAN.csv" --model-name "time_series_library.DLinear" --model-hyper-params '{"batch_size": 128, "d_ff": 8, "d_model": 8, "e_layers": 3, "horizon": 0, "norm": true, "num_epochs": 3, "seq_len": 100, "anomaly_ratio": 25.0}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "label/DLinear" --adapter "transformer_adapter"