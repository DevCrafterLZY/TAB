python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_multi_config.json" --data-name-list "SWAT.csv" --model-name "time_series_library.DLinear" --model-hyper-params '{"anomaly_ratio": 10.0, "batch_size": 16, "d_ff": 64, "d_model": 64, "e_layers": 2, "horizon": 0, "norm": true, "num_epochs": 1, "seq_len": 50}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "label/DLinear" --adapter "transformer_adapter"