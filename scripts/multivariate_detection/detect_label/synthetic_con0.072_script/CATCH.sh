python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_multi_config.json" --data-name-list "synthetic_con0.072.csv" --model-name "self_impl.CATCH" --model-hyper-params '{"Mlr": 0.05, "auxi_lambda": 0.1, "batch_size": 128, "cf_dim": 8, "d_ff": 32, "d_model": 32, "dc_lambda": 0.3, "dropout": 0.05, "e_layers": 1, "head_dim": 32, "head_dropout": 0.1, "lr": 0.0005, "n_heads": 16, "num_epochs": 2, "patch_size": 16, "patch_stride": 8, "score_lambda": 0.001, "seq_len": 192, "anomaly_ratio": 3.0}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "label/CATCH"