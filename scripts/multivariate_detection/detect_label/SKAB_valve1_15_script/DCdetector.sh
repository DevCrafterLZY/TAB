python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_multi_config.json" --data-name-list "SKAB_valve1_15.csv" --model-name "self_impl.DCdetector" --model-hyper-params '{"batch_size": 256, "num_epochs": 2, "patch_size": [5, 7], "win_size": 105, "anomaly_ratio": 0.5}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "label/DCdetector"