python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_multi_config.json" --data-name-list "SMD.csv" --model-name "merlion.AutoEncoder" --model-hyper-params '{"hidden_size": 5, "lr": 5e-05, "num_epochs": 10}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "label/AutoEncoder"