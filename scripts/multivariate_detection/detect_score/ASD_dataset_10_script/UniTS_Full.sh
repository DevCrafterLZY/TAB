python ./scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --data-name-list "ASD_dataset_10.csv" --model-name "pre_train.UniTS" --model-hyper-params '{"seq_len": 96, "is_train": 1, "sampling_rate": 1, "target_dim": 19}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "score/UniTS_Full"