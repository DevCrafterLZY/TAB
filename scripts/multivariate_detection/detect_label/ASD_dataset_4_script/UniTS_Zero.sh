python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_multi_config.json" --data-name-list "ASD_dataset_4.csv" --model-name "pre_train.UniTS" --model-hyper-params '{"seq_len": 96, "target_dim": 19}' --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "label/UniTS_Zero"