python ./scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --data-name-list "Creditcard.csv" --model-name "pre_train.TimerModel" --model-hyper-params '{"seq_len": 672, "is_train": 1, "sampling_rate": 0.05}'  --adapter "PreTrain_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "score/Timer_Few"