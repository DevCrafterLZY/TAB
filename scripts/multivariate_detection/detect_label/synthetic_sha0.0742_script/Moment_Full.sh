python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_multi_config.json" --data-name-list "synthetic_sha0.0742.csv" --model-name "pre_train.Moment" --model-hyper-params '{"seq_len": 512, "is_train": 1, "sampling_rate": 1}'  --adapter "PreTrain_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "label/Moment_Full"