python ./scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --data-name-list "Exathlon_4_1_100000_61-32.csv" --model-name "self_impl.TranAD" --model-hyper-params '{"batch_size": 32}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "score/TranAD"