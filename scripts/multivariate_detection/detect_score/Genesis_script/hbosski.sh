python ./scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --data-name-list "Genesis.csv" --model-name "tods.hbosski" --model-hyper-params '{"alpha": 1e-09, "contamination": 0.2, "n_bins": 40}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "score/hbosski"