python ./scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --data-name-list "synthetic_glo0.048.csv" --model-name "LLM.GPT4TSModel" --model-hyper-params '{"seq_len": 75, "sampling_rate": 0.05}'  --adapter "llm_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "score/GPT4TS_Few"