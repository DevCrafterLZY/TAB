python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_multi_config.json" --data-name-list "synthetic_tre0.0778.csv" --model-name "LLM.GPT4TSModel" --model-hyper-params '{"seq_len": 100, "sampling_rate": 1}'  --adapter "llm_adapter"  --gpus 0  --num-workers 1  --timeout 60000  --save-path "label/GPT4TS_Full"