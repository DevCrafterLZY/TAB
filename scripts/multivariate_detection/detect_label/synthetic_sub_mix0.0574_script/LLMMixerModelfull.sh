python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_multi_config.json" --data-name-list "synthetic_sub_mix0.0574.csv" --model-name "LLM.LLMMixerModel" --model-hyper-params '{"d_model": 32, "dataset": "synthetic", "horizon": 1, "lr": 0.001, "n_heads": 4, "norm": true, "sampling_rate": 1, "seq_len": 96, "use_norm": 1, "anomaly_ratio": 5.0}' --adapter "llm_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "label/LLMMixerModelfull"
