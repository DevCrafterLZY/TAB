python ./scripts/run_benchmark.py --config-path "unfixed_detect_score_config.json" --data-set-name "large_detect" --model-name "pre_train.UniTS" --model-hyper-params '{"num_epochs": 3, "seq_len": 96, "sampling_rate": 0.05, "is_train": 1}' --adapter "PreTrain_adapter" --gpus 0  --num-workers 1  --timeout 60000  --save-path "single/score/Units_Few"

python ./scripts/run_benchmark.py --config-path "unfixed_detect_score_config.json" --data-set-name "large_detect" --model-name "pre_train.TinyTimeMixer" --model-hyper-params '{"num_epochs": 3, "seq_len": 96, "sampling_rate": 0.05, "is_train": 1}' --adapter "PreTrain_adapter" --gpus 0  --num-workers 1  --timeout 60000  --save-path "single/score/TinyTimeMixer_Few"

python ./scripts/run_benchmark.py --config-path "unfixed_detect_score_config.json" --data-set-name "large_detect" --model-name "pre_train.TimerModel" --model-hyper-params '{"num_epochs": 3, "seq_len": 96, "sampling_rate": 0.05, "is_train": 1}' --adapter "PreTrain_adapter" --gpus 0  --num-workers 1  --timeout 60000  --save-path "single/score/TimerModel_Few"

python ./scripts/run_benchmark.py --config-path "unfixed_detect_score_config.json" --data-set-name "large_detect" --model-name "pre_train.Moment" --model-hyper-params '{"num_epochs": 3, "seq_len": 96, "sampling_rate": 0.05, "is_train": 1}' --adapter "PreTrain_adapter" --gpus 0  --num-workers 1  --timeout 60000  --save-path "single/score/Moment_Few"

python ./scripts/run_benchmark.py --config-path "unfixed_detect_score_config.json" --data-set-name "large_detect" --model-name "LLM.UniTimeModel" --model-hyper-params '{"num_epochs": 3, "seq_len": 96, "sampling_rate": 0.05, "max_token_num": 17, "stride": 16, "patch_size": 16, "dataset": "UV", "max_backcast_len": 96}' --adapter "llm_adapter" --gpus 0  --num-workers 1  --timeout 60000  --save-path "single/score/UniTimeModel_Few"

python ./scripts/run_benchmark.py --config-path "unfixed_detect_score_config.json" --data-set-name "large_detect" --model-name "LLM.GPT4TSModel" --model-hyper-params '{"num_epochs": 3, "seq_len": 100, "sampling_rate": 0.05}' --adapter "llm_adapter" --gpus 0  --num-workers 1  --timeout 60000  --save-path "single/score/GPT4TS_Few"

