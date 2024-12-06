python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_config.json" --data-set-name "large_detect" --model-name "pre_train.UniTS" --model-hyper-params '{"num_epochs": 3, "seq_len": 96, "sampling_rate": 0.05, "is_train": 1}' --adapter "PreTrain_adapter" --gpus 0  --num-workers 1  --timeout 60000  --save-path "single/label/Units_Few"

python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_config.json" --data-set-name "large_detect" --model-name "pre_train.TinyTimeMixer" --model-hyper-params '{"num_epochs": 3, "seq_len": 96, "sampling_rate": 0.05, "is_train": 1}' --adapter "PreTrain_adapter" --gpus 0  --num-workers 1  --timeout 60000  --save-path "single/label/TinyTimeMixer_Few"

python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_config.json" --data-set-name "large_detect" --model-name "pre_train.TimerModel" --model-hyper-params '{"num_epochs": 3, "seq_len": 96, "sampling_rate": 0.05, "is_train": 1}' --adapter "PreTrain_adapter" --gpus 0  --num-workers 1  --timeout 60000  --save-path "single/label/TimerModel_Few"

python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_config.json" --data-set-name "large_detect" --model-name "pre_train.Moment" --model-hyper-params '{"num_epochs": 3, "seq_len": 96, "sampling_rate": 0.05, "is_train": 1}' --adapter "PreTrain_adapter" --gpus 0  --num-workers 1  --timeout 60000  --save-path "single/label/Moment_Few"

python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_config.json" --data-set-name "large_detect" --model-name "LLM.UniTimeModel" --model-hyper-params '{"num_epochs": 3, "seq_len": 96, "sampling_rate": 0.05, "max_token_num": 17, "stride": 16, "patch_size": 16, "dataset": "UV", "max_backcast_len": 96}' --adapter "llm_adapter" --gpus 0  --num-workers 1  --timeout 60000  --save-path "single/label/UniTimeModel_Few"

python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_config.json" --data-set-name "large_detect" --model-name "LLM.GPT4TSModel" --model-hyper-params '{"num_epochs": 3, "seq_len": 100, "sampling_rate": 0.05}' --adapter "llm_adapter" --gpus 0  --num-workers 1  --timeout 60000  --save-path "single/label/GPT4TS_Few"

