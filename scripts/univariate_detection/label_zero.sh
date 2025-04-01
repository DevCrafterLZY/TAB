python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_config.json" --data-set-name "large_detect" --model-name "pre_train.UniTS" --model-hyper-params '{"num_epochs": 3, "seq_len": 96}' --adapter "PreTrain_adapter" --gpus 0  --num-workers 1  --timeout 60000  --save-path "single/label/Units_Zero"

python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_config.json" --data-set-name "large_detect" --model-name "pre_train.TinyTimeMixer" --model-hyper-params '{"num_epochs": 3, "seq_len": 512}' --adapter "PreTrain_adapter" --gpus 0  --num-workers 1  --timeout 60000  --save-path "single/label/TinyTimeMixer_Zero"

python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_config.json" --data-set-name "large_detect" --model-name "pre_train.TimerModel" --model-hyper-params '{"num_epochs": 3, "seq_len": 96}' --adapter "PreTrain_adapter" --gpus 0  --num-workers 1  --timeout 60000  --save-path "single/label/TimerModel_Zero"

python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_config.json" --data-set-name "large_detect" --model-name "pre_train.Moment" --model-hyper-params '{"num_epochs": 3, "seq_len": 96}' --adapter "PreTrain_adapter" --gpus 0  --num-workers 1  --timeout 60000  --save-path "single/label/Moment_Zero"

python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_config.json" --data-set-name "large_detect" --model-name "pre_train.Chronos" --model-hyper-params '{"horizon": 1, "is_train": 0, "norm": true, "sampling_rate": 0.05, "seq_len": 96, "anomaly_ratio": 0.1}' --adapter "chronos_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "single/label/Chronos_Zero"

python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_config.json" --data-set-name "large_detect" --model-name "pre_train.DadaModel" --model-hyper-params '{"horizon": 1, "is_train": 0, "lr": 0.005, "norm": true, "sampling_rate": 0.05, "seq_len": 100, "anomaly_ratio": 0.1}' --adapter "PreTrain_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "single/label/DadaModel_Zero"

python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_config.json" --data-set-name "large_detect" --model-name "pre_train.TimesFM" --model-hyper-params '{"horizon": 1, "input_patch_len": 32, "is_train": 0, "norm": true, "output_patch_len": 128, "sampling_rate": 0.05, "seq_len": 96, "anomaly_ratio": 0.1}' --adapter "PreTrain_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "single/label/TimesFM_Zero"
