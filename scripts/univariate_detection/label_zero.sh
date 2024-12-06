python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_config.json" --data-set-name "large_detect" --model-name "pre_train.UniTS" --model-hyper-params '{"num_epochs": 3, "seq_len": 96}' --adapter "PreTrain_adapter" --gpus 0  --num-workers 1  --timeout 60000  --save-path "single/label/Units_Zero"

python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_config.json" --data-set-name "large_detect" --model-name "pre_train.TinyTimeMixer" --model-hyper-params '{"num_epochs": 3, "seq_len": 96}' --adapter "PreTrain_adapter" --gpus 0  --num-workers 1  --timeout 60000  --save-path "single/label/TinyTimeMixer_Zero"

python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_config.json" --data-set-name "large_detect" --model-name "pre_train.TimerModel" --model-hyper-params '{"num_epochs": 3, "seq_len": 96}' --adapter "PreTrain_adapter" --gpus 0  --num-workers 1  --timeout 60000  --save-path "single/label/TimerModel_Zero"

python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_config.json" --data-set-name "large_detect" --model-name "pre_train.Moment" --model-hyper-params '{"num_epochs": 3, "seq_len": 96}' --adapter "PreTrain_adapter" --gpus 0  --num-workers 1  --timeout 60000  --save-path "single/label/Moment_Zero"
