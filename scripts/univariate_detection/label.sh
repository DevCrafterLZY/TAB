python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_config.json" --data-set-name "large_detect" --model-name "tods.lofski" --gpus 0  --num-workers 1  --timeout 60000  --save-path "single/label/tods_lofski"

python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_config.json" --data-set-name "large_detect" --model-name "tods.hbosski" --model-hyper-params '{}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "single/label/hbosski"

python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_config.json" --data-set-name "large_detect" --model-name "merlion.ArimaDetector" --model-hyper-params '{"max_forecast_steps": 24, "enable_threshold": 0}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "single/label/ArimaDetector"

python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_config.json" --data-set-name "large_detect" --model-name "merlion.SarimaDetector" --model-hyper-params '{"max_forecast_steps": 24, "enable_threshold": 0}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "single/label/SarimaDetector"

python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_config.json" --data-set-name "large_detect" --model-name "merlion.StatThreshold" --model-hyper-params '{}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "single/label/StatThreshold"

python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_config.json" --data-set-name "large_detect" --model-name "merlion.ZMS" --model-hyper-params '{}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "single/label/ZMS"

python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_config.json" --data-set-name "large_detect" --model-name "tods.ocsvmski" --model-hyper-params '{}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "single/label/ocsvmski"

python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_config.json" --data-set-name "large_detect" --model-name "merlion.DeepPointAnomalyDetector" --model-hyper-params '{"enable_threshold": 0}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "single/label/DeepPointAnomalyDetector"

python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_config.json" --data-set-name "large_detect" --model-name "tods.knnski" --gpus 0  --num-workers 1  --timeout 60000  --save-path "single/label/tods_knnski"

python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_config.json" --data-set-name "large_detect" --model-name "merlion.IsolationForest" --model-hyper-params '{}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "single/label/IsolationForest"

python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_config.json" --data-set-name "large_detect" --model-name "merlion.SpectralResidual" --model-hyper-params '{}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "single/label/SpectralResidual"

python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_config.json" --data-set-name "large_detect" --model-name "tods.lodaski" --model-hyper-params '{}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "single/label/lodaski"

python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_config.json" --data-set-name "large_detect" --model-name "tods.pcaodetectorski" --model-hyper-params '{}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "single/label/pcaodetectorski"

python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_config.json" --data-set-name "large_detect" --model-name "merlion.DAGMM" --gpus 0  --num-workers 1  --timeout 60000  --save-path "single/label/merlion_dagmm"

python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_config.json" --data-set-name "large_detect" --model-name "time_series_library.iTransformer" --model-hyper-params '{"batch_size": 16, "d_ff": 256, "d_model": 128, "e_layers": 1, "num_epochs": 3, "seq_len": 12, "horizon": 0}' --adapter "transformer_adapter" --gpus 0  --num-workers 1  --timeout 60000  --save-path "single/label/iTransformer"

python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_config.json" --data-set-name "large_detect" --model-name "time_series_library.NLinear" --model-hyper-params '{"batch_size": 128, "d_ff": 8, "d_model": 8, "e_layers": 3, "horizon": 0, "norm": true, "num_epochs": 3, "seq_len": 30}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "single/label/NLinear" --adapter "transformer_adapter"

python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_config.json" --data-set-name "large_detect" --model-name "time_series_library.DLinear" --model-hyper-params '{"batch_size": 128, "d_ff": 16, "d_model": 8, "e_layers": 1, "horizon": 0, "norm": true, "num_epochs": 1, "seq_len": 30}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "single/label/DLinear" --adapter "transformer_adapter"

python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_config.json" --data-set-name "large_detect" --model-name "merlion.AutoEncoder" --model-hyper-params '{"hidden_size": 5, "lr": 5e-05, "num_epochs": 10}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "single/label/AutoEncoder"

python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_config.json" --data-set-name "large_detect" --model-name "self_impl.ModernTCN" --model-hyper-params '{"seq_len": 12, "label_len": 0, "horizon": 12, "ffn_ratio": 1, "patch_size": 8, "patch_stride": 4, "num_blocks": [1], "large_size": [51], "small_size": [5], "dims": [8], "head_dropout": 0.1, "dropout": 0.1, "lr": 0.0001, "itr": 1, "batch_size": 16, "num_epochs": 2, "patience": 10, "use_multi_scale": false, "small_kernel_merged": false}' --gpus 0  --num-workers 1  --timeout 60000  --save-path "single/label/ModernTCN"

python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_config.json" --data-set-name "large_detect" --model-name "time_series_library.PatchTST" --model-hyper-params '{"batch_size": 128, "d_ff": 8, "d_model": 8, "e_layers": 3, "horizon": 0, "norm": true, "num_epochs": 3, "seq_len": 30}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "single/label/PatchTST" --adapter "transformer_adapter"

python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_config.json" --data-set-name "large_detect" --model-name "time_series_library.TimesNet" --model-hyper-params '{"batch_size": 64, "d_ff": 8, "d_model": 8, "e_layers": 3, "horizon": 0, "norm": true, "num_epochs": 3, "seq_len": 30}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "single/label/TimesNet" --adapter "transformer_adapter"

python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_config.json" --data-set-name "large_detect" --model-name "self_impl.AnomalyTransformer" --model-hyper-params '{"batch_size": 128, "lr": 0.0001, "num_epochs": 3, "win_size": 50}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "single/label/AnomalyTransformer"

python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_config.json" --data-set-name "large_detect" --model-name "merlion.VAE" --model-hyper-params '{}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "single/label/VAE"

python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_config.json" --data-set-name "large_detect" --model-name "self_impl.TranAD" --model-hyper-params '{"lr": 0.0001, "batch_size": 128, "num_epochs": 5, "n_window": 8, "patience": 3}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "single/label/TranAD"

python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_config.json" --data-set-name "large_detect" --model-name "self_impl.DualTF" --model-hyper-params '{"batch_size": 8, "fre_anomaly_ratio": 5, "num_epochs": 3, "seq_len": 40, "nest_len": 20}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "single/label/DualTF"

python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_config.json" --data-set-name "large_detect" --model-name "merlion.LSTMED" --model-hyper-params '{}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "single/label/LSTMED"

python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_config.json" --data-set-name "large_detect" --model-name "self_impl.DCdetector" --model-hyper-params '{"batch_size": 64, "num_epochs": 3, "patch_size": [3, 5], "win_size": 30}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "single/label/DCdetector"
