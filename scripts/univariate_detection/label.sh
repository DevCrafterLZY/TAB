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

python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_config.json" --data-set-name "large_detect" --model-name "self_impl.Torsk" --model-hyper-params '{}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "single/label/Torsk"

python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_config.json" --data-set-name "large_detect" --model-name "self_impl.SAND" --model-hyper-params '{}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "single/label/SAND"

python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_config.json" --data-set-name "large_detect" --model-name "tods.cblofski" --model-hyper-params '{}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "single/label/cblofski"

python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_config.json" --data-set-name "large_detect" --model-name "self_impl.DWT_MLEAD" --model-hyper-params '{}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "single/label/DWT_MLEAD"

python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_config.json" --data-set-name "large_detect" --model-name "self_impl.EIF" --model-hyper-params '{}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "single/label/EIF"

python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_config.json" --data-set-name "large_detect" --model-name "self_impl.KMeans" --model-hyper-params '{}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "single/label/KMeans"

python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_config.json" --data-set-name "large_detect" --model-name "self_impl.LeftSTAMPi" --model-hyper-params '{}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "single/label/LeftSTAMPi"

python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_config.json" --data-set-name "large_detect" --model-name "self_impl.Series2Graph" --model-hyper-params '{}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "single/label/Series2Graph"

python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_config.json" --data-set-name "large_detect" --model-name "self_impl.ContraAD" --model-hyper-params '{}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "single/label/ContraAD"

python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_config.json" --data-set-name "large_detect" --model-name "pre_train.UniTS" --model-hyper-params '{"num_epochs": 3, "seq_len": 96, "sampling_rate": 1, "is_train": 1}' --adapter "PreTrain_adapter" --gpus 0  --num-workers 1  --timeout 60000  --save-path "single/label/Units_Full"

python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_config.json" --data-set-name "large_detect" --model-name "pre_train.TinyTimeMixer" --model-hyper-params '{"num_epochs": 3, "seq_len": 512, "sampling_rate": 1, "is_train": 1}' --adapter "PreTrain_adapter" --gpus 0  --num-workers 1  --timeout 60000  --save-path "single/label/TinyTimeMixer_Full"

python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_config.json" --data-set-name "large_detect" --model-name "pre_train.TimerModel" --model-hyper-params '{"num_epochs": 3, "seq_len": 96, "sampling_rate": 1, "is_train": 1}' --adapter "PreTrain_adapter" --gpus 0  --num-workers 1  --timeout 60000  --save-path "single/label/TimerModel_Full"

python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_config.json" --data-set-name "large_detect" --model-name "pre_train.Moment" --model-hyper-params '{"num_epochs": 3, "seq_len": 96, "sampling_rate": 1, "is_train": 1}' --adapter "PreTrain_adapter" --gpus 0  --num-workers 1  --timeout 60000  --save-path "single/label/Moment_Full"

python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_config.json" --data-set-name "large_detect" --model-name "LLM.UniTimeModel" --model-hyper-params '{"num_epochs": 3, "seq_len": 96, "sampling_rate": 1, "max_token_num": 17, "stride": 16, "patch_size": 16, "dataset": "UV", "max_backcast_len": 96}' --adapter "llm_adapter" --gpus 0  --num-workers 1  --timeout 60000  --save-path "single/label/UniTimeModel_Full"

python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_config.json" --data-set-name "large_detect" --model-name "LLM.GPT4TSModel" --model-hyper-params '{"num_epochs": 3, "seq_len": 100, "sampling_rate": 1}' --adapter "llm_adapter" --gpus 0  --num-workers 1  --timeout 60000  --save-path "single/label/GPT4TS_Full"

python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_config.json" --data-set-name "large_detect" --model-name "LLM.CALFModel" --model-hyper-params '{"d_ff": 768, "d_model": 768, "dropout": 0.3, "gpt_layer": 6, "horizon": 1, "lr": 0.0005, "n_heads": 4, "norm": true, "sampling_rate": 1, "seq_len": 96}' --adapter "llm_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "single/label/CALFModel_Full"

python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_config.json" --data-set-name "large_detect" --model-name "pre_train.Chronos" --model-hyper-params '{"horizon": 1, "is_train": 1, "norm": true, "sampling_rate": 1, "seq_len": 96, "anomaly_ratio": 0.1}' --adapter "chronos_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "single/label/Chronos_Full"

python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_config.json" --data-set-name "large_detect" --model-name "pre_train.DadaModel" --model-hyper-params '{"horizon": 1, "is_train": 1, "lr": 0.005, "norm": true, "sampling_rate": 1, "seq_len": 100, "anomaly_ratio": 0.1}' --adapter "PreTrain_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "single/label/DadaModel_Full"

python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_config.json" --data-set-name "large_detect" --model-name "LLM.LLMMixerModel" --model-hyper-params '{"d_model": 32, "horizon": 1, "lr": 0.001, "n_heads": 4, "norm": true, "sampling_rate": 1, "seq_len": 96, "use_norm": 1, "anomaly_ratio": 0.5}' --adapter "llm_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "single/label/LLMMixerModel_Full"

python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_config.json" --data-set-name "large_detect" --model-name "pre_train.TimesFM" --model-hyper-params '{"horizon": 1, "input_patch_len": 32, "is_train": 1, "norm": true, "output_patch_len": 128, "sampling_rate": 1, "seq_len": 96, "anomaly_ratio": 0.1}' --adapter "PreTrain_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "single/label/TimesFM_Full"
