# TAB: Unified Benchmarking of Time Series Anomaly Detection Methods

## Introduction

TAB is an open-source library designed for time series anomaly detection researchers.

We provide a clean codebase for end-to-end evaluation of time series anomaly detection models, comparing their performance with baseline algorithms under various evaluation strategies and metrics.

The below figure provides a visual overview of TAB's pipeline.

<div align="center">
    <img src="docs/TAB_pipeline.png" alt="TAB_pipeline"  width="75%" />
</div>


The table below provides a visual overview of how TAB's key features compare to other libraries for time series anomaly detection.

<div align="center">
    <img src="docs/feature.png" alt="TAB_pipeline"  width="80%" />
</div>

## Quickstart

### Installation
Follow the steps below to configure the TAB environment.
#### 1. Install Git
Some models require Git to run. Please ensure that Git is installed on your system. If not, you can install it using the following command:
```shell
apt-get install git
```
#### 2. Download and Extract Environment
Download the environment from [Google Drive](https://drive.google.com/file/d/1PacXawsWoAVgifapUvFvAK7VnOJgc6FZ/view?usp=drive_link). (This may take some time, please wait patiently.) Then, create a directory my_env (we recommend creating it under `conda/envs/`) and extract the environment into this directory:
```shell
mkdir -p my_env
tar -xzf tab.tar.gz -C my_env
```
#### 3. Activate the Environment
```shell
source my_env/bin/activate
```
#### 4. Clean Environment Prefix
You can clear the environment prefix using the following command. Now the environment is the same as one created directly with conda at this path.
```shell
(my_env) $ conda-unpack
```

### Data preparation

Prepare Data. You can obtain the well pre-processed datasets from [Google Drive](https://drive.google.com/file/d/1_aW2AYwXtGk7ArjDjcSazqD0N9zp-zXx/view?usp=sharing). Then place the downloaded data under the folder `./dataset`. 

### Checkpoints preparation
You can download the checkpoints from [Google Drive](https://drive.google.com/file/d/1cPona0MKVls4ZXsDXq6URsb5TePUS48B/view?usp=drive_link). After obtaining the files, follow these steps to organize them:  

1. Move the files from `checkpoints/llm_checkpoints` to the folder `ts_benchmark/baselines/LLM/checkpoints`.
2. Move the files from `checkpoints/pre_train_checkpoints` to the directory `TAB/ts_benchmark/baselines/pretrain/checkpoints`.  

Ensure the files are placed in the correct directories for proper functionality.

### Train and evaluate model

We provide the experiment scripts for all benchmarks under the folder `./scripts/multivariate_detection`, and `./scripts/univariate_detection`. For example, you can reproduce an experiment result as the following:

```shell
sh ./scripts/multivariate_detection/detect_label/MSL_script/ModernTCN.sh
```

### Full experimental results
<div align="center">
    <img src="docs/results/table1.png" alt="table1"/>
</div>
<div align="center">
    <img src="docs/results/table2.png" alt="table2"/>
</div>
<div align="center">
    <img src="docs/results/table3.png" alt="table3"/>
</div>
<div align="center">
    <img src="docs/results/table4.png" alt="table4"/>
</div>
<div align="center">
    <img src="docs/results/table5.png" alt="table5"/>
</div>
<div align="center">
    <img src="docs/results/table6.png" alt="table6"/>
</div>
<div align="center">
    <img src="docs/results/table7.png" alt="table7"/>
</div>
<div align="center">
    <img src="docs/results/table8.png" alt="table8"/>
</div>
<div align="center">
    <img src="docs/results/table9.png" alt="table9"/>
</div>
<div align="center">
    <img src="docs/results/table10.png" alt="table10"/>
</div>
<div align="center">
    <img src="docs/results/table11.png" alt="table11"/>
</div>
<div align="center">
    <img src="docs/results/table12.png" alt="table12"/>
</div>
<div align="center">
    <img src="docs/results/table13.png" alt="table13"/>
</div>
<div align="center">
    <img src="docs/results/table14.png" alt="table14"/>
</div>
<div align="center">
    <img src="docs/results/table15.png" alt="table15"/>
</div>
<div align="center">
    <img src="docs/results/table16.png" alt="table16"/>
</div>
<div align="center">
    <img src="docs/results/table17.png" alt="table17"/>
</div>
<div align="center">
    <img src="docs/results/table18.png" alt="table18"/>
</div>
<div align="center">
    <img src="docs/results/table19.png" alt="table19"/>
</div>
<div align="center">
    <img src="docs/results/table20.png" alt="table20"/>
</div>
<div align="center">
    <img src="docs/results/table21.png" alt="table21"/>
</div>
<div align="center">
    <img src="docs/results/table22.png" alt="table22"/>
</div>
<div align="center">
    <img src="docs/results/table23.png" alt="table23"/>
</div>
<div align="center">
    <img src="docs/results/table24.png" alt="table24"/>
</div>
<div align="center">
    <img src="docs/results/table25.png" alt="table25"/>
</div>
<div align="center">
    <img src="docs/results/table26.png" alt="table26"/>
</div>
<div align="center">
    <img src="docs/results/table27.png" alt="table27"/>
</div>
<div align="center">
    <img src="docs/results/table28.png" alt="table28"/>
</div>
<div align="center">
    <img src="docs/results/table29.png" alt="table29"/>
</div>
<div align="center">
    <img src="docs/results/table30.png" alt="table30"/>
</div>
<div align="center">
    <img src="docs/results/table31.png" alt="table31"/>
</div>
<div align="center">
    <img src="docs/results/table32.png" alt="table32"/>
</div>
