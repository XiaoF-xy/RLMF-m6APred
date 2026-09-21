# RLMF-m6APred

[English](README.md) | [简体中文](README_zh-CN.md)

RLMF-m6APred用于预测41-nt RNA序列中的m6A修饰位点。本仓库提供推理代码和复现独立测试集结果所需的数据。已发布的模型权重托管于[Hugging Face](https://huggingface.co/Yu-star/RLMF-m6APred)。

**在线预测服务器：** [https://xiaof-xy--rlmf.modal.run](https://xiaof-xy--rlmf.modal.run)

## 1. 环境配置

论文中的实验在配备3张NVIDIA A10 GPU（每张24 GB显存）的工作站上完成。复现已发布的测试结果只需要执行推理，可使用一张CUDA GPU，也支持CPU运行。

```bash
conda env create -f environment.yml
conda activate rlmf_m6apred
```

## 2. 下载已发布的模型权重

运行以下命令，将领域适配后的RNA语言模型和全部55个五折检查点从Hugging Face下载到程序所需目录：

```bash
python scripts/download_model.py
```

下载后的主要目录结构如下：

```text
RLMF-m6APred/
|-- pretrained/rna_language_model/
|-- weights/
|   |-- H_b/fold_01.pt ... fold_05.pt
|   |-- ...
|   `-- R_l/fold_01.pt ... fold_05.pt
|-- data/independent_test/
`-- evaluate_independent.py
```

## 3. 复现独立测试集结果

评估一个数据集：

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_independent.py --dataset H_b
```

评估全部11个数据集：

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_independent.py --all
```

若使用CPU推理，省略`CUDA_VISIBLE_DEVICES=0`即可。程序对五折模型输出的概率取平均值，并将ACC、MCC、AUC、AUPRC、F1、Precision、Recall和Specificity写入`reproduced_results/`目录。

## 4. 评估其他带标签的数据集

输入CSV文件必须包含`sequence`和`label`两列：

```bash
CUDA_VISIBLE_DEVICES=0 python predict.py \
  --dataset H_b \
  --input path/to/data.csv
```

模型接收长度为41 nt的序列。输入RNA中的`U`会在预处理阶段自动转换为`T`。
