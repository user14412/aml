我现在在做我的机器学习课程大作业。下面是我的老师在本次作业中给我的要求的部分截取，请你务必仔细阅读。这个文件夹就是用来跑实验的，里面现在主要有benchmark的数据集。我现在还没写代码，未来我需要你的协助来写代码、跑实验、收集数据、验证想法。而你现在的主要任务是确保理解我目前面临的任务。懂了吗？
Project Requirements

This project aims to design a self-supervised or weakly-supervised tabular machine learning method based on the TableShift benchmark, in order to improve model generalization under Out-of-Distribution (OOD) settings. The focus of this project is not merely to compare existing approaches, but to propose and implement a novel self-supervised or weakly-supervised mechanism (e.g., Test-Time Adaptation, TTA) that mitigates performance degradation caused by distribution shifts, without relying on target-domain labels.

The core research objectives include:

Designing effective self-supervised or weakly-supervised learning signals;
Improving model robustness under strict OOD settings;
Analyzing the applicability and limitations of the proposed method under different types of distribution shifts.
I. Dataset Requirements

This project will be conducted on the TableShift benchmark. TableShift is a standardized benchmark designed for evaluating OOD generalization in tabular data. It includes multiple real-world datasets and provides clearly defined In-Distribution (ID) and Out-of-Distribution (OOD) splits.

All experiments must strictly follow the official train/validation/test splits. Under the standard OOD setting:

OOD test labels must NOT be used for training or hyperparameter tuning.
The required datasets are:

assistments
nhanes_lead
brfss_diabetes
acsfoodstamps
physionet
acsunemployment
For details, refer to the official TableShift GitHub: https://github.com/mlfoundations/tableshift

For convenience, we provide the data download link: https://box.nju.edu.cn/d/958c50ca9223485eadac/ password: will be provided via QQ group.
Category 2: Test-Time Adaptation (TTA) Methods

Core idea: Adapt the model at test time using unlabeled target-domain data to mitigate distribution shifts.

Data constraints:

Training stage: ID training data (with labels only)
Test stage: Access to OOD test features (without labels)
Strictly forbidden: OOD test labels
You must clearly specify:

Whether model parameters are updated at test time;
Which parameters are updated (e.g., BatchNorm layers, classifier head, or full model);
The self-supervised objective used for adaptation.
Example directions:

Entropy minimization
Consistency regularization
Feature statistics matching
Pseudo-label self-training
Feature distribution alignment for tabular data
Special attention should be given to the non-exchangeability of tabular features when designing TTA strategies.

III. Experimental Report

(1) Method Design

Motivation
Self-supervised / weakly-supervised mechanism
Mathematical formulation
(2) Experimental Setup

Dataset description
Model architecture
Training procedure (single-stage or two-stage)
Whether parameters are updated at test time
Hyperparameter search strategy
(3) Experimental Results

Report OOD performance of your method and at least three baselines:

Accuracy
Balanced Accuracy
F1-score
Additionally report:

ID vs OOD performance comparison
Generalization Gap
Mean and standard deviation over at least three runs
IV. Experimental Protocol

Fix random seeds for reproducibility;
Tune hyperparameters only on validation sets;
Report average results over at least three independent runs;
Clearly specify training details (optimizer, learning rate, batch size, number of epochs, etc.).





---



懂了。

你当前的任务不是单纯跑几个现成模型，而是要基于 **TableShift benchmark** 设计一个有新意的 **自监督或弱监督表格 OOD 泛化方法**，重点方向可以是 **Test-Time Adaptation, TTA**。核心约束我理解为：

- 只能用 **ID train labels** 训练。
- 超参数只能用 **validation set** 调。
- 测试阶段可以访问 **OOD test features**，但绝对不能用 **OOD labels** 做训练、调参或适配。
- 最终要在 6 个数据集上实验：assistments、nhanes_lead、brfss_diabetes、acsfoodstamps、physionet、acsunemployment。
- 至少和 3 个 baseline 比较，报告 Accuracy、Balanced Accuracy、F1-score、ID vs OOD、Generalization Gap，并且至少 3 个随机种子的均值和标准差。
- 方法报告里必须说清楚：是否 test-time 更新参数、更新哪些参数、用什么自监督目标、为什么适合 tabular shift，尤其要注意表格特征不可交换。

我也快速看了当前文件夹，6 个 TableShift 数据集都已经在 C:\Users\zanyan\Desktop\AMLas\TableShift Dataset 里，每个数据集都有 Xtrain/Xval/Xidtest/Xood 和对应的 ytrain/yval/yidtest/yood。也就是说后续可以直接围绕这些官方 split 搭实验管线。

我会把这件事当成一个完整研究项目来协助你：后面可以一起做方法设计、baseline、TTA 机制、训练脚本、评估脚本、结果表格、消融实验和报告写作。当前理解到位。