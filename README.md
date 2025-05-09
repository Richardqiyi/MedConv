# MedConv
This is the code repository for the paper:
> **MedConv: Convolutions Beat Transformers on Long-Tailed Bone Density Prediction**
> 
> [Xuyin Qi](https://au.linkedin.com/in/xuyin-q-29672524a)\*, [Zeyu Zhang](https://steve-zeyu-zhang.github.io/)\*†, Huazhan Zheng\*, Mingxi Chen, Numan Kutaiba, Ruth Lim, Cherie Chiang, Zi En Tham, Xuan Ren, Wenxin Zhang, Lei Zhang, Hao Zhang, Wenbing Lv, Guangzhen Yao, Renda Han, Kangsheng Wang, Mingyuan Li, Hongtao Mao, Yu Li, [Zhibin Liao](https://researchers.adelaide.edu.au/profile/zhibin.liao), [Yang Zhao](https://yangyangkiki.github.io/)\**, [Minh-Son To](https://scholar.google.com.au/citations?user=NIc4qPsAAAAJ&hl=en)
>
> \*Equal contribution. \**Corresponding author. †Project lead.
>
> <em><b>IJCNN 2025</b></em>
> 
> [**[arXiv]**](https://arxiv.org/abs/2502.00631) [**[Paper with Code]**](https://paperswithcode.com/paper/medconv-convolutions-beat-transformers-on) **[[HF Paper]](https://huggingface.co/papers/2502.00631)**

![framework.](https://github.com/Richardqiyi/MedConv/blob/main/main-graph.png)

## Citation

```
@article{qi2025medconv,
  title={MedConv: Convolutions Beat Transformers on Long-Tailed Bone Density Prediction},
  author={Qi, Xuyin and Zhang, Zeyu and Zheng, Huazhan and Chen, Mingxi and Kutaiba, Numan and Lim, Ruth and Chiang, Cherie and Tham, Zi En and Ren, Xuan and Zhang, Wenxin and others},
  journal={arXiv preprint arXiv:2502.00631},
  year={2025}
}
```

## Introduction

Bone health, crucial for mobility, fracture prevention, and overall well-being, is particularly important for aging populations or those with osteoporosis, a common skeletal disease that compromises bone strength by causing low bone mass and microarchitectural deterioration. This condition, increasing the risk of fragility fractures from low-energy impacts, often affects critical areas like the spine, hip, and wrist, significantly reducing quality of life. Predicting bone density through CT scans to estimate T-scores offers a more precise and detailed assessment of bone health compared to traditional methods like X-ray bone density tests, which have lower spatial resolution and limited ability to detect localized bone changes. CT-based assessments can measure volumetric bone mineral density (BMD) and provide three-dimensional imaging, allowing for a comprehensive evaluation of bone quality. Studies have demonstrated that deep learning models applied to CT images can accurately predict BMD and T-scores, enhancing the detection and management of osteoporosis. Additionally, quantitative computed tomography (QCT) has been shown to be a superior method for diagnosing osteoporosis and predicting fractures when compared to dual-energy X-ray absorptiometry (DXA). These advancements highlight the potential of CT imaging in providing detailed insights into bone health, surpassing the capabilities of traditional X-ray-based methods. Recent advances in representation learning and dense prediction, particularly in the domain of medical imaging, have significantly enhanced the accuracy and automation of osteoporosis detection. These advancements facilitate early diagnosis and timely intervention, providing a foundation for more effective personalized treatment and prevention strategies.

![pipeline.](https://github.com/Richardqiyi/MedConv/blob/main/Comparison_of_segmentators.png)


## Environment Setup
Python = 3.7, Pytorch = 1.8.0, Torchvision = 1.9.0, CUDA = 11.1

## GPU
1 NVIDIA RTX A6000

## Docker
```
docker pull qiyi007/oct:1.0
```

## Dataset
```
|---MedConv
|---|---train_data <NIFTI file>
|-------|---...
|---|---val_data <NIFTI file>
|-------|---...
|---|---test_data <NIFTI file>
|-------|---...
|---|---labels.xlsx
```
## Train
```
python train.py --save_dir <Directory to save model checkpoints> --logit_adj_post <adjust logits post hoc, default 1> --logit_adj_train <adjust logits in training, default 1>
```
## Test
```
python test.py
```
It will test all choices of tro.







