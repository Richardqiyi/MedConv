# Medconv
This is the code repository for the paper:

## Introduction

Bone health, crucial for mobility, fracture prevention, and overall well-being, is particularly important for aging populations or those with osteoporosis, a common skeletal disease that compromises bone strength by causing low bone mass and microarchitectural deterioration. This condition, increasing the risk of fragility fractures from low-energy impacts, often affects critical areas like the spine, hip, and wrist, significantly reducing quality of life. Predicting bone density through CT scans to estimate T-scores offers a more precise and detailed assessment of bone health compared to traditional methods like X-ray bone density tests, which have lower spatial resolution and limited ability to detect localized bone changes. CT-based assessments can measure volumetric bone mineral density (BMD) and provide three-dimensional imaging, allowing for a comprehensive evaluation of bone quality. Studies have demonstrated that deep learning models applied to CT images can accurately predict BMD and T-scores, enhancing the detection and management of osteoporosis. Additionally, quantitative computed tomography (QCT) has been shown to be a superior method for diagnosing osteoporosis and predicting fractures when compared to dual-energy X-ray absorptiometry (DXA). These advancements highlight the potential of CT imaging in providing detailed insights into bone health, surpassing the capabilities of traditional X-ray-based methods. Recent advances in representation learning and dense prediction, particularly in the domain of medical imaging, have significantly enhanced the accuracy and automation of osteoporosis detection. These advancements facilitate early diagnosis and timely intervention, providing a foundation for more effective personalized treatment and prevention strategies.

## Environment Setup
```
# create a clean conda environment from scratch
conda create --name python=3.10
conda activate MedConv
# install pip
conda install ipython
conda install pip
# install required packages
pip install -r requirements.txt
```

