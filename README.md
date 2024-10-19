# CUNSB-RFIE: Context-aware Unpaired Neural Schr\"{o}dinger Bridge in Retinal Fundus Image Enhancement
Retinal fundus photography is significant in diagnosing and monitoring retinal diseases. However, systemic imperfections and operator/patient-related factors can hinder the acquisition of high-quality retinal images. Previous efforts in retinal image enhancement primarily relied on GANs, which are limited by the trade-off between training stability and output diversity. In contrast, the Schr\"{o}dinger Bridge (SB), offers a more stable solution by utilizing Optimal Transport (OT) theory to model a stochastic differential equation (SDE) between two arbitrary distributions. This allows SB to effectively transform low-quality retinal images into their high-quality counterparts. In this work, we leverage the SB framework to propose an image-to-image translation pipeline for retinal image enhancement. Additionally, previous methods often fail to capture fine structural details, such as blood vessels. To address this, we enhance our pipeline by introducing Dynamic Snake Convolution, whose tortuous receptive field can better preserve tubular structures. We name the resulting retinal fundus image enhancement framework the Context-aware Unpaired Neural Schr\"{o}dinger Bridge (CUNSB-RFIE). To the best of our knowledge, this is the first endeavor to use the SB approach for retinal image enhancement. Experimental results on a large-scale dataset demonstrate the advantage of the proposed method compared to several state-of-the-art supervised and unsupervised methods in terms of image quality and performance on downstream tasks. 

## Model Architecture

![Model Overview](https://github.com/Retinal-Research/CUNSB-RFIE/blob/main/images/network_structure.png)

## Enhancement  Result Illustration. 

![Results](https://github.com/Retinal-Research/CUNSB-RFIE/blob/main/images/Eye_Q_generated.png)

## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

- [Git](https://git-scm.com)
- [Python](https://www.python.org/downloads/) and [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) 

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/Retinal-Research/CUNSB-RFIE.git

2. Create a Python Environment and install the required libraries by running
   ```sh
   conda env create -f environment.yml

## Downloading EyeQ Dataset 

The original EyeQ dataset can be downloaded by following the instructions provided [here](https://github.com/HzFu/EyeQ). The synthetic degraded images were generated using the algorithms described [here](https://github.com/joanshen0508/Fundus-correction-cofe-Net).

## Pretrained weight on EyeQ Dataset
The pre-trained weights on EyeQ Dataset can be got in folder **./pretrained**.  


## Training / testing on Customized Dataset
To train from scratch on a custom dataset, first create a directory named **datasets** to store your data. Organize the data in the following format: **Phase (train, test, val) A or B** (e.g., trainA, testB, valB). Once organized, start the training process by running the following command:
```sh
bash run_train.sh
```
To test your custom dataset, run the testing script:
```sh
bash run_test.sh
```
All arguments for training and testing are stored in the **options** folder and **./models/sb_model.py**.

### Thanks for the code provided by:

UNSB: https://github.com/cyclomon/UNSB
DSCNet: https://github.com/yaoleiqi/dscnet


