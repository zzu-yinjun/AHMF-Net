# Echo Depth Estimation via Attention based Hierarchical Multi-scale Feature Fusion Network
This repository contains snippets of test code related to [AHMF-net] that are used to demonstrate and validate the methods mentioned in the paper. To protect the integrity of the project and sensitive information, we have not released the full research code.We provide the test code of the model to verify the accuracy of the model

# Dataset
Replica-VisualEchoes can be obatined from  [here](https://github.com/facebookresearch/VisualEchoes). We have used the 128x128 image resolution for our experiment.
MatterportEchoes is an extension of existing [matterport3D](https://niessner.github.io/Matterport/) dataset. In order to obtain the raw frames please forward the access request acceptance from the authors of matterport3D dataset. We will release the procedure to obtain the frames and echoes using habitat-sim and soundspaces in near future.

The BatVision dataset is separated in two parts: BatVision V1, recorded at UC Berkely and BatVision V2, recorded at Ecole des Mines de Paris. While BV1 contains more data, BV2 contains more complex scenes featuring a wide variety of material, room shapes and objects (including a few outdoor data).

Binaural echoes are 0.5s long and sampled at 44,1kHz. They are synchronized with corresponding RGB-D images.
Batvision V1 and BatVision V2 have csv files necessary to split data in train, val and test.
All data of BatVision V1 are listed in BatvisionV1/train.csv, val.csv and test.csv. In BatVision V2, each location is stored in separate folders containing train.csv, val.csv and test.csv.
To get more information about the data and data collection, please check out [Batvision datasets](https://cloud.minesparis.psl.eu/index.php/s/qurl3oySgTmT85M).

# Evaluation
Configure the relevant yml files before testing.We will give the pre-trained model parameters [here](https://drive.google.com/file/d/1BiNgFQNvO8n4_RZGusPzk4qksGiGQgX6/view?usp=drive_link)
```
pip install requirements.txt -r
python test.py
```
