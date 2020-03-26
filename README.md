This repository stores scripts implementing the algorithms described in the paper "A Learning Approach with Programmable Data Plane towards IoT Security" which is accepted by ICDCS 2020.

## Algorithm Stage I: Train a dilated 1D-CNN
### Usage:
```
python stage1_dilatedCNN.py [Dataset Name]
```
### Parameters:
 - **Dataset Name**: A text file in the ./dataset folder. Each line represents a packet with bytes converted into integers. We provide two datasets, "xbee"(default) and "rpl". The details can be seen in the original paper.
 
### Outputs:
 - **[Dataset Name]_cnn_model.h5**: A trained neural network in .h5 format.
 
## Algorithm Stage II: Calculate the P4 data plane definition
### Usage:
```
python stage2_pruning.py [Dataset Name] [Number of Fields] [Index of Layer]
```
### Parameters:
 - **Dataset Name**: Same as Stage I.
 - **Number of Fields**: An integer with default value 2.
 - **Index of Layer**: 0, 1 or 2(default). The length of each field will be 2^[Index of Layer].
 
### Outputs:
 - **[Dataset Name]_importance_scores.txt**: Importance scores of neurons in the first three layers calculated by the pruning algorithm.
 - **[Dataset Name]_P4_definition.txt**: Data plane definition in P4 language.
