# Dual-Attention-Person-Re-Id
Project Work - Person Reid by dual attention mechanism

### Data Preparation

Download and extract datasets iLIDS-VID, PRID2011 and MARS into the data/ directory. data/iLIDS-VID for example.

Modify and run data/computeOpticalFlow.m with Matlab to generate Optical Flow data. Optical Flow data will be generated in the same dir of your datasets. data/iLIDS-VID-OF-HVP for example.

### Training

Run this command for training iLIDS-VID.

python main.py


### Model explanation

Note that inside the model folder there are respectively 2 models. fcn32s.py and model2.py . This is because since we require to evaluate dual attention features we perform evaluation of INTER-ATTENTION between the intermediate features collected at the end of the encoder of FCN.
