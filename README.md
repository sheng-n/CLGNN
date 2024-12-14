# CLGNN
## 1. Overview
The code for paper xx". The repository is organized as follows:

+ `data/` contains the dataset in the paper;
  * `CTD_chemicals_diseases.xlsx` contain known chemical-disease associations collected and processed from the CTD database;
  * `RDA.edgelist` contain known chemical-disease associations pairs, respectively;
  * `non_RDA.edgelist` contain unknown chemical-disease associations pairs;
  * `chemical_name.xlsx` contain chemical names;
  * `disease_name.xlsx` contains disease names;
  
+ `code/`
  * `parms_setting.py`contains hyperparmeters;
  * `utils.py` contains preprocessing function of the data;
  * `data_preprocess.py` contains the preprocess of data;
  * `layer.py` contains CLGNN's model layer;
  * `train.py` contains training and testing code;
  * `main.py` runs CLGNN;

## 2. Dependencies
* numpy == 1.21.1
* torch == 2.0.0+cu118
* sklearn == 0.24.1
* torch-geometric == 2.3.0

## 3. Quick Start
Here we provide a example to predict the chemical-disease association scores:

1. Download and upzip our data and code files
2. Run main.py

## 4. Contacts
If you have any questions, please email Nan Sheng (shengnan@jlu.edu.cn)
