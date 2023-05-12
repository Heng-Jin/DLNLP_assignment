# DLNLP_assignment
The program code of ELEC0141 DLNLP, Heng Jin, SN22212102

### This is a multi-calssification task using 'News Category Dataset' in Kaggle
Dataset link:<https://www.kaggle.com/datasets/rmisra/news-category-dataset>

The cleaned dataset has been generated and saved as Kaggle_new_train.csv and Kaggle_new_test.csv. 
The categories are mapped to an integer from 0 to 42.
The samples with missing labels were discarded. The dataset
is divided into training set and test set according to the ratio
of 8 to 2.

### LSTM and BERT is implemented and fused in this tasks.  
- LSTM model: 12 layers LSTM + MLP classifier
- BERT model: BERT-base + MLP classifier
- BERT+LSTM model: BERT-base + 2 layers LSTM + MLP classifier


### Python libraries used
- pytorch
- transformers
- numpy
- matplotlib
- pandas
- json
- csv
- pathlib
- logging
- seaborn
- sklearn

All these are frequently used libraries, you can install them manually.
Or run the command : pip install -r requirements.txt to on python 3.6 
virtual environment to configure relevant libraries.

For pytorch, please check official website to install it correctly.
<https://pytorch.org/get-started/locally/>

### run the code

Before the training for LSTMs, run embeding.py to generate token embeddings for LSTM input, which are the same token embeddings used by BERT.

To run the training code of LSTM in original dataset: python main.py

To run the training code of BERT in original dataset: python train_bert.py



### Program run instruction
The main.py/train_modelname_suffix.py defines the training and validation pipeline of the model. 
The inference of the validation dataset will be implemented after each 
training epoch. The model will be saved after each epoch as well. 
All the outputs of each training will be saved into a separate folder.

&emsp; main.py : TrainLSTM in original dataset

The models.py defines the model structure of each model.

The dataset.py defines the Class Dataset to offer data pair for train and validation.

The embeding.py is 

### Results Display


