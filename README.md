## Attribution
This repository is based on the ProcessTransformer library by Buksh et al. Citation: 
Zaharah A. Bukhsh, Aaqib Saeed, & Remco M. Dijkman. (2021). "ProcessTransformer: Predictive Business Process Monitoring with Transformer Network". arXiv preprint arXiv:2104.00721

## Information
After every model execution, a number of average measures are calculated.
The averages are calculated for all examples as a whole, as well as for every prefix length. 
The measures are accuracy, f1-score, precision and recall. 

Furthermore, each model contains three callback functions:
    - CSVLogger: saves training and validation scores (accuracy and loss) after every epoch
    - Tensorboard: logs training and validation scores for automatic dashboarding on Tensorboard
    - EarlyStopping: stops training if the validation loss did not improve in the last five epochs

## Instructions
### Preprocessing
The ```data_processing.py``` script requires an event log file in csv format.
In order to use the script for outcome prediction, make sure to add a column with outcome labels (column name: 'outcome').
```
python data_processing.py --dataset=bpic2012 --raw_log_file=./datasets/bpic2012/bpic2012.csv --task=outcome_prediction
python data_processing.py --dataset=bpic2012 --raw_log_file=./datasets/bpic2012/bpic2012.csv --task=next_activity
```

### Experiment 1: model comparison for outcome prediction
```
python outcome_transformer.py --dataset=bpic2012
python outcome_lstm.py --dataset=bpic2012
python outcome_cnn.py --dataset=bpic2012
```

### Experiment 2: transferring learned representations
```
python next_activity.py --dataset=bpic2012
python outcome_pretrained.py --dataset=bpic2012
python outcome_transformer.py --dataset=bpic2012
```



 