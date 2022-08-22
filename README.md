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

## Tensorboard Training information

| Dataset   | Model       | Tensorboard                                                                             |
| --------- | ----------- | --------------------------------------------------------------------------------------- |
| BPIC2012a | Transformer | [tjH7ytA4SAqncKfznqNkQw](https://tensorboard.dev/experiment/tjH7ytA4SAqncKfznqNkQw)  |
|           | LSTM        | [2HahLuoqSAyQsX5tvLWKsA](https://tensorboard.dev/experiment/2HahLuoqSAyQsX5tvLWKsA)  |
|           | CNN         | [vVeADqw2RXi9zT8yQAPv9Q](https://tensorboard.dev/experiment/vVeADqw2RXi9zT8yQAPv9Q)  |
|           | Pre-trained | [0U5L2najRBC6dWtPgEJPgw](https://tensorboard.dev/experiment/0U5L2najRBC6dWtPgEJPgw)  |
|           | Benchmark   | [Gt8bhqPzRYGFEgQ6ZTlRug](https://tensorboard.dev/experiment/Gt8bhqPzRYGFEgQ6ZTlRug)  |
| BPIC2012c | Transformer | [GMuewOwZTaaR8cv4Vd30rA](https://tensorboard.dev/experiment/GMuewOwZTaaR8cv4Vd30rA)  |
|           | LSTM        | [skztVHQuRC20jFvKZuNDyA](https://tensorboard.dev/experiment/skztVHQuRC20jFvKZuNDyA)  |
|           | CNN         | [teSr5eAcQEeVomAjHk0KvQ](https://tensorboard.dev/experiment/teSr5eAcQEeVomAjHk0KvQ)  |
|           | Pre-trained | [Qfvz1ID5Txmk77vuGcKdhA](https://tensorboard.dev/experiment/Qfvz1ID5Txmk77vuGcKdhA)  |
|           | Benchmark   | [jzwxR49fQhKdyf7eRKsXsg](https://tensorboard.dev/experiment/jzwxR49fQhKdyf7eRKsXsg)  |
| BPIC2012d | Transformer | [Ur8RGK9VRkuxUEUgR6kzUA](https://tensorboard.dev/experiment/Ur8RGK9VRkuxUEUgR6kzUA)  |
|           | LSTM        | [xMneB2WlTlu6GWZp5yy1ag](https://tensorboard.dev/experiment/xMneB2WlTlu6GWZp5yy1ag)  |
|           | CNN         | [k41zgYDvTXK5QUrV9LV7LA](https://tensorboard.dev/experiment/k41zgYDvTXK5QUrV9LV7LA)  |
|           | Pre-trained | [cdAhzk82RWeYtzlEWPGNfg](https://tensorboard.dev/experiment/cdAhzk82RWeYtzlEWPGNfg)  |
|           | Benchmark   | [QorKdGyFR6exuna98qKZNg](https://tensorboard.dev/experiment/QorKdGyFR6exuna98qKZNg)  |
| BPIC2017a | Transformer | [wXFLt4nZRjeJwsZpBrzCwQ](https://tensorboard.dev/experiment/wXFLt4nZRjeJwsZpBrzCwQ)  |
|           | LSTM        | [whJxlqa9QxmDb33Mhghxgg](https://tensorboard.dev/experiment/whJxlqa9QxmDb33Mhghxgg)  |
|           | CNN         | [tOabNA0BS8CeMCDLrd65sg](https://tensorboard.dev/experiment/tOabNA0BS8CeMCDLrd65sg)  |
|           | Pre-trained | [LL1fLrEqTQ2lcbcN2fEVOg](https://tensorboard.dev/experiment/LL1fLrEqTQ2lcbcN2fEVOg)  |
|           | Benchmark   | [B3v4kRmDRserclX6LaHWWQ](https://tensorboard.dev/experiment/B3v4kRmDRserclX6LaHWWQ)  |
| BPIC2017c | Transformer | [HAn9GxheQxWs2Te6RU9Vvg](https://tensorboard.dev/experiment/HAn9GxheQxWs2Te6RU9Vvg)  |
|           | LSTM        | [j88uXwrZQDiR2U2OH6FqEQ](https://tensorboard.dev/experiment/j88uXwrZQDiR2U2OH6FqEQ)  |
|           | CNN         | [4bXUjDnuRnaMqGumu4kAtg](https://tensorboard.dev/experiment/4bXUjDnuRnaMqGumu4kAtg) |
|           | Pre-trained | [pk0AFVixQYyQeZDYAr7L8A](https://tensorboard.dev/experiment/pk0AFVixQYyQeZDYAr7L8A)  |
|           | Benchmark   | [UqCMLro3Q5OUeP5VDkypUg](https://tensorboard.dev/experiment/UqCMLro3Q5OUeP5VDkypUg)  |
| BPIC2017d | Transformer | [lvC8hYHJSKef8sOw8O3gJg](https://tensorboard.dev/experiment/lvC8hYHJSKef8sOw8O3gJg)  |
|           | LSTM        | [700PQpEmQjGGoMNeHm8Tdg](https://tensorboard.dev/experiment/700PQpEmQjGGoMNeHm8Tdg)  |
|           | CNN         | [6OQOJD9JQOy0NvjIBdBWsw](https://tensorboard.dev/experiment/6OQOJD9JQOy0NvjIBdBWsw)  |
|           | Pre-trained | [7ylz0XIaQtmM2aPKYD8UGA](https://tensorboard.dev/experiment/7ylz0XIaQtmM2aPKYD8UGA)  |
|           | Benchmark   | [qqAMj9O0Qhynru8pfZLJew](https://tensorboard.dev/experiment/qqAMj9O0Qhynru8pfZLJew)  |
| Traffic   | Transformer | [GXDqnpcaQnKJgwT0MGtqsw](https://tensorboard.dev/experiment/GXDqnpcaQnKJgwT0MGtqsw)  |
|           | LSTM        | [WeAJcA4lT4yHq1sYcWJBQA](https://tensorboard.dev/experiment/WeAJcA4lT4yHq1sYcWJBQA)  |
|           | CNN         | [WF8voDWsRnSdJiok9cAEpg](https://tensorboard.dev/experiment/WF8voDWsRnSdJiok9cAEpg)  |
|           | Pre-trained | [HQy8vW0ASkaf7MUZw30KRA](https://tensorboard.dev/experiment/HQy8vW0ASkaf7MUZw30KRA)  |
|           | Benchmark   | [brPrvjuPQ4uoDpYUpRhO0Q](https://tensorboard.dev/experiment/brPrvjuPQ4uoDpYUpRhO0Q)  |

 
