import os
import time
import argparse
import warnings
import numpy as np
import pandas as pd
import datetime as dt
import tensorflow as tf

from sklearn import metrics 
from posixpath import abspath
from processtransformer import constants
from processtransformer.data import loader
from processtransformer.models import transformer

warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description="Process Outcome Prediction through Transfer Learning.")
parser.add_argument("--dataset", required=True, type=str, help="dataset name")
parser.add_argument("--model_dir", default="./models", type=str, help="model directory")
parser.add_argument("--result_dir", default="./results", type=str, help="results directory")
parser.add_argument("--log_dir", default="./logs", type=str, help="training logs directory")
parser.add_argument("--tsb_dir", default="./tensorboard", type=str, help="tensorboard directory")
parser.add_argument("--model", default="transfer", type=str, help="description model")
parser.add_argument("--task", type=constants.Task, default=constants.Task.OUTCOME,  help="task name")
parser.add_argument("--epochs", default=100, type=int, help="number of total epochs")
parser.add_argument("--batch_size", default=12, type=int, help="batch size")
parser.add_argument("--learning_rate", default=0.001, type=float, help="learning rate")
parser.add_argument("--gpu", default=0, type=int, help="gpu id")
args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


if __name__ == "__main__":
    # --define paths
    log_path = f"{args.log_dir}/{args.dataset}"
    if not os.path.exists(log_path): os.makedirs(log_path)
    log_path = f"{log_path}/training_{args.task.value}_{args.model}.csv"

    result_path = f"{args.result_dir}/{args.dataset}"
    if not os.path.exists(result_path): os.makedirs(result_path)
    result_path = f"{result_path}/results_{args.task.value}_{args.model}"

    model_path = f"{args.model_dir}/{args.dataset}"
    if not os.path.exists(model_path): os.makedirs(model_path)
    model_path = f"{model_path}/{args.task.value}_{args.model}"

    tsb_path = f"{args.tsb_dir}/{args.dataset}"
    if not os.path.exists(tsb_path): os.makedirs(tsb_path)
    tsb_path = f"{tsb_path}/{args.task.value}_{args.model}/" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")

    pretrained_path = f"{args.model_dir}/{args.dataset}/next_activity"


    # --create callbacks
    callback_list = [
        tf.keras.callbacks.TensorBoard(tsb_path, histogram_freq=1),
        tf.keras.callbacks.CSVLogger(log_path, append = True),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    ]
 

    # --load data
    data_loader = loader.LogsDataLoader(name = args.dataset)
    (train_df, test_df, x_word_dict, y_word_dict, max_case_length, vocab_size, num_output) = data_loader.load_data(args.task)
    

    # --tokenize train data
    train_token_x, train_token_y = data_loader.prepare_data_outcome_prediction(train_df, x_word_dict, y_word_dict, max_case_length)


    # --check if pretrained model exists
    if os.path.isdir(pretrained_path) == False:
        print(f"\n \nPretrained model not found in the following path:")
        print(os.path.abspath(pretrained_path))
        print(f"Execution will be stopped.\n\n")
        exit()


    # --load pretrained model
    pretrained = tf.keras.models.load_model(pretrained_path) 


    # --freeze layers
    for i in range(len(pretrained.layers)):
        if i < len(pretrained.layers) - 7:
            pretrained.layers[i].trainable = False
                
        
    # --replace final layer
    outputs = tf.keras.layers.Dense(num_output, activation='relu')(pretrained.layers[-2].output)
    
        
    # --recreate model
    pretrained = tf.keras.Model(pretrained.inputs, outputs)

    pretrained.compile(
        optimizer = tf.keras.optimizers.Adam(args.learning_rate),
        loss =  tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy()])

    pretrained.fit(train_token_x, train_token_y, 
        epochs=args.epochs,
        batch_size=args.batch_size, 
        shuffle=True,
        verbose=1, 
        validation_split = 0.2,
        callbacks=callback_list)

    pretrained.save(model_path)

    # --evaluate 
    # --tokenize test data
    test_token_x, test_token_y = data_loader.prepare_data_outcome_prediction(
        test_df, x_word_dict, y_word_dict, max_case_length)

    # --make predictions
    y_pred = np.argmax(pretrained.predict(test_token_x), axis=1)

    # --calculate metrics
    accuracy = metrics.accuracy_score(test_token_y, y_pred)
    precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            test_token_y, y_pred, average="weighted")

    # --save results
    results_df = pd.DataFrame([{
        "accuracy":accuracy,
        "fscore": fscore, 
        "precision": precision,
        "recall": recall}])

    results_df.to_csv(f"{result_path}_general.csv", index=False)


    # --evaluate for different prefix lengths (k)
    k, accuracies,fscores, precisions, recalls = [],[],[],[],[]
    for i in range(max_case_length):
        test_data_subset = test_df[test_df["k"]==i]

        if len(test_data_subset) > 0:
            # --tokenize test data
            test_token_x, test_token_y = data_loader.prepare_data_outcome_prediction(test_data_subset, 
                x_word_dict, y_word_dict, max_case_length)   
            
            # --make predictions
            y_pred = np.argmax(pretrained.predict(test_token_x), axis=1)

            # --calculate metrics
            accuracy = metrics.accuracy_score(test_token_y, y_pred)
            precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
                test_token_y, y_pred, average="weighted")

            # --append scores
            k.append(i)
            accuracies.append(accuracy)
            fscores.append(fscore)
            precisions.append(precision)
            recalls.append(recall)


    # --print averages
    print('Average accuracy across all prefixes:', np.mean(accuracies))
    print('Average f-score across all prefixes:', np.mean(fscores))
    print('Average precision across all prefixes:', np.mean(precisions))
    print('Average recall across all prefixes:', np.mean(recalls))

    # --save results
    results_df = pd.DataFrame({
        "prefix_length":k,
        "accuracy":accuracies,
        "fscore": fscores, 
        "precision":precisions,
        "recall":recalls})

    results_df.to_csv(f"{result_path}_prefixes.csv", index=False)


