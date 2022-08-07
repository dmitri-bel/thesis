import os
import argparse
import time
import numpy as np
import pandas as pd

from processtransformer import constants
from processtransformer.data.processor import LogsDataProcessor

parser = argparse.ArgumentParser(
    description="Process Transformer - Data Processing.")

parser.add_argument("--dataset", 
    type=str, 
    default="helpdesk", 
    help="dataset name")

parser.add_argument("--dir_path", 
    type=str, 
    default="./datasets", 
    help="path to store processed data")

parser.add_argument("--raw_log_file", 
    type=str, 
    default="./datasets/helpdesk/helpdesk.csv", 
    help="path to raw csv log file")

parser.add_argument("--task", 
    type=constants.Task, 
    default=constants.Task.REMAINING_TIME, 
    help="task name")

parser.add_argument("--sort_temporally", 
    type=bool, 
    default=False, 
    help="sort cases by timestamp")

args = parser.parse_args()
print(os.getcwd())

if __name__ == "__main__": 
    ### Process raw logs ###
    # Record start time for benchmarking
    start = time.time()
    # Identify columns according to task type
    if args.task == constants.Task.OUTCOME:
        cols = ["CASE_concept_name", "activity_id", "timestamp", "outcome"]
    else:
        cols = ["CASE_concept_name", "activity_id", "timestamp"]
        #cols = ["Case ID", "Activity", "Complete Timestamp"],
        #cols = ["case:concept:name", "concept:name", "time:timestamp"], 
    # Initiate DataProcessor class
    data_processor = LogsDataProcessor(name=args.dataset, 
        filepath = args.raw_log_file, 
        columns = cols,
        dir_path=args.dir_path, pool = 8) #changed from 4 to 1
    # Process logs according to task type
    data_processor.process_logs(task=args.task, sort_temporally= args.sort_temporally)
    # Calculate processing time
    end = time.time()
    print(f"Total processing time: {end - start}")