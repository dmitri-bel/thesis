import os
import json
import pandas as pd
import numpy as np
import datetime
from multiprocessing import  Pool
from ..constants import Task

class LogsDataProcessor:
    def __init__(self, name, filepath, columns, dir_path = "./datasets/processed", pool = 1):
        """Provides support for processing raw logs.
        Args:
            name: str: Dataset name
            filepath: str: Path to raw logs dataset
            columns: list: name of column names
            dir_path:  str: Path to directory for saving the processed dataset
            pool: Number of CPUs (processes) to be used for data processing
        """
        self._name = name
        self._filepath = filepath
        self._org_columns = columns
        self._dir_path = dir_path
        if not os.path.exists(f"{dir_path}/{self._name}/processed"):
            os.makedirs(f"{dir_path}/{self._name}/processed")
        self._dir_path = f"{self._dir_path}/{self._name}/processed"
        self._pool = pool

    def _load_df(self, task, sort_temporally = False):
        df = pd.read_csv(self._filepath)
        df = df[self._org_columns]

        if task == Task.OUTCOME:
            df.columns = ["case:concept:name", 
                "concept:name", "time:timestamp", "outcome"]
            df["outcome"] = df["outcome"].str.lower()    
        else:
            df.columns = ["case:concept:name", 
                "concept:name", "time:timestamp"]
        df["concept:name"] = df["concept:name"].str.lower()
        df["concept:name"] = df["concept:name"].str.replace(" ", "-")
        df["time:timestamp"] = df["time:timestamp"].str.replace("/", "-")
        df["time:timestamp"]= pd.to_datetime(df["time:timestamp"],  
            dayfirst=True).map(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
        
        if sort_temporally:
            df.sort_values(by = ["time:timestamp"], inplace = True)
        return df

    def _extract_logs_metadata(self, df, task):


        if task == Task.OUTCOME:
            outcomes = list(df["outcome"].unique())
            activities = list(df["concept:name"].unique()) + outcomes
        else:
            activities = list(df["concept:name"].unique())
    
        keys = ["[PAD]", "[UNK]"]
        keys.extend(activities)

        if task == Task.OUTCOME:
            # --creates two dictionairies: one with activities + PAD/UNK tokens and one with just process outcomes.
            x_word_dict = dict({"x_word_dict": dict(zip(keys, range(len(keys))))})
            y_word_dict = dict({"y_word_dict": dict(zip(outcomes, range(len(outcomes))))})
        else: 
            # --creates two dictionairies: one with PAD & UNK tokens and one without.
            x_word_dict = dict({"x_word_dict": dict(zip(keys, range(len(keys))))})
            y_word_dict = dict({"y_word_dict": dict(zip(activities, range(len(activities))))})

        # --uses dictionary to create json file with metadata
        x_word_dict.update(y_word_dict)
        coded_json = json.dumps(x_word_dict)
        with open(f"{self._dir_path}/{task.value}_metadata.json", "w") as metadata_file:
            metadata_file.write(coded_json)

    def _next_activity_helper_func(self, df):
        # --define variables
        case_id, case_name = "case:concept:name", "concept:name"

        # --create empty dataframe for storing data after processing is done
        processed_df = pd.DataFrame(columns = ["case_id", 
        "prefix", "k", "next_act"])

        # --initialize counter (will be used as unique id for each row in processed_df)
        idx = 0

        # --extract all unique case_ids from the raw supplied dataframe
        unique_cases = df[case_id].unique()

        # --for each unique case...
        for _, case in enumerate(unique_cases):

            # ... create a list 'act' for all the activities related to that case 
            act = df[df[case_id] == case][case_name].to_list()

            # --for each activity in the new 'act' list...
            for i in range(len(act) - 1):

                # --extract the prefix of the activity i, including itself 
                # (starting from act[0] and continuing until act[i] NOT INCLUDING act[i+1])
                prefix = np.where(i == 0, act[0], " ".join(act[:i+1]))  

                # --extract activity i+1     
                next_act = act[i+1]

                # --populate processed_df
                processed_df.at[idx, "case_id"]  =  case
                processed_df.at[idx, "prefix"]  =  prefix
                processed_df.at[idx, "k"] =  i
                processed_df.at[idx, "next_act"] = next_act

                # --increment counter
                idx = idx + 1
        return processed_df

    def _process_next_activity(self, df, train_list, test_list):
        # --split the input df into X sections. 
        # Output is a list of (sub)arrays, with each (sub)array being a section of the split.
        # (in this case with X = self._pool, the amount of processes allocated for work distribution)
        df_split = np.array_split(df, self._pool)

        # The pool function distributes the workload over the available amount of processors 
        with Pool(processes=self._pool) as pool:

            # pd.concat concatenates dataframes (or other pandas objects)
            # imap_unordered is a method from the pool class. It applies a function f to each element of an iterable.
            # imap_unordered returns an iterator and NOT a list, df or other iterable object.
            processed_df = pd.concat(pool.imap_unordered(self._next_activity_helper_func, df_split))

        # --train_list is a list containing all the case_id that belong to the train set
        # --test_list is a list containing all the case_id that belong to the test set
        # --use the train_list and test_list to create the respective dataframes
        train_df = processed_df[processed_df["case_id"].isin(train_list)]
        test_df = processed_df[processed_df["case_id"].isin(test_list)]

        # --save dataframes to csv 
        train_df.to_csv(f"{self._dir_path}/{Task.NEXT_ACTIVITY.value}_train.csv", index = False)
        test_df.to_csv(f"{self._dir_path}/{Task.NEXT_ACTIVITY.value}_test.csv", index = False)

    def _next_time_helper_func(self, df):
        case_id = "case:concept:name"
        event_name = "concept:name"
        event_time = "time:timestamp"
        processed_df = pd.DataFrame(columns = ["case_id", "prefix", "k", "time_passed", 
            "recent_time", "latest_time", "next_time", "remaining_time_days"])
        idx = 0
        unique_cases = df[case_id].unique()
        for _, case in enumerate(unique_cases):
            act = df[df[case_id] == case][event_name].to_list()
            time = df[df[case_id] == case][event_time].str[:19].to_list()
            time_passed = 0
            latest_diff = datetime.timedelta()
            recent_diff = datetime.timedelta()
            next_time =  datetime.timedelta()
            for i in range(0, len(act)):
                prefix = np.where(i == 0, act[0], " ".join(act[:i+1]))
                if i > 0:
                    latest_diff = datetime.datetime.strptime(time[i], "%Y-%m-%d %H:%M:%S") - \
                                        datetime.datetime.strptime(time[i-1], "%Y-%m-%d %H:%M:%S")
                if i > 1:
                    recent_diff = datetime.datetime.strptime(time[i], "%Y-%m-%d %H:%M:%S")- \
                                    datetime.datetime.strptime(time[i-2], "%Y-%m-%d %H:%M:%S")
                latest_time = np.where(i == 0, 0, latest_diff.days)
                recent_time = np.where(i <=1, 0, recent_diff.days)
                time_passed = time_passed + latest_time
                if i+1 < len(time):
                    next_time = datetime.datetime.strptime(time[i+1], "%Y-%m-%d %H:%M:%S") - \
                                datetime.datetime.strptime(time[i], "%Y-%m-%d %H:%M:%S")
                    next_time_days = str(int(next_time.days))
                else:
                    next_time_days = str(1)
                processed_df.at[idx, "case_id"]  = case
                processed_df.at[idx, "prefix"]  =  prefix
                processed_df.at[idx, "k"] = i
                processed_df.at[idx, "time_passed"] = time_passed
                processed_df.at[idx, "recent_time"] = recent_time
                processed_df.at[idx, "latest_time"] =  latest_time
                processed_df.at[idx, "next_time"] = next_time_days
                idx = idx + 1
        processed_df_time = processed_df[["case_id", "prefix", "k", "time_passed", 
            "recent_time", "latest_time","next_time"]]
        return processed_df_time

    def _process_next_time(self, df, train_list, test_list):
        df_split = np.array_split(df, self._pool)
        with Pool(processes=self._pool) as pool:
            processed_df = pd.concat(pool.imap_unordered(self._next_time_helper_func, df_split))
        train_df = processed_df[processed_df["case_id"].isin(train_list)]
        test_df = processed_df[processed_df["case_id"].isin(test_list)]
        train_df.to_csv(f"{self._dir_path}/{Task.NEXT_TIME.value}_train.csv", index = False)
        test_df.to_csv(f"{self._dir_path}/{Task.NEXT_TIME.value}_test.csv", index = False)

    def _remaining_time_helper_func(self, df):
        case_id = "case:concept:name"
        event_name = "concept:name"
        event_time = "time:timestamp"
        processed_df = pd.DataFrame(columns = ["case_id", "prefix", "k", "time_passed", 
                "recent_time", "latest_time", "next_act", "remaining_time_days"])
        idx = 0
        unique_cases = df[case_id].unique()
        for _, case in enumerate(unique_cases):
            act = df[df[case_id] == case][event_name].to_list()
            time = df[df[case_id] == case][event_time].str[:19].to_list()
            time_passed = 0
            latest_diff = datetime.timedelta()
            recent_diff = datetime.timedelta()
            for i in range(0, len(act)):
                prefix = np.where(i == 0, act[0], " ".join(act[:i+1]))
                if i > 0:
                    latest_diff = datetime.datetime.strptime(time[i], "%Y-%m-%d %H:%M:%S") - \
                                        datetime.datetime.strptime(time[i-1], "%Y-%m-%d %H:%M:%S")
                if i > 1:
                    recent_diff = datetime.datetime.strptime(time[i], "%Y-%m-%d %H:%M:%S")- \
                                    datetime.datetime.strptime(time[i-2], "%Y-%m-%d %H:%M:%S")

                latest_time = np.where(i == 0, 0, latest_diff.days)
                recent_time = np.where(i <=1, 0, recent_diff.days)
                time_passed = time_passed + latest_time

                time_stamp = str(np.where(i == 0, time[0], time[i]))
                ttc = datetime.datetime.strptime(time[-1], "%Y-%m-%d %H:%M:%S") - \
                        datetime.datetime.strptime(time_stamp, "%Y-%m-%d %H:%M:%S")
                ttc = str(ttc.days)  

                processed_df.at[idx, "case_id"]  = case
                processed_df.at[idx, "prefix"]  =  prefix
                processed_df.at[idx, "k"] = i
                processed_df.at[idx, "time_passed"] = time_passed
                processed_df.at[idx, "recent_time"] = recent_time
                processed_df.at[idx, "latest_time"] =  latest_time
                processed_df.at[idx, "remaining_time_days"] = ttc
                idx = idx + 1
        processed_df_remaining_time = processed_df[["case_id", "prefix", "k", 
            "time_passed", "recent_time", "latest_time","remaining_time_days"]]
        return processed_df_remaining_time

    def _process_remaining_time(self, df, train_list, test_list):
        df_split = np.array_split(df, self._pool)
        with Pool(processes=self._pool) as pool:
            processed_df = pd.concat(pool.imap_unordered(self._remaining_time_helper_func, df_split))
        train_remaining_time = processed_df[processed_df["case_id"].isin(train_list)]
        test_remaining_time = processed_df[processed_df["case_id"].isin(test_list)]
        train_remaining_time.to_csv(f"{self._dir_path}/{Task.REMAINING_TIME.value}_train.csv", index = False)
        test_remaining_time.to_csv(f"{self._dir_path}/{Task.REMAINING_TIME.value}_test.csv", index = False)

    def _outcome_prediction_helper_func(self, df):
        # --define variables
        case_id, case_name = "case:concept:name", "concept:name"

        # --create empty dataframe for storing data after processing is done
        processed_df = pd.DataFrame(columns = ["case_id", 
        "prefix", "k", "outcome"])

        # --initialize counter (will be used as unique id for each row in processed_df)
        idx = 0

        # --extract all unique case_ids from the raw supplied dataframe
        unique_cases = df[case_id].unique()

        # --for each unique case...
        for _, case in enumerate(unique_cases):

            # ... save the process-outcome label
            #outcome = df[df[case_id] == case][outcome].unique()
            outcomes = df[df[case_id] == case]["outcome"].to_list()


            # ... create a list 'act' for all the activities related to that case 
            act = df[df[case_id] == case][case_name].to_list()

            # --for each activity in the new 'act' list...
            for i in range(len(act) - 1):

                # --extract the prefix of the activity i, including itself 
                # (starting from act[0] and continuing until act[i] NOT INCLUDING act[i+1])
                prefix = np.where(i == 0, act[0], " ".join(act[:i+1]))  

                # --extract processoutcome   
                outcome = outcomes[0]

                # --populate processed_df
                processed_df.at[idx, "case_id"]  =  case
                processed_df.at[idx, "prefix"]  =  prefix
                processed_df.at[idx, "k"] =  i
                processed_df.at[idx, "outcome"] = outcome

                # --increment counter
                idx = idx + 1
        return processed_df

    def _process_outcome_prediction(self, df, train_list, test_list):
        # --split the input df into X sections. 
        # Output is a list of (sub)arrays, with each (sub)array being a section of the split.
        # (in this case with X = self._pool, the amount of processes allocated for work distribution)
        df_split = np.array_split(df, self._pool)

        # The pool function distributes the workload over the available amount of processors 
        with Pool(processes=self._pool) as pool:

            # pd.concat concatenates dataframes (or other pandas objects)
            # imap_unordered is a method from the pool class. It applies a function f to each element of an iterable.
            # imap_unordered returns an iterator and NOT a list, df or other iterable object.
            processed_df = pd.concat(pool.imap_unordered(self._outcome_prediction_helper_func, df_split))

        # --train_list is a list containing all the case_id that belong to the train set
        # --test_list is a list containing all the case_id that belong to the test set
        # --use the train_list and test_list to create the respective dataframes
        train_df = processed_df[processed_df["case_id"].isin(train_list)]
        test_df = processed_df[processed_df["case_id"].isin(test_list)]

        # --save dataframes to csv 
        train_df.to_csv(f"{self._dir_path}/{Task.OUTCOME.value}_train.csv", index = False)
        test_df.to_csv(f"{self._dir_path}/{Task.OUTCOME.value}_test.csv", index = False)

    def process_logs(self, task, 
        sort_temporally = False, 
        train_test_ratio = 0.80):

        df = self._load_df(task, sort_temporally)
        self._extract_logs_metadata(df, task)

        train_test_ratio = int(abs(df["case:concept:name"].nunique()*train_test_ratio))
        train_list = df["case:concept:name"].unique()[:train_test_ratio]
        test_list = df["case:concept:name"].unique()[train_test_ratio:]
        if task == Task.NEXT_ACTIVITY:
            self._process_next_activity(df, train_list, test_list)
        elif task == Task.NEXT_TIME:
            self._process_next_time(df, train_list, test_list)
        elif task == Task.REMAINING_TIME:
            self._process_remaining_time(df, train_list, test_list)
        elif task == Task.OUTCOME:
            self._process_outcome_prediction(df, train_list, test_list)
        else:
            raise ValueError("Invalid task.")