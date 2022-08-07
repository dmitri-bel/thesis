import enum

@enum.unique
class Task(enum.Enum):
  """Look up for tasks."""
  
  NEXT_ACTIVITY = "next_activity"
  NEXT_TIME = "next_time"
  REMAINING_TIME = "remaining_time"
  TRANSFER = "transfer_learning"
  OUTCOME = "outcome_prediction"
  OUTCOME_LSTM = "outcome_prediction_lstm"
  OUTCOME_CNN = "outcome_prediction_cnn"

