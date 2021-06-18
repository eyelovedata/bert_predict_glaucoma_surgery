import logging
from datetime import datetime

import utils

current_time = datetime.now()
print_current_time = current_time.strftime("%d/%m/%Y %H:%M:%S")
folder_adding_current_time = current_time.strftime("_%d_%m_%Y_%H_%M_%S")
data_url = "gs://stanfordoptimagroup/STRIDE/predictglaucomasurgery/extrinsicdatabalanced.csv"
num_train_epochs = 10  # total number of epochs
per_device_train_batch_size = 16  # batch size per device during training
per_device_eval_batch_size = 8  # batch size for evaluation
warmup_steps = 4  # number of warmup steps for learning rate scheduler
learning_rate = 0.00003 #learning rate
weight_decay = 0.01  # strength of weight decay
logging_steps = 113 #evey logging_steps print out the results
evaluation_strategy = 'steps' #every logging_steps print out the results
seed = 42 #Seed to make sure the replicate of the training
load_best_model_at_end = True #Load the best model after training
metric_for_best_model = 'eval_loss' #best model was selected by using this metrics
greater_is_better = False #How to use metric_for_best_model
pos_weight = 3011/601 #Applied pos_weights when calculating the BCELoss
gradient_accumulation_steps = 2  #gradient_accumulation_steps
model_name = 'bert-base-uncased' #Pre-trained model want to use
do_lower_case = True #Whether do lower case processing on the test
model_saving_dir = '/home/jupyter/finetunedmodel/' + model_name + folder_adding_current_time + '_seed_' + str(seed)
output_dir = model_saving_dir + '/results'  # output directory
logging_dir = model_saving_dir + '/logs'  # directory for storing logs
max_length = 512 #Maximum length of the text
early_stop = 3  #Number of steps of early stop
early_stop_thresh = 0.0 #Threshold of early stops
hidden_dropout_prob = 0.1   #Dropout rate before the linear classifier layer

#Make dir for saving the model
utils.mkdir_for_saving_model(model_saving_dir)

#Set up a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
file_handler = logging.FileHandler(model_saving_dir + '/config_py_setting.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

#Saving the config param into a log file
logger.info(f'current_time = {print_current_time}')
logger.info(f'data_url = {data_url}')
logger.info(f'output_dir = {output_dir}')  # output directory
logger.info(f'num_train_epochs = {num_train_epochs}') # total number of training epochs
logger.info(f'per_device_train_batch_size = {per_device_train_batch_size}')  # batch size per device during training
logger.info(f'per_device_eval_batch_size = {per_device_eval_batch_size}')  # batch size for evaluation
logger.info(f'warmup_steps = {warmup_steps}')  # number of warmup steps for learning rate scheduler
logger.info(f'learning_rate = {learning_rate}') #Learning rate
logger.info(f'weight_decay = {weight_decay}')  # strength of weight decay
logger.info(f'logging_dir = {logging_dir}')  # directory for storing huggingface logs
logger.info(f'logging_steps = {logging_steps}')  # evey logging_steps print out the results
logger.info(f'evaluation_strategy = {evaluation_strategy}')  # every logging_steps print out the results
logger.info(f'seed = {seed}')  # Seed to make sure the replicate of the training
logger.info(f'load_best_model_at_end = {load_best_model_at_end}')  # Load the best model after training
logger.info(f'metric_for_best_model = {metric_for_best_model}')  # best model was selected by using this metrics
logger.info(f'greater_is_better = {greater_is_better}')
logger.info(f'pos_weight = {pos_weight}')
logger.info(f'gradient_accumulation_steps = {gradient_accumulation_steps}')
logger.info(f'model_name = {model_name}')
logger.info(f'model_saving_dir = {model_saving_dir}')
logger.info(f'max_length = {max_length}')
logger.info(f'early_stop = {early_stop}')
logger.info(f'early_stop_thresh = {early_stop_thresh}')
logger.info(f'weight_decay = {weight_decay}')
logger.info(f'hidden_dropout_prob = {hidden_dropout_prob}')
