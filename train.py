import logging

import config
import dataset
import model
import utils

from transformers import TrainingArguments
from transformers import EarlyStoppingCallback
from transformers import AutoModelForSequenceClassification, AutoTokenizer

#Define the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


def run():
    """
    This function read in the specified train, val, test data.
    Data was tokenized and processed to the model required format.
    Specified pre-trained model and param were loaded into the Trainer.
    The pre-trained model was finetuned on the training data and evaluated on the val data.
    """

    #Read in the train, val, test glaucoma surgey data
    train_data, val_data, test_data = dataset.read_glaucoma_surgery_train_test_val(config.data_url)
    logger.info('load in glaucoma surgery data for train, val and test')

    #Split the text and labels for train, val and test
    train_texts, train_labels = dataset.get_texts_labels_split(train_data)
    val_texts, val_labels = dataset.get_texts_labels_split(val_data)
    test_texts, test_labels = dataset.get_texts_labels_split(test_data)
    logger.info(f'train texts shape, labels shape {len(train_texts), len(train_labels)}')
    logger.info(f'val texts shape, labels shape {len(val_texts), len(val_labels)}')
    logger.info(f'test texts shape, labels shape {len(test_texts), len(test_labels)}')

    #Load in the pretrained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, do_lower_case=config.do_lower_case)
    pretrained_model = AutoModelForSequenceClassification.from_pretrained(config.model_name, num_labels=2, hidden_dropout_prob=config.hidden_dropout_prob)
    logger.info(f'load in the specified pretrained model {pretrained_model.name_or_path} and its associated tokenizer')

    #Tokenize the text with the tokenizer
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=config.max_length)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=config.max_length)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=config.max_length)
    logger.info('tokenize the train, val and test text with the given tokenizer')

    #Prepare the Pytorch Dataset
    train_dataset = dataset.GlaucomaSurgeryDataset(train_encodings, train_labels)
    val_dataset = dataset.GlaucomaSurgeryDataset(val_encodings, val_labels)
    test_dataset = dataset.GlaucomaSurgeryDataset(test_encodings, test_labels)
    logger.info('Put the train, val and test encodings into the GlaucomaSurgeryDataset class')

    #Set up the training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,          # output directory
        num_train_epochs=config.num_train_epochs,              # total number of training epochs
        per_device_train_batch_size=config.per_device_train_batch_size,  # batch size per device during training
        per_device_eval_batch_size=config.per_device_eval_batch_size,   # batch size for evaluation
        warmup_steps=config.warmup_steps,                # number of warmup steps for learning rate scheduler
        learning_rate=config.learning_rate,              #Learning rate
        weight_decay=config.weight_decay,               # strength of weight decay
        logging_dir=config.logging_dir,            # directory for storing logs
        logging_steps=config.logging_steps,        #loging steps
        evaluation_strategy=config.evaluation_strategy,     #evaluation strategy: steps or epochs
        seed=config.seed,   #Seed to control the random
        load_best_model_at_end=config.load_best_model_at_end,   #Whether load in the best model in the end
        metric_for_best_model=config.metric_for_best_model,     #Metrics used for the best model
        greater_is_better=config.greater_is_better,     #higher/lower metrics score, better model performance
        gradient_accumulation_steps=config.gradient_accumulation_steps,     #number of gradient accumulation steps
        report_to="wandb"       #Platform to monitor the training process
    )
    logger.info('Load in the specified training arguments')

    #set up the glaucoma_surgery_trainer
    glaucoma_surgery_trainer = model.BinarySequenceClassificationTrainer(
        model=pretrained_model,  # the instantiated Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=val_dataset,  # evaluation dataset
        compute_metrics=model.compute_metrics,  #evaluation Metrics
        callbacks=[EarlyStoppingCallback(config.early_stop, config.early_stop_thresh)] #early Stop
    )
    logger.info('Load the glaucoma_surgery_trainer')

    #Take model training
    glaucoma_surgery_trainer.train()
    logger.info('Training finished')

    #Saving the best model and its associated tokenizer
    glaucoma_surgery_trainer.save_model(config.model_saving_dir)
    tokenizer.save_pretrained(config.model_saving_dir)
    logger.info(f'Saving best model and its associated tokenizer in {config.model_saving_dir}')

    #Model Evaluation Results on val data
    logger.info('Results from the best model on evaluation dataset:')
    glaucoma_surgery_trainer.evaluate()

if __name__=="__main__":
    utils.set_seed(config.seed)
    run()
    utils.convert_config_log_to_json(config.model_saving_dir)


