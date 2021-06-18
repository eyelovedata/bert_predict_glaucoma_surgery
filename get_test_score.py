import model
import dataset

import logging
import json
import sys
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def get_test_score(model_path):
    #Load in the fine tuned pretrained model
    """
    This function load in the model and tokenizer and evaluate the model on the hold-out test dataset.
    The test score will be saved in test_score.log in the model_path folder
    """
    #Get the tokenizer name and data url from the config
    config_file_pt = model_path + '/config_py_setting.json'
    with open(config_file_pt) as f:
        config = json.load(f)

    # Define the logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(model_path + '/test_score.log')
    stream_handler = logging.StreamHandler()
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    #Load in the fine-tuned model and its tokenizer
    pretrained_model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    print(f'load in the specified pretrained model {model_path} and its associated tokenizer')
    print('')

    # Read in the train, val, test glaucoma surgey data
    train_data, val_data, test_data = dataset.read_glaucoma_surgery_train_test_val(config['data_url'])
    print('load in glaucoma surgery data for train, val and test')
    print('')

    # Split the text and labels for test
    test_texts, test_labels = dataset.get_texts_labels_split(test_data)
    print(f'test texts shape, labels shape {len(test_texts), len(test_labels)}')
    print('')

    #Tokenize the test data
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=config['max_length'])
    print('tokenize the train, val and test text with the given tokenizer')
    print('')

    # Prepare the Pytorch Dataset
    test_dataset = dataset.GlaucomaSurgeryDataset(test_encodings, test_labels)
    print('put the train, val and test encodings into the GlaucomaSurgeryDataset class')
    print('')

    #Prepare the trainer
    trainer = model.BinarySequenceClassificationTrainer(model=pretrained_model, compute_metrics=model.compute_metrics)
    print('Trainer is ready')

    #Make prediction on the test dataset
    predictions = trainer.predict(test_dataset)

    #Specified required metrics
    specified_metrics = ['eval_loss', 'eval_accuracy', 'eval_f1', 'eval_precision', 'eval_recall', 'eval_roc_auc', 'eval_runtime', 'eval_samples_per_second']

    #Print out and record the required metrics
    for sm in specified_metrics:
        logger.info(f'{sm}: {predictions.metrics[sm]}')

if __name__ == "__main__":
    model_path = sys.argv[1]
    get_test_score(model_path)
