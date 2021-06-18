import config
from transformers import Trainer
from sklearn import metrics
import torch

class BinarySequenceClassificationTrainer(Trainer):
    """
    Override the compute_loss in the Trainer by integrating the weighted loss.
    """
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        logits = logits[:, 1]-logits[:, 0]
        if config.pos_weight:
            loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([config.pos_weight]).to('cuda'))
        else:
            loss_fct = torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(logits,
                        labels.float())
        return (loss, outputs) if return_outputs else loss

def compute_metrics(pred):
    """
    Redefined the evaluation metrics which will be used by the Trainer
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = metrics.precision_recall_fscore_support(labels, preds, average='binary')
    acc = metrics.accuracy_score(labels, preds)
    soft_m = torch.nn.Softmax(dim=1)
    proba = soft_m(torch.tensor(pred.predictions))
    roc_auc = metrics.roc_auc_score(labels, proba[:, 1])
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc
    }