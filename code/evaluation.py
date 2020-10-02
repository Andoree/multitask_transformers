from multitask_model import DataLoaderWithTaskname
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


def evaluate_classification(trainer, features_dict, dataset_dict):
    preds_dict = {}
    for task_name in ["task_1", "task_2"]:
        eval_dataloader = DataLoaderWithTaskname(
            task_name,
            trainer.get_eval_dataloader(eval_dataset=features_dict[task_name]["validation"])
        )
        preds_dict[task_name] = trainer._prediction_loop(
            eval_dataloader,
            description=f"Validation: {task_name}",
        )
    for task_name in preds_dict.keys():
        prediction = preds_dict[task_name].predictions
        predicted_labels = np.argmax(prediction, axis=1)
        val_labels = dataset_dict[task_name]["validation"]["label"]
        accuracy = accuracy_score(val_labels, predicted_labels, )
        macro_precision = precision_score(val_labels, predicted_labels, average="macro")
        macro_recall = recall_score(val_labels, predicted_labels, average="macro")
        macro_f1 = f1_score(val_labels, predicted_labels, average="macro")
        task_results = {
            "Accuracy": accuracy,
            "Macro precision": macro_precision,
            "Macro recall": macro_recall,
            "Macro F1": macro_f1
        }
        preds_dict[task_name] = task_results
    return preds_dict
