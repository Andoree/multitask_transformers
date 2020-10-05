import torch
from multitask_model import DataLoaderWithTaskname
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import pandas as pd


def evaluate_classification(trainer, features_dict, dataset_dict, collection="validation"):
    preds_dict = {}
    for task_name in ["task_1", "task_2"]:
        eval_dataloader = DataLoaderWithTaskname(
            task_name,
            trainer.get_eval_dataloader(eval_dataset=features_dict[task_name][collection])
        )
        preds_dict[task_name] = trainer._prediction_loop(
            eval_dataloader,
            description=f"Validation: {task_name}",
        )
    for task_name in preds_dict.keys():
        prediction = preds_dict[task_name].predictions
        predicted_labels = np.argmax(prediction, axis=1)
        val_labels = dataset_dict[task_name][collection]["label"]
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


def get_predictions(trainer, features_dict, class_names_dict, collection="validation"):
    predictions_dict = {}
    for task_name in ["task_1", "task_2"]:
        eval_dataloader = DataLoaderWithTaskname(
            task_name,
            trainer.get_eval_dataloader(eval_dataset=features_dict[task_name][collection])
        )
        prediction_output = trainer._prediction_loop(
            eval_dataloader,
            description=f"{collection}: {task_name}",
        )
        class_names_list = class_names_dict[task_name]
        prediction = prediction_output.predictions
        prediction_df = pd.DataFrame(data=prediction, columns=class_names_list, )
        predictions_dict[task_name] = prediction_df

    return predictions_dict


def get_last_layer_embedding(multitask_model, trainer, features_dict, collection="validation"):
    embeddings_dict = {}
    for task_name in ["task_1", "task_2"]:
        # eval_dataloader = DataLoaderWithTaskname(
        #     task_name,
        eval_dataloader = trainer.get_eval_dataloader(eval_dataset=features_dict[task_name][collection])
        #)
        with torch.no_grad():
            embeddings = multitask_model.taskmodels_dict[task_name](eval_dataloader)[0]
            embeddings_df = pd.DataFrame({"embedding": [embeddings, ]})
            embeddings_dict[task_name] = embeddings_df

    return embeddings_dict
