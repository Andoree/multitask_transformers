import torch
from multitask_model import DataLoaderWithTaskname
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import pandas as pd
from tqdm import trange


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
        # )
        with torch.no_grad():
            embeddings = multitask_model.taskmodels_dict[task_name](eval_dataloader)[0]
            embeddings_df = pd.DataFrame({"embedding": [embeddings, ]})
            embeddings_dict[task_name] = embeddings_df

    return embeddings_dict


def find_sent_end(example):
    for i in range(len(example)):
        if i == 1 or i == 2:
            return i


def get_embeddings(multitask_model, features_dict, collection="validation", emb_size=768):
    task_embeddings_dict = {}
    multitask_model.eval()
    with torch.no_grad():
        for task_name in ["task_1", "task_2"]:
            inference_examples = features_dict[task_name][collection]["input_ids"]
            embeddings_dict = {}
            cls_embeddings_matrix = np.empty(shape=(inference_examples.shape[0], emb_size), dtype=float)
            mean_embeddings_matrix = np.empty(shape=(inference_examples.shape[0], emb_size), dtype=float)
            for i in trange(inference_examples.shape[0]):
                example = inference_examples[i].astype(int)
                text_end_position = find_sent_end(example)
                example = example[np.newaxis, :]
                example = torch.from_numpy(example).to("cuda")
                embedding = multitask_model.encoder(example)[0]
                embedding = embedding.cpu().detach().numpy()
                cls_embedding = embedding[0][0]
                cls_embeddings_matrix[i] = cls_embedding
                mean_embedding = embedding[0][:text_end_position].mean(axis=1)
                mean_embeddings_matrix[i] = mean_embedding
            print(f"Completed embeddings inference for task: {task_name}")
            cls_embeddings_df = pd.DataFrame({"cls_embedding": [cls_embeddings_matrix, ]})
            embeddings_dict["cls"] = cls_embeddings_df
            mean_embeddings_df = pd.DataFrame({"mean_embedding": [mean_embeddings_matrix, ]})
            embeddings_dict["mean"] = mean_embeddings_df
            task_embeddings_dict[task_name] = embeddings_dict

    return embeddings_dict
