import logging
from argparse import ArgumentParser
import os

import torch
import transformers
import pandas as pd
from evaluation import evaluate_classification
from multitask_model import MultitaskModel, NLPDataCollator, MultitaskTrainer
from multitask_preprocessing import load_dataset, convert_features_function, data_to_features

from evaluation import get_predictions, get_embeddings

logging.basicConfig(level=logging.INFO)


def main():
    parser = ArgumentParser()
    parser.add_argument('--corpus_dir', required=True)
    parser.add_argument('--text_column')
    parser.add_argument('--model_name')
    parser.add_argument('--max_seq_length', type=int)
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--output_dir')
    args = parser.parse_args()
    torch.manual_seed(42)

    logging.basicConfig(level=logging.INFO)

    corpus_dir = args.corpus_dir
    text_column_name = args.text_column
    model_name = args.model_name
    max_seq_length = args.max_seq_length
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    output_dir = args.output_dir

    train_df_1 = pd.read_csv(os.path.join(corpus_dir, "train_{}.csv".format("sentiments_cloudvision")),
                             encoding="utf-8")
    train_df_1.dropna(subset=[text_column_name], inplace=True)
    val_df_1 = pd.read_csv(os.path.join(corpus_dir, "val_{}.csv".format("sentiments_cloudvision")), encoding="utf-8")
    val_df_1.dropna(subset=[text_column_name], inplace=True)
    train_df_2 = pd.read_csv(os.path.join(corpus_dir, "train_{}.csv".format("topics_cloudvision")), encoding="utf-8")
    train_df_2.dropna(subset=[text_column_name], inplace=True)
    val_df_2 = pd.read_csv(os.path.join(corpus_dir, "val_{}.csv".format("topics_cloudvision")), encoding="utf-8")
    val_df_2.dropna(subset=[text_column_name], inplace=True)
    train_dfs = {
        "task_1": train_df_1,
        "task_2": train_df_2
    }
    val_dfs = {
        "task_1": val_df_1,
        "task_2": val_df_2
    }
    dataset_dict_1, id_to_class_1 = load_dataset(train_df_1, val_df_1, text_column_name)
    dataset_dict_2, id_to_class_2 = load_dataset(train_df_2, val_df_2, text_column_name)
    classes_list_1 = []
    for i in range(len(id_to_class_1.keys())):
        class_label = id_to_class_1[i]
        classes_list_1.append(class_label)
    classes_list_2 = []
    for i in range(len(id_to_class_2.keys())):
        class_label = id_to_class_2[i]
        classes_list_2.append(class_label)
    dataset_dict = {
        "task_1": dataset_dict_1,
        "task_2": dataset_dict_2
    }
    id_to_class_dicts = {
        "task_1": id_to_class_1,
        "task_2": id_to_class_2
    }
    id_to_class = {
        "task_1": classes_list_1,
        "task_2": classes_list_2
    }

    multitask_model = MultitaskModel.create(
        model_name=model_name,
        model_type_dict={
            "task_1": transformers.AutoModelForSequenceClassification,
            "task_2": transformers.AutoModelForSequenceClassification,
        },
        model_config_dict={
            "task_1": transformers.AutoConfig.from_pretrained(model_name,
                                                              num_labels=len(id_to_class_dicts["task_1"].keys())),
            "task_2": transformers.AutoConfig.from_pretrained(model_name,
                                                              num_labels=len(id_to_class_dicts["task_2"].keys())),
        },
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    feature_fn = convert_features_function(tokenizer, max_seq_length)
    convert_func_dict = {
        "task_1": feature_fn,
        "task_2": feature_fn,
    }
    columns_dict = {
        "task_1": ['input_ids', 'attention_mask', 'labels'],
        "task_2": ['input_ids', 'attention_mask', 'labels'],
    }
    features_dict = data_to_features(dataset_dict, convert_func_dict, columns_dict)

    train_dataset = {
        task_name: dataset["train"]
        for task_name, dataset in features_dict.items()
    }
    val_dataset_dict = {
        task_name: dataset["validation"]
        for task_name, dataset in features_dict.items()
    }

    trainer = MultitaskTrainer(
        model=multitask_model,
        args=transformers.TrainingArguments(
            output_dir="./models/multitask_model",
            overwrite_output_dir=True,
            learning_rate=learning_rate,
            do_train=True,
            num_train_epochs=num_epochs,
            # Adjust batch size if this doesn't fit on the Colab GPU
            per_device_train_batch_size=batch_size,
            save_steps=3000,
        ),
        # compute_metrics=classification_metrics,
        data_collator=NLPDataCollator(),
        train_dataset=train_dataset,
        eval_dataset=val_dataset_dict
    )
    trainer.train()

    validation_results = evaluate_classification(trainer, features_dict, dataset_dict)
    for task_name, results_dict in validation_results.items():
        for metric_name, value in results_dict.items():
            print(f"Validation quality: After training, task: {task_name},"
                  f" {metric_name} : {value}")
    training_results = evaluate_classification(trainer, features_dict, dataset_dict, collection="train")
    for task_name, results_dict in training_results.items():
        for metric_name, value in results_dict.items():
            print(f"Training quality: After training, task: {task_name},"
                  f" {metric_name} : {value}")

    validation_predictions = get_predictions(trainer, features_dict, id_to_class, collection="validation")
    train_predictions = get_predictions(trainer, features_dict, id_to_class, collection="train")

    # train_embeddings = get_last_layer_embedding(multitask_model, trainer, features_dict, collection="train")
    # validation_embeddings = get_last_layer_embedding(multitask_model, trainer, features_dict, collection="validation")

    train_embeddings = get_embeddings(multitask_model, features_dict, collection="train", )
    validation_embeddings = get_embeddings(multitask_model, features_dict, collection="validation", )

    for task_name in ["task_1", "task_2"]:
        train_df = train_dfs[task_name]
        prediction_df = train_predictions[task_name]
        cls_emb_df = train_embeddings[task_name]["cls"]
        mean_emb_df = train_embeddings[task_name]["mean"]
        train_df = pd.concat([train_df, prediction_df, cls_emb_df, mean_emb_df], axis=1, )
        output_path = os.path.join(output_dir, task_name, "train.csv")
        d = os.path.dirname(output_path)
        if not os.path.exists(d):
            os.makedirs(d)
        train_df.to_csv(output_path, encoding="utf-8", index=False)

        val_df = val_dfs[task_name]
        prediction_df = validation_predictions[task_name]
        cls_emb_df = validation_embeddings[task_name]["cls"]
        mean_emb_df = validation_embeddings[task_name]["mean"]
        val_df = pd.concat([val_df, prediction_df, cls_emb_df, mean_emb_df], axis=1)
        output_path = os.path.join(output_dir, task_name, "val.csv")
        d = os.path.dirname(output_path)
        if not os.path.exists(d):
            os.makedirs(d)
        val_df.to_csv(output_path, encoding="utf-8", index=False)


if __name__ == '__main__':
    main()
