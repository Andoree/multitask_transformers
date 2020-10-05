import nlp
import pandas as pd
import os


def preprocess_dataset(dataset_df, text_column_name, class_to_id_dict, class_column="class"):
    dataset_df.dropna(subset=[text_column_name], inplace=True)
    dataset_df["label"] = dataset_df[class_column].apply(lambda x: class_to_id_dict[x])
    dataset_df.rename(columns={text_column_name: "text"}, inplace=True)
    dataset_df.text = dataset_df.text.apply(lambda x: x.replace("\n", " "))
    dataset_df = dataset_df[["text", "label"]]
    dataset_nlp = nlp.arrow_dataset.Dataset.from_pandas(dataset_df)

    return dataset_nlp


def load_dataset(train_df, val_df, text_column_name, class_column="class"):
    classes = set(train_df[class_column].unique().tolist())
    classes.update(set(val_df[class_column].unique().tolist()))
    id_to_class = {i: cl for i, cl in enumerate(classes)}
    class_to_id = {cl: i for i, cl in id_to_class.items()}
    train_dataset = preprocess_dataset(train_df, text_column_name, class_to_id, )
    val_dataset = preprocess_dataset(val_df, text_column_name, class_to_id, )
    datasets_dict = {
        "train": train_dataset,
        "validation": val_dataset
    }
    return datasets_dict, id_to_class


def convert_features_function(tokenizer, max_seq_length):
    def convert_to_features(example_batch):
        inputs = list(example_batch['text'])
        features = tokenizer.batch_encode_plus(
            inputs, max_length=max_seq_length, pad_to_max_length=True
        )
        features["labels"] = example_batch["label"]
        return features

    return convert_to_features


def data_to_features(dataset_dict, convert_func_dict, columns_dict):
    features_dict = {}
    for task_name, dataset in dataset_dict.items():
        features_dict[task_name] = {}
        for phase, phase_dataset in dataset.items():
            features_dict[task_name][phase] = phase_dataset.map(
                convert_func_dict[task_name],
                batched=True,
                load_from_cache_file=False,
            )
            print(task_name, phase, len(phase_dataset), len(features_dict[task_name][phase]))
            features_dict[task_name][phase].set_format(
                type="torch",
                columns=columns_dict[task_name],
            )
            print(task_name, phase, len(phase_dataset), len(features_dict[task_name][phase]))
    return features_dict

