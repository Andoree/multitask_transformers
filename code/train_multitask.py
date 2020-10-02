import logging
from argparse import ArgumentParser

import transformers

from code.evaluation import evaluate_classification
from code.multitask_model import MultitaskModel, NLPDataCollator, MultitaskTrainer
from code.multitask_preprocessing import load_dataset, convert_features_function, data_to_features

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
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    corpus_dir = args.corpus_dir
    text_column_name = args.text_column
    model_name = args.model_name
    max_seq_length = args.max_seq_length
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    batch_size = args.batch_size

    dataset_dict_1, id_to_class_1 = load_dataset(corpus_dir, "sentiments_cloudvision", text_column_name)
    dataset_dict_2, id_to_class_2 = load_dataset(corpus_dir, "topics_cloudvision", text_column_name)
    dataset_dict = {
        "task_1": dataset_dict_1,
        "task_2": dataset_dict_2
    }
    id_to_class_dicts = {
        "task_1": id_to_class_1,
        "task_2": id_to_class_2
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
            num_train_epochs=1,
            # Adjust batch size if this doesn't fit on the Colab GPU
            per_device_train_batch_size=batch_size,
            save_steps=3000,
        ),
        # compute_metrics=classification_metrics,
        data_collator=NLPDataCollator(),
        train_dataset=train_dataset,
        eval_dataset=val_dataset_dict
    )
    for epoch_number in range(num_epochs):
        validation_results = evaluate_classification(trainer, features_dict, dataset_dict)
        for task_name, results_dict in validation_results.items():
            for metric_name, value in results_dict.items():
                print(f"Validation quality: Epoch {epoch_number}, {metric_name} : {value}")
        trainer.train()


if __name__ == '__main__':
    main()
