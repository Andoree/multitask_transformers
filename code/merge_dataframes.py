import os
from argparse import ArgumentParser

import pandas as pd


def main():
    parser = ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)
    train_path = os.path.join(input_dir, "train.csv")
    cls_emb_path = os.path.join(input_dir, "tr_cls_emb.csv")
    mean_emb_path = os.path.join(input_dir, "tr_mean_emb.csv")
    prediction_path = os.path.join(input_dir, "tr_prediction.csv")

    train_df = pd.read_csv(train_path, encoding="utf-8")
    print("train", train_df)
    train_df.dropna(subset=["ocr_text"], inplace=True)
    print("train", train_df)
    cls_emb_df = pd.read_csv(cls_emb_path, encoding="utf-8")
    print("train cls", cls_emb_df)
    mean_emb_df = pd.read_csv(mean_emb_path, encoding="utf-8")
    print("train mean", mean_emb_df)
    prediction_df = pd.read_csv(prediction_path, encoding="utf-8")
    print("train prediction", prediction_df)

    train_result_df = pd.concat([train_df, cls_emb_df, mean_emb_df, prediction_df], axis=1, ignore_index=True)
    print("result train", train_result_df)

    val_path = os.path.join(input_dir, "val.csv")
    cls_emb_path = os.path.join(input_dir, "val_cls_emb.csv")
    mean_emb_path = os.path.join(input_dir, "val_mean_emb.csv")
    prediction_path = os.path.join(input_dir, "val_prediction.csv")

    val_df = pd.read_csv(val_path, encoding="utf-8")
    print("val", val_df)
    val_df.dropna(subset=["ocr_text"], inplace=True)
    print("val", val_df)
    cls_emb_df = pd.read_csv(cls_emb_path, encoding="utf-8")
    print("val cls", cls_emb_df)
    mean_emb_df = pd.read_csv(mean_emb_path, encoding="utf-8")
    print("val mean", mean_emb_df)
    prediction_df = pd.read_csv(prediction_path, encoding="utf-8")
    print("val prediction", prediction_df)

    val_result_df = pd.concat([val_df, cls_emb_df, mean_emb_df, prediction_df], axis=1, ignore_index=True)
    print("result val", val_result_df)
    output_train_path = os.path.join(output_dir, "result_train.csv")
    output_val_path = os.path.join(output_dir, "result_val.csv")

    train_result_df.to_csv(output_train_path, index=False, encoding="utf-8")
    val_result_df.to_csv(output_val_path, index=False, encoding="utf-8")


if __name__ == '__main__':
    main()
