import pandas as pd

if __name__ == '__main__':
    path = r"results_bert_4/task_1/train/.csv"
    df = pd.read_csv(path, )
    print(df)
    print(df.columns)