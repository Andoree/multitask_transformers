import os

import pandas as pd
import numpy as np

def main():
    train_df_1 = pd.read_csv(os.path.join("../data", "train_{}.csv".format("sentiments_cloudvision")),
                             encoding="utf-8")
    print(train_df_1)
    train_df_1.dropna(subset=["ocr_text"], inplace=True)
    print()


if __name__ == '__main__':
    main()