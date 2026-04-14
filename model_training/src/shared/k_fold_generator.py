from sklearn.model_selection import StratifiedKFold
from datasets import Dataset
from shared.utils import logging

def generate_folds(dataframe, n_splits=10, seed=42):

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold, (train_idx, test_idx) in enumerate(skf.split(dataframe, dataframe["result"])):

        train_df = dataframe.iloc[train_idx]
        test_df = dataframe.iloc[test_idx]

        train_dataset = Dataset.from_pandas(train_df)
        test_dataset = Dataset.from_pandas(test_df)

        # Internal validation split (10% of train)
        train_val = train_dataset.train_test_split(
            test_size=0.1,
            seed=seed,
        )

        yield {
            "fold": fold,
            "train": train_val["train"],
            "validation": train_val["test"],
            "test": test_dataset
        }