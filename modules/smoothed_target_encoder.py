import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

class SmoothedTargetEncoder:
    def __init__(self, categorical_feature, target_feature, m=10, n_splits=5, random_state=42):
        self.categorical_feature = categorical_feature
        self.target_feature = target_feature
        self.m = m
        self.n_splits = n_splits
        self.random_state = random_state
        self.global_mean_ = None
        self.category_mapping_ = None

    def fit(self, df):
        self.global_mean_ = df[self.target_feature].mean()
        agg = df.groupby(self.categorical_feature)[self.target_feature].agg(['count', 'mean'])
        self.category_mapping_ = (
            (agg['count'] * agg['mean'] + self.m * self.global_mean_) /
            (agg['count'] + self.m)
        )
        return self
    
    def transform(self, df_input):
        if isinstance(df_input, pd.Series):
            col_data = df_input
        else:
            col_data = df_input[self.categorical_feature]
            
        return col_data.map(self.category_mapping_).fillna(self.global_mean_)

    def fit_transform(self, df):
        self.global_mean_ = df[self.target_feature].mean()
        
        encoded_feature = pd.Series(index=df.index, dtype=float)
        
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        for train_index, val_index in kf.split(df):
            df_train = df.iloc[train_index]
            df_val = df.iloc[val_index]

            agg = df_train.groupby(self.categorical_feature)[self.target_feature].agg(['count', 'mean'])
            smoothed = (
                (agg['count'] * agg['mean'] + self.m * self.global_mean_) /
                (agg['count'] + self.m)
            )

            encoded_feature.iloc[val_index] = df_val[self.categorical_feature].map(smoothed)

        encoded_feature = encoded_feature.fillna(self.global_mean_)
        self.fit(df)
        
        return encoded_feature