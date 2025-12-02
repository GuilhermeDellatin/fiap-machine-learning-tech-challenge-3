import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

class SmoothedTargetEncoder:
    """
    K-Fold Smoothed Target Encoder for categorical features.
    """

    def __init__(self, categorical_feature, target_feature, m=10, n_splits=5, random_state=42):
        self.categorical_feature = categorical_feature
        self.target_feature = target_feature
        self.m = m
        self.n_splits = n_splits
        self.random_state = random_state
        
        # Will be filled during fit
        self.global_mean_ = None
        self.category_mapping_ = None

    def fit(self, df):
        """Computes category → smoothed mean mapping using full dataset."""
        self.global_mean_ = df[self.target_feature].mean()

        agg = df.groupby(self.categorical_feature)[self.target_feature].agg(['count', 'mean'])
        self.category_mapping_ = (
            (agg['count'] * agg['mean'] + self.m * self.global_mean_) /
            (agg['count'] + self.m)
        )

        return self
    
    def transform(self, df):
        """Applies the learned smoothed target encoding."""
        encoded = df[self.categorical_feature].map(self.category_mapping_)
        encoded = encoded.fillna(self.global_mean_)
        return encoded.rename(f"{self.categorical_feature}_Encoded")

    def fit_transform(self, df):
        """K-Fold out-of-fold smoothed target encoding."""
        self.global_mean_ = df[self.target_feature].mean()
        encoded_feature = np.zeros(df.shape[0])
        
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        for train_index, val_index in kf.split(df):
            df_train = df.iloc[train_index]
            df_val = df.iloc[val_index]

            agg = df_train.groupby(self.categorical_feature)[self.target_feature].agg(['count', 'mean'])

            smoothed = (
                (agg['count'] * agg['mean'] + self.m * self.global_mean_) /
                (agg['count'] + self.m)
            )

            encoded_feature[val_index] = df_val[self.categorical_feature].map(smoothed)

        encoded_feature = pd.Series(encoded_feature, name=f"{self.categorical_feature}_Encoded")
        encoded_feature = encoded_feature.fillna(self.global_mean_)
        
        # Fit full mapping for later use on new data
        self.fit(df)
        
        return encoded_feature


   
