import optuna
from optuna.samplers import TPESampler
import xgboost as xgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import VotingClassifier
import warnings


warnings.filterwarnings('ignore')




class HyperparameterOptimizer:
   """
   Hyperparameter optimization using Optuna for XGBoost, LightGBM, and CatBoost.
   """


   def __init__(self, random_state=42):
       self.random_state = random_state
       self.best_params_ = {}
       self.study_ = None


   def optimize_xgboost(self, X_train, y_train, n_trials=50, cv_folds=5, verbose=True):
       """
       Optimize XGBoost hyperparameters using Optuna.


       Parameters:
       -----------
       X_train : pd.DataFrame or np.ndarray
           Training features
       y_train : pd.Series or np.ndarray
           Training target
       n_trials : int
           Number of optimization trials
       cv_folds : int
           Number of cross-validation folds
       verbose : bool
           Whether to print progress


       Returns:
       --------
       dict : Best hyperparameters found
       """


       # Calculate scale_pos_weight for imbalanced data
       scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)


       def objective(trial):
           params = {
               'objective': 'binary:logistic',
               'eval_metric': 'auc',
               'n_estimators': 1000,
               'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15),
               'max_depth': trial.suggest_int('max_depth', 3, 8),
               'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
               'subsample': trial.suggest_float('subsample', 0.6, 1.0),
               'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
               'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
               'reg_lambda': trial.suggest_float('reg_lambda', 0, 2.0),
               'gamma': trial.suggest_float('gamma', 0, 1.0),
               'scale_pos_weight': scale_pos_weight,
               'tree_method': 'hist',
               'random_state': self.random_state,
               'n_jobs': -1,
               'verbosity': 0
           }


           model = xgb.XGBClassifier(**params)


           cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
           scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)


           return scores.mean()


       sampler = TPESampler(seed=self.random_state)
       self.study_ = optuna.create_study(direction='maximize', sampler=sampler)


       optuna.logging.set_verbosity(optuna.logging.WARNING if not verbose else optuna.logging.INFO)


       self.study_.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)


       self.best_params_['xgboost'] = self.study_.best_params


       if verbose:
           print(f"\n=== XGBoost Optimization Complete ===")
           print(f"Best ROC-AUC (CV): {self.study_.best_value:.4f}")
           print(f"Best Parameters: {self.study_.best_params}")


       return self.study_.best_params


   def optimize_lightgbm(self, X_train, y_train, n_trials=50, cv_folds=5, verbose=True):
       """
       Optimize LightGBM hyperparameters using Optuna.
       """


       # Calculate scale_pos_weight for imbalanced data
       scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)


       def objective(trial):
           params = {
               'objective': 'binary',
               'metric': 'auc',
               'n_estimators': 1000,
               'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15),
               'max_depth': trial.suggest_int('max_depth', 3, 8),
               'num_leaves': trial.suggest_int('num_leaves', 20, 150),
               'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
               'subsample': trial.suggest_float('subsample', 0.6, 1.0),
               'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
               'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
               'reg_lambda': trial.suggest_float('reg_lambda', 0, 2.0),
               'scale_pos_weight': scale_pos_weight,
               'random_state': self.random_state,
               'n_jobs': -1,
               'verbosity': -1
           }


           model = LGBMClassifier(**params)


           cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
           scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)


           return scores.mean()


       sampler = TPESampler(seed=self.random_state)
       study = optuna.create_study(direction='maximize', sampler=sampler)


       optuna.logging.set_verbosity(optuna.logging.WARNING if not verbose else optuna.logging.INFO)


       study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)


       self.best_params_['lightgbm'] = study.best_params


       if verbose:
           print(f"\n=== LightGBM Optimization Complete ===")
           print(f"Best ROC-AUC (CV): {study.best_value:.4f}")
           print(f"Best Parameters: {study.best_params}")


       return study.best_params


   def get_optimized_models(self, X_train, y_train):
       """
       Returns optimized models based on stored best parameters.


       Returns:
       --------
       dict : Dictionary with optimized model instances
       """
       scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)


       models = {}


       # XGBoost with best params
       if 'xgboost' in self.best_params_:
           xgb_params = self.best_params_['xgboost'].copy()
           models['xgboost'] = xgb.XGBClassifier(
               objective='binary:logistic',
               eval_metric='auc',
               n_estimators=5000,
               early_stopping_rounds=100,
               scale_pos_weight=scale_pos_weight,
               tree_method='hist',
               random_state=self.random_state,
               n_jobs=-1,
               **xgb_params
           )


       # LightGBM with best params
       if 'lightgbm' in self.best_params_:
           lgb_params = self.best_params_['lightgbm'].copy()
           models['lightgbm'] = LGBMClassifier(
               objective='binary',
               n_estimators=5000,
               scale_pos_weight=scale_pos_weight,
               random_state=self.random_state,
               n_jobs=-1,
               verbosity=-1,
               **lgb_params
           )


       return models




def create_ensemble(X_train, y_train, random_state=42):
   """
   Create a voting ensemble of XGBoost, LightGBM, and CatBoost.


   Parameters:
   -----------
   X_train : pd.DataFrame or np.ndarray
       Training features (used to calculate scale_pos_weight)
   y_train : pd.Series or np.ndarray
       Training target
   random_state : int
       Random seed


   Returns:
   --------
   VotingClassifier : Ensemble model
   """


   # Calculate scale_pos_weight for imbalanced data
   scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)


   # XGBoost classifier
   xgb_clf = xgb.XGBClassifier(
       objective='binary:logistic',
       eval_metric='auc',
       n_estimators=500,
       learning_rate=0.05,
       max_depth=5,
       min_child_weight=3,
       subsample=0.8,
       colsample_bytree=0.8,
       reg_lambda=1.0,
       scale_pos_weight=scale_pos_weight,
       tree_method='hist',
       random_state=random_state,
       n_jobs=-1
   )


   # LightGBM classifier
   lgb_clf = LGBMClassifier(
       objective='binary',
       n_estimators=500,
       learning_rate=0.05,
       max_depth=5,
       num_leaves=31,
       subsample=0.8,
       colsample_bytree=0.8,
       reg_lambda=1.0,
       scale_pos_weight=scale_pos_weight,
       random_state=random_state,
       n_jobs=-1,
       verbosity=-1
   )


   # CatBoost classifier
   cat_clf = CatBoostClassifier(
       iterations=500,
       learning_rate=0.05,
       depth=5,
       scale_pos_weight=scale_pos_weight,
       random_state=random_state,
       verbose=0
   )


   # Create voting ensemble
   ensemble = VotingClassifier(
       estimators=[
           ('xgb', xgb_clf),
           ('lgb', lgb_clf),
           ('cat', cat_clf)
       ],
       voting='soft'  # Use probabilities for voting
   )


   return ensemble




def get_improved_xgboost(X_train, y_train, random_state=42):
   """
   Returns an improved XGBoost model with better default hyperparameters
   and proper handling of class imbalance.


   Parameters:
   -----------
   X_train : pd.DataFrame or np.ndarray
       Training features (used to calculate scale_pos_weight)
   y_train : pd.Series or np.ndarray
       Training target
   random_state : int
       Random seed


   Returns:
   --------
   XGBClassifier : Configured XGBoost model
   """


   # Calculate scale_pos_weight for imbalanced data
   scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)


   model = xgb.XGBClassifier(
       objective='binary:logistic',
       eval_metric='auc',
       n_estimators=5000,
       learning_rate=0.05,
       max_depth=5,
       min_child_weight=5,
       subsample=0.8,
       colsample_bytree=0.8,
       reg_lambda=1.0,
       reg_alpha=0.1,
       gamma=0.1,
       scale_pos_weight=scale_pos_weight,
       early_stopping_rounds=100,
       tree_method='hist',
       random_state=random_state,
       n_jobs=-1
   )

   return model