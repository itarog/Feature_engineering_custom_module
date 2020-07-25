import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, make_scorer

from itertools import combinations, product


# the function gets df that has a 'date' column containing a timedate datatype,
# and att_list that specify which attributes you want to extract from the column.
# the function returns the dataframe with new columns specified by the attr_list
def time_features_extractor (df, attr_list = ('year', 'month', 'day', 'hour', 'minute', 'second', 'dayofweek')):
    for attr in attr_list:
        df[attr] = df['date'].apply(lambda x: getattr(x, attr))
    return df

# the function gets df, a list of features intended for average, and a name for
# the new column that will be created with the mean values
# the function returns the dataframe with new column containing the mean values
def features_average (df, features_to_average, new_feature_name):
    df[new_feature_name] = np.mean(df[features_to_average], axis=1)
    return df

# custom transformer (sklearn friendly) that preforms min-max scaling using the global
# min-max values in a group of columns, then transforms the values in those columns
# to values between 0 and 1 by those global min-max values

class CustomMinMaxScaler (BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_scale):
        self.columns_to_scale = columns_to_scale

    def fit(self, X, y=None):
        self.min_value_ = self._get_global_min(X[self.columns_to_scale])
        self.max_value_ = self._get_global_max(X[self.columns_to_scale])
        return self

    def transform(self, X, *_):
        norm_value = self.max_value_ - self.min_value_ 
        X[self.columns_to_scale] = X[self.columns_to_scale].apply(lambda x: (x-self.min_value_)/norm_value)
        return X

    def _get_global_max(self, part_X):
        return max(np.max(part_X))

    def _get_global_min(self, part_X):
        return min(np.min(part_X))

# a feature selector tool(obj) contains several functions that are aimed at choosing features
# for a reg_model (provided) from the entire dataset. the main output of this tool is a
# dictionary called models_data that provides data about all models ran by the tool.
# a viewing function is also inculded for easy inspection in dataframe format


class FeatureSelector ():
    def get_model_data (self, reg_model, X_train, X_test, y_train, y_test):
        model_data = {}
        model_data['features'] = set(X_train.columns)
        model_data['parameters'] = reg_model.get_params()
        reg_model.fit(X_train, y_train)
        y_hat_train = reg_model.predict(X_train)
        y_hat_test = reg_model.predict(X_test)
        model_data['Train_RMSE_score'] = round(rmse_loss(y_train, y_hat_train),2)
        model_data['Train_R^2_score'] = round(r2_score(y_train, y_hat_train),2)
        model_data['Test_RMSE_score'] = round(rmse_loss(y_test, y_hat_test),2)
        model_data['Test_R^2_score'] = round(r2_score(y_test, y_hat_test),2)
        return model_data

    def build_and_test_model (self, reg_model, df, target_column):
        X_train, X_test, y_train, y_test = train_test_split(df, target_column, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        return self.get_model_data (reg_model, X_train, X_test, y_train, y_test)

    def feature_iterator (self, df, target_column, reg_model, min_features_num, max_features_num):
        models_data = {}
        for features_num in range(min_features_num, max_features_num):
            possible_features = set(combinations(df.columns, features_num))
            for inner_index, selected_features in enumerate(possible_features):
                model_name = 'reg_model_' + str(features_num) + '_' + str(inner_index)
                models_data[model_name] = self.build_and_test_model (reg_model, df[list(selected_features)], target_column)
        return models_data

    def get_n_best_models(self, models_data, key_parameter, result_goal, n_results):
        best_models_names = []
        temp_models_data = models_data.copy()
        for index in range(n_results):
            if result_goal == 'max':
                current_best_model_name = self.get_model_max_value(temp_models_data, key_parameter)
            elif result_goal == 'min':
                current_best_model_name = self.get_model_min_value(temp_models_data, key_parameter)
            best_models_names.append(current_best_model_name)
            del temp_models_data[current_best_model_name]
        return best_models_names

    def get_model_max_value (self, models_data, key_parameter):
        max_value = 0
        for name, model in models_data.items():
            if model[key_parameter] > max_value:
                max_value = model[key_parameter]
                model_name = name
        return model_name

    def get_model_min_value (self, models_data, key_parameter):
        min_value = 1000000
        for name, model in models_data.items():
            if model[key_parameter] < min_value:
                min_value = model[key_parameter]
                model_name = name
        return model_name

    def models_data_to_df (self, models_data, models_list, selected_keys):
        temp_list = []    
        for model_name in models_list:
            temp_dict = {key: value for key, value in models_data[model_name].items() if key in selected_keys}
            temp_list.append(temp_dict)
        result_df = pd.DataFrame(temp_list, index=models_list)
        return result_df

    def set_parameters (self, reg_model, parameters_name, parameters_value):
        for index, parameter_name in enumerate(parameters_name):
            setattr(reg_model, parameter_name, parameters_value[index])
        return reg_model
    
    def tune_models_parameters (self, df, target_column, reg_model, models_data, models_list, parameters_to_tune):
        parameter_matrix = product(*parameters_to_tune.values())
        new_models_data = models_data.copy()
        for model_name in models_list:
            selected_features = models_data[model_name].get('features')
            for inner_index, parameters in enumerate(parameter_matrix):
                reg_model = self.set_parameters(reg_model, parameters_to_tune.keys() ,parameters)
                new_model_name = model_name + '_' + str(inner_index)
                new_models_data[new_model_name] = self.build_and_test_model (reg_model, df[list(selected_features)], target_column)
        return new_models_data
