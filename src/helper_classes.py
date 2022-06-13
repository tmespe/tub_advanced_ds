from tabnanny import verbose
from typing import List
import matplotlib.pyplot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import plotly.express as px

# import scikit learn libraries
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor# you may need to install XGBoost
from sklearn.metrics import accuracy_score, f1_score, precision_score, r2_score,mean_absolute_error, mean_squared_error, recall_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer


from yellowbrick.model_selection import learning_curve

plt.style.use('seaborn')


class EDA:
    """
    Create a new EDA object from a csv or pickle file. Allows for structured EDA on csv datasets with
    methods for summarising, cleaning and plotting data.

    Takes a CSV file as input in the format of a string to the file's location
    """

    def __init__(self, file: str):
        self.target_type = None
        self.target_var = None
        self.target_name = None
        self.df = self.load_file(file)  # Turn the file into a pandas dataframe
        self.rows = self.df.shape[0]
        self.cols = self.df.shape[1]
        self.df_clean = self.df

    @staticmethod
    def load_file(file: str) -> pd.DataFrame:
        file_extension = EDA.check_extension(file)

        if file_extension == ".csv":
            return pd.read_csv(file)
        elif file_extension == ".pkl":
            return pd.read_pickle(file)
        else:
            raise ValueError(f"{file_extension} is an invalid file type. Please provide a CSV or pkl file")

    def __repr__(self) -> str:
        """
        Prints the number of rows and columns of the dataset
        :return: String with n_rows and n_cols
        """
        return f"A dataframe with {self.rows} rows and {self.cols} columns"

    def summary(self) -> None:
        """
        Prints summary of dataset as well as descriptive statistics
        :return: None
        """
        self.print_header("Info")  # Print a header to highlight what is being printed
        print(self.df_clean.info())

        self.print_header("Sample rows")  # Print a header to highlight what is being printed
        print(self.df_clean.sample(10))

        self.print_header("Description")  # Print a header to highlight what is being printed
        print(self.df_clean.describe())

        self.print_header("Standard deviation")  # Print a header to highlight what is being printed
        print(self.df_clean.std())

    def info(self) -> None:
        """
        Returns the dataframe info.

        :return: Pandas dataframe with dataframe info
        """
        return self.df_clean.info()

    def describe(self, include="all"):
        """
        Returns the dataframes description, by default includes all columns regardless of dtype.

        :return: Pandas dataframe with dataframe description
        """
        return self.df_clean.describe(datetime_is_numeric=True, include=include)  # Include all dtypes

    def std(self) -> pd.DataFrame:
        """
        Returns dataframe standard deviations for each column

        :return: Pandas dataframe with standard deviations
        """
        return self.df_clean.std().round(2)

    def remove_low_variance_cols(self, threshhold : float =0.1) -> pd.DataFrame:
        """Removes columns having variance lower than the set threshold

        :param threshhold: threshold to consider as low variance, defaults to 0.1
        :type threshhold: float, optional
        :return: Dataframe with low variance columns removed
        :rtype: pd.Dataframe()
        """
        low_variance = self.low_variance()
        low_var_cols = low_variance.index.to_list()
        self.df_clean = self.df_clean.drop(columns=low_var_cols)

        print(f"Removed columns '{','.join(low_var_cols)}' having variance below {threshhold}")
        return self.df_clean

    def low_variance(self, threshold: float = 0.2) -> pd.DataFrame:
        """Displays columns with low variance

        :return: Dataframe showing columns with low variance
        :rtype: pd.Dataframe
        """
        variance = self.std()
        low_variance = variance[variance < threshold]

        print(low_variance)
        return low_variance

    @staticmethod
    def print_header(title: str, sep: str = "__", n_sep: int = 15) -> None:
        """
        Helper function to print a header to separate text. Accepts a character to use as separator
        and the number of times separator should be repeated on each side of the header
        :param title: Title of the header
        :param sep: Character to use as separator. _ by default
        :param n_sep: Times separator should be printed on each side of header. 15 by default
        :return: None
        """
        return print(f'{sep * n_sep} {title} {sep * n_sep} \n')

    def target(self, target_name: str, target_type: str) -> None:
        """
        Sets the target variable
        :param target_name: Name of target variable in dataset
        :param target_type: Type of variable cat (categorical) or cont (continuous)
        :return: None
        """
        self.target_name = target_name
        try:
            self.target_var = self.df[target_name]
            self.target_type = target_type
            self.df_clean = self.df_clean.drop(columns=target_name)

            if target_type == "cat":
                self.df["target_dummy"] = pd.factorize(self.df[target_name])[0]
                self.target_var = self.df["target_dummy"]

        except KeyError:
            raise KeyError("Target variable not in Dataset")

    def plot_target(self) -> matplotlib.pyplot.plot:
        """
        Plots distribution of the target variable
        :return:
        """
        if self.target_type == "cat":
            fig = px.bar(self.df[self.target_name].value_counts(),
                          title="Distribution of categorical target variable", text=self.df[self.target_name].value_counts(),
                          color=self.df[self.target_name].value_counts(),  
                          labels={"value": "Count", "index": "Category"},  # Change x and y label from default to something more meaningful
                          ) 
            fig.update_coloraxes(showscale=False)  # Hide unnecessary color scale
            return fig
            #self.target_var.value_counts().plot(kind="bar")
        else:
            return px.histogram(self.target_var, title="Histogram of target variable")

    def plot_target_corr(self, cols : List = None) -> px.imshow:
        """Plots correlations between target column and passed columns. If no columns
        are passed plots target against all other columns.

        :param cols: List of columns to plot target against, defaults to None
        :type cols: List, optional
        :return: Plotly imshow correlation heatmap
        :rtype: px.imshow
        """
        if not cols:  # If no columns passed plot correlations for all variables
            return px.imshow(self.correlations(), title="Correlations between target and all columns", text_auto=True)
        else:
            cols.append(self.target_var.name)  # Add target name to cols for df subsetting
            # Plot correlationsplot using plotly with text_auto for values in cells
            return px.imshow(self.df[cols].corr(), title="Correlations between target and columns", text_auto=True)

    def num_cats(self) -> pd.DataFrame:
        """
        Returns the number of categories for all object columns.
        :return: Pandas dataframe with number of categories in each column
        """
        try:
            cat_features = self.df_clean.select_dtypes("object").nunique(dropna=False).sort_values(ascending=False)
            return cat_features
        except TypeError as e:
            raise TypeError("No categorical variables")

    # def remove_cats(self, n_cats):
    #     num_cats = self.num_cats()

    #     print(num_cats > n_cats)

    def correlations(self) -> pd.DataFrame:
        """Gets rounded correlations for dataframe

        :return: Rounded correalation matrix
        :rtype: Dataframe
        """
        return self.df_clean.corr().round(2)

    def remove_highly_corr(self, threshhold: float = 0.6, exclude:  list=[]) -> pd.DataFrame:
        """Removes highly correlated features according to set treshhold

        :param threshhold: Which treshhold to consider for droppig, defaults to 0.6
        :type threshhold: float, optional
        """
        corr = self.df_clean.drop(columns=exclude).corr().abs()  # Create a correlation matrix and drop exluced. Use abs to turn all positive for threshold comparison
        corr_df = pd.DataFrame(corr.unstack())
        corr_df = corr_df.rename(columns={0: "correlation"})
        corr_filter = corr_df.query('correlation > @threshhold & correlation != 1')
        print(corr_filter)
        unique_corrs = set(corr_filter.index.get_level_values(0).join(corr_filter.index.get_level_values(1)))

        self.df_clean = self.df_clean.drop(columns=unique_corrs)
        return self.df_clean
        #return corr_filter.index.get_level_values(0).join(corr_filter.index.get_level_values(1))

    def plot_heatmap(self) -> px.imshow:
        """Plots heatmap of all feature correlations

        :return: Heatmap of all feature correlations
        :rtype: px.imshow 
        """
        correlations = self.correlations()

        return px.imshow(correlations, text_auto=True, title="Heatmap of feature correlations")

    def identify_missing_data(self):
        """
        This function is used to identify missing data

        :param df pandas DataFrame

        :return a DataFrame with the percentage of missing data for every feature and the data types
        """
        missing_pct = (self.df_clean.isna().mean() * 100).round(2)
        dtypes = self.df_clean.dtypes

        pct_missing_df = (pd.concat([missing_pct, dtypes], axis=1)
                          .rename(columns={0: "percent_missing", 1: "dtype"})
                          .sort_values("percent_missing", ascending=False))

        return pct_missing_df

    def remove_missing_data(self, threshold: float =0.6) -> pd.DataFrame:
        """Removes columns with missing data over threshold

        :param threshold: Threshhold of missing of which to remove columns, defaults to 0.6
        :type threshold: float, optional
        :return: Pandas dataframe with removed columns that have missing data above threshold
        :rtype: pd.dataframe
        """
        df_missing = self.identify_missing_data()
        missing_cols = df_missing[df_missing["percent_missing"] >= threshold].index.to_list()

        self.df_clean = self.df_clean.drop(columns=missing_cols)
        return self.df_clean

    @staticmethod
    def check_extension(file: str) -> str:
        """Returns file extension

        :param file: Path to file
        :type file: str
        :return: File extension
        :rtype: str
        """
        try:
            return Path(file).suffix
        except FileNotFoundError as e:
            return file


class Model:
    def __init__(self, target, features, model):
        self.target = target
        self.features = features
        self.model = model
        self.scores = {}
        self.pipeline = None
        self.model_name = self.model.__class__.__name__
        self.training_data = None
        self.testing_data = None
        self.trained_model = None
        self.metrics = None

    def split_and_print_shape(self) -> None:
        X_train, X_test, y_train, y_test = train_test_split(self.features, self.target, random_state=42, test_size=0.2)

        print(f"X_train shape {X_train.shape}")
        print(f"X_test shape {X_test.shape}")
        print(f"y_train shape {y_train.shape}")
        print(f"y_test shape {y_test.shape}")

        self.training_data = X_train, y_train
        self.testing_data = X_test, y_test

    def print_5(self, y_pred):
        print("first five predicted values:", y_pred[0:5])
        print("first five actual values:", list(self.testing_data[1][0:5]))

    def one_hot(self):
        cats = self.features.select_dtypes("object").columns.to_list()
        transformer = make_column_transformer(
            (OneHotEncoder(), cats),
            remainder='passthrough')

        transformed = transformer.fit_transform(self.features)
        transformed_df = pd.DataFrame(
            transformed, 
            columns=transformer.get_feature_names_out()
        )
        self.features = transformed_df
        return self.features
        # encoder = OneHotEncoder()
        # self.df_clean = encoder.fit_transform(self.df_clean.select_dtypes("object"))
        # return self.df_clean

    def train_model(self):
        self.one_hot()
        self.split_and_print_shape()
        model = self.model

        model.fit(*self.training_data)
        y_pred = model.predict(self.testing_data[0])

        self.trained_model = model
        self.print_5(y_pred)
        return model, y_pred

    def evaluate_model(self, type, plot_learning_curve=True):
        model = self.train_model()
        y_test = self.testing_data[1]

        if type == "reg":
            self.metrics = ["r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"]
            
            r2 = r2_score(y_test, model[1])
            print("R2:", round(r2, 2))
            mae = mean_absolute_error(y_test, model[1])
            print("Mean Absolute Error:", round(mae))
            mse = mean_squared_error(y_test, model[1])
            print("Mean Squared Error:", round(mse))

            self.scores = pd.DataFrame(index=["R2", "MAE", "MSE"], data=[r2, mae, mse], columns=[self.model_name])
        else:
            self.metrics = ["f1", "accuracy", "precision"]
            f1 = f1_score(y_test, model[1])
            accuracy = accuracy_score(y_test, model[1])
            precision = precision_score(y_test, model[1])
            recall = recall_score(y_test, model[1])

            print(f"Accuracy: {accuracy}")
            print(f"Precision: {precision}")
            print(f"Recall: {recall}")
            print(f"F1: {f1}")

            self.scores = pd.DataFrame(index=["Accuracy", "Precision", "Recall", "F1"], data=[accuracy, precision, recall, f1], columns=[self.model_name])

        if plot_learning_curve:
            for score in self.metrics:
                learning_curve(model[0], self.features, self.target, cv=10, scoring=score, verbose=0)
                #plt.title(f"Learning curves for {self.model_name} with {score} as metric")

    def feature_importance_plot(self, n=5):
        """Plots feature importance - this only works for Decision Tree based Models"""
        plt.figure(figsize=(8, 5)) # set figure size
        feat_importances = pd.Series(self.model.feature_importances_,
                                    index = self.training_data[0].columns)
        feat_importances.nlargest(n).plot(kind = 'bar')
        plt.title(f"Top {n} Features for {self.model_name}")
        plt.xticks(rotation=45)
        plt.show()