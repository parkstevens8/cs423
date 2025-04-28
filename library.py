from __future__ import annotations  # must be first line in your library!
import pandas as pd
import numpy as np
import types
from typing import (
    Dict, Any, Optional, Union, List, Set, Hashable,
    Literal, Tuple, Self, Iterable
)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import sklearn
import warnings

# This sets built-in transformers to output pandas DataFrames
sklearn.set_config(transform_output="pandas")

class CustomMappingTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that maps values in a specified column according to a provided dictionary.

    This transformer follows the scikit-learn transformer interface and can be used in
    a scikit-learn pipeline. It applies value substitution to a specified column using
    a mapping dictionary, which can be useful for encoding categorical variables or
    transforming numeric values.

    Parameters
    ----------
    mapping_column : str or int
        The name (str) or position (int) of the column to which the mapping will be applied.
    mapping_dict : dict
        A dictionary defining the mapping from existing values to new values.
        Keys should be values present in the mapping_column, and values should
        be their desired replacements.

    Attributes
    ----------
    mapping_dict : dict
        The dictionary used for mapping values.
    mapping_column : str or int
        The column (by name or position) that will be transformed.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'category': ['A', 'B', 'C', 'A']})
    >>> mapper = CustomMappingTransformer('category', {'A': 1, 'B': 2, 'C': 3})
    >>> transformed_df = mapper.fit_transform(df)
    >>> transformed_df
       category
    0        1
    1        2
    2        3
    3        1
    """

    def __init__(self, mapping_column: Union[str, int], mapping_dict: Dict[Hashable, Any]) -> None:
        """
        Initialize the CustomMappingTransformer.

        Parameters
        ----------
        mapping_column : str or int
            The name (str) or position (int) of the column to apply the mapping to.
        mapping_dict : Dict[Hashable, Any]
            A dictionary defining the mapping from existing values to new values.

        Raises
        ------
        AssertionError
            If mapping_dict is not a dictionary.
        """
        assert isinstance(mapping_dict, dict), f'{self.__class__.__name__} constructor expected dictionary but got {type(mapping_dict)} instead.'
        self.mapping_dict: Dict[Hashable, Any] = mapping_dict
        self.mapping_column: Union[str, int] = mapping_column  #column to focus on

    def fit(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> Self:
        """
        Fit method - performs no actual fitting operation.

        This method is implemented to adhere to the scikit-learn transformer interface
        but doesn't perform any computation.

        Parameters
        ----------
        X : pandas.DataFrame
            The input data to fit.
        y : array-like, default=None
            Ignored. Present for compatibility with scikit-learn interface.

        Returns
        -------
        self : instance of CustomMappingTransformer
            Returns self to allow method chaining.
        """
        print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
        return self  #always the return value of fit

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the mapping to the specified column in the input DataFrame.

        Parameters
        ----------
        X : pandas.DataFrame
            The DataFrame containing the column to transform.

        Returns
        -------
        pandas.DataFrame
            A copy of the input DataFrame with mapping applied to the specified column.

        Raises
        ------
        AssertionError
            If X is not a pandas DataFrame or if mapping_column is not in X.

        Notes
        -----
        This method provides warnings if:
        1. Keys in mapping_dict are not found in the column values
        2. Values in the column don't have corresponding keys in mapping_dict
        """
        assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
        assert self.mapping_column in X.columns.to_list(), f'{self.__class__.__name__}.transform unknown column "{self.mapping_column}"'  #column legit?
        warnings.filterwarnings('ignore', message='.*downcasting.*')  #squash warning in replace method below

        #now check to see if all keys are contained in column
        column_set: Set[Any] = set(X[self.mapping_column].unique())
        keys_not_found: Set[Any] = set(self.mapping_dict.keys()) - column_set
        if keys_not_found:
            print(f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] does not contain these keys as values {keys_not_found}\n")

        #now check to see if some keys are absent
        keys_absent: Set[Any] = column_set - set(self.mapping_dict.keys())
        if keys_absent:
            print(f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] does not contain keys for these values {keys_absent}\n")

        X_: pd.DataFrame = X.copy()
        X_[self.mapping_column] = X_[self.mapping_column].replace(self.mapping_dict)
        return X_

    def fit_transform(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> pd.DataFrame:
        """
        Fit to data, then transform it.

        Combines fit() and transform() methods for convenience.

        Parameters
        ----------
        X : pandas.DataFrame
            The DataFrame containing the column to transform.
        y : array-like, default=None
            Ignored. Present for compatibility with scikit-learn interface.

        Returns
        -------
        pandas.DataFrame
            A copy of the input DataFrame with mapping applied to the specified column.
        """
        #self.fit(X,y)  #commented out to avoid warning message in fit
        result: pd.DataFrame = self.transform(X)
        return result
    
class CustomOHETransformer(BaseEstimator, TransformerMixin):
    def __init__(self, target_column: str):
        self.target_column = target_column

    def fit(self, X, y=None):
        return self  # no fitting logic required for now

    def transform(self, X):
        # check input
        assert isinstance(X, pd.DataFrame), f'{self.__class__.__name__}.transform expected DataFrame but got {type(X)} instead.'
        assert self.target_column in X.columns, f'{self.__class__.__name__}.transform unknown column {self.target_column}'

        # one-hot encode the target column only
        X_ = X.copy()
        dummies = pd.get_dummies(X_[self.target_column], prefix=self.target_column)
        X_ = X_.drop(columns=[self.target_column])
        X_ = pd.concat([X_, dummies], axis=1)
        return X_
    
class CustomDropColumnsTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that either drops or keeps specified columns in a DataFrame.

    This transformer follows the scikit-learn transformer interface and can be used in
    a scikit-learn pipeline. It allows for selectively keeping or dropping columns
    from a DataFrame based on a provided list.

    Parameters
    ----------
    column_list : List[str]
        List of column names to either drop or keep, depending on the action parameter.
    action : str, default='drop'
        The action to perform on the specified columns. Must be one of:
        - 'drop': Remove the specified columns from the DataFrame
        - 'keep': Keep only the specified columns in the DataFrame

    Attributes
    ----------
    column_list : List[str]
        The list of column names to operate on.
    action : str
        The action to perform ('drop' or 'keep').

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
    >>>
    >>> # Drop columns example
    >>> dropper = CustomDropColumnsTransformer(column_list=['A', 'B'], action='drop')
    >>> dropped_df = dropper.fit_transform(df)
    >>> dropped_df.columns.tolist()
    ['C']
    >>>
    >>> # Keep columns example
    >>> keeper = CustomDropColumnsTransformer(column_list=['A', 'C'], action='keep')
    >>> kept_df = keeper.fit_transform(df)
    >>> kept_df.columns.tolist()
    ['A', 'C']
    """

    def __init__(self, column_list: List[str], action: Literal['drop', 'keep'] = 'drop') -> None:
        """
        Initialize the CustomDropColumnsTransformer.

        Parameters
        ----------
        column_list : List[str]
            List of column names to either drop or keep.
        action : str, default='drop'
            The action to perform on the specified columns.
            Must be either 'drop' or 'keep'.

        Raises
        ------
        AssertionError
            If action is not 'drop' or 'keep', or if column_list is not a list.
        """
        assert action in ['keep', 'drop'], f'DropColumnsTransformer action {action} not in ["keep", "drop"]'
        assert isinstance(column_list, list), f'DropColumnsTransformer expected list but saw {type(column_list)}'
        self.column_list: List[str] = column_list
        self.action: Literal['drop', 'keep'] = action

    #your code below
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame), f'{self.__class__.__name__}.transform expected DataFrame but got {type(X)} instead.'
        X_ = X.copy()

        if self.action == 'keep':
            # Raise an error if trying to keep a column that doesn't exist
            missing = [col for col in self.column_list if col not in X.columns]
            assert not missing, f'{self.__class__.__name__}.transform missing columns requested in "keep": {missing}'
            return X_[self.column_list]

        elif self.action == 'drop':
            # Warn but don't fail if trying to drop a column that doesn't exist
            missing = [col for col in self.column_list if col not in X.columns]
            if missing:
                warnings.warn(f'{self.__class__.__name__}.transform columns not found and skipped in "drop": {missing}')
            return X_.drop(columns=self.column_list, errors='ignore')

class CustomPearsonTransformer(BaseEstimator, TransformerMixin):
    """
    A custom scikit-learn transformer that removes highly correlated features
    based on Pearson correlation.

    Parameters
    ----------
    threshold : float
        The correlation threshold above which features are considered too highly correlated
        and will be removed.

    Attributes
    ----------
    correlated_columns : Optional[List[Hashable]]
        A list of column names that are identified as highly correlated and will be removed.
    """

    def __init__(self, threshold: float = 0.4):
        self.threshold = threshold
        self.correlated_columns: Optional[List[Hashable]] = None

    def fit(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> Self:
        assert isinstance(X, pd.DataFrame), f'{self.__class__.__name__}.fit expected DataFrame but got {type(X)} instead.'

        numeric_df = X.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr().abs()
        mask = np.triu(corr_matrix.values, k=1).astype(bool)

        self.correlated_columns = [
            col for col_idx, col in enumerate(corr_matrix.columns)
            if np.any(mask[:, col_idx])
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        assert self.correlated_columns is not None, f"{self.__class__.__name__}.transform called before fit."

        X_ = X.copy()
        return X_.drop(columns=self.correlated_columns, errors='ignore')

    def fit_transform(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)
class CustomSigma3Transformer(BaseEstimator, TransformerMixin):
    """
    A transformer that applies 3-sigma clipping to a specified column in a pandas DataFrame.

    This transformer follows the scikit-learn transformer interface and can be used in
    a scikit-learn pipeline. It clips values in the target column to be within three standard
    deviations from the mean.

    Parameters
    ----------
    target_column : Hashable
        The name of the column to apply 3-sigma clipping on.

    Attributes
    ----------
    high_wall : Optional[float]
        The upper bound for clipping, computed as mean + 3 * standard deviation.
    low_wall : Optional[float]
        The lower bound for clipping, computed as mean - 3 * standard deviation.
    """

    def __init__(self, target_column: Hashable):
        self.target_column = target_column
        self.low_wall: Optional[float] = None
        self.high_wall: Optional[float] = None

    def fit(self, X: pd.DataFrame, y=None):
        if self.target_column not in X.columns:
            raise ValueError(f"Column '{self.target_column}' not found in DataFrame.")

        if not pd.api.types.is_numeric_dtype(X[self.target_column]):
            raise TypeError(f"Column '{self.target_column}' must be numeric.")

        mean = X[self.target_column].mean()
        std = X[self.target_column].std()

        self.low_wall = mean - 3 * std
        self.high_wall = mean + 3 * std

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.low_wall is None or self.high_wall is None:
            raise ValueError("Transformer has not been fitted yet.")

        X = X.copy()
        X[self.target_column] = X[self.target_column].clip(lower=self.low_wall, upper=self.high_wall)
        return X
    
class CustomTukeyTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that applies Tukey's fences (inner or outer) to a specified column in a pandas DataFrame.

    This transformer follows the scikit-learn transformer interface and can be used in a scikit-learn pipeline.
    It clips values in the target column based on Tukey's inner or outer fences.

    Parameters
    ----------
    target_column : Hashable
        The name of the column to apply Tukey's fences on.
    fence : Literal['inner', 'outer'], default='outer'
        Determines whether to use the inner fence (1.5 * IQR) or the outer fence (3.0 * IQR).

    Attributes
    ----------
    inner_low : Optional[float]
        The lower bound for clipping using the inner fence (Q1 - 1.5 * IQR).
    outer_low : Optional[float]
        The lower bound for clipping using the outer fence (Q1 - 3.0 * IQR).
    inner_high : Optional[float]
        The upper bound for clipping using the inner fence (Q3 + 1.5 * IQR).
    outer_high : Optional[float]
        The upper bound for clipping using the outer fence (Q3 + 3.0 * IQR).

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'values': [10, 15, 14, 20, 100, 5, 7]})
    >>> tukey_transformer = CustomTukeyTransformer(target_column='values', fence='inner')
    >>> transformed_df = tukey_transformer.fit_transform(df)
    >>> transformed_df
    """

    def __init__(self, target_column: Hashable, fence: Literal['inner', 'outer'] = 'outer'):
        self.target_column = target_column
        self.fence = fence

        self.inner_low: Optional[float] = None
        self.inner_high: Optional[float] = None
        self.outer_low: Optional[float] = None
        self.outer_high: Optional[float] = None

    def fit(self, X: pd.DataFrame, y=None):
        if self.target_column not in X.columns:
            raise ValueError(f"Column '{self.target_column}' not found in DataFrame.")
        if not pd.api.types.is_numeric_dtype(X[self.target_column]):
            raise TypeError(f"Column '{self.target_column}' must be numeric.")

        Q1 = X[self.target_column].quantile(0.25)
        Q3 = X[self.target_column].quantile(0.75)
        IQR = Q3 - Q1

        self.inner_low = Q1 - 1.5 * IQR
        self.inner_high = Q3 + 1.5 * IQR
        self.outer_low = Q1 - 3.0 * IQR
        self.outer_high = Q3 + 3.0 * IQR

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.inner_low is None or self.outer_low is None:
            raise ValueError("Transformer has not been fitted yet.")

        X = X.copy()

        if self.fence == 'inner':
            low, high = self.inner_low, self.inner_high
        elif self.fence == 'outer':
            low, high = self.outer_low, self.outer_high
        else:
            raise ValueError(f"Invalid fence type: {self.fence}. Use 'inner' or 'outer'.")

        X[self.target_column] = X[self.target_column].clip(lower=low, upper=high)
        return X

class CustomRobustTransformer(BaseEstimator, TransformerMixin):
  """Applies robust scaling to a specified column in a pandas DataFrame.
    This transformer calculates the interquartile range (IQR) and median
    during the `fit` method and then uses these values to scale the
    target column in the `transform` method.

    Parameters
    ----------
    column : str
        The name of the column to be scaled.

    Attributes
    ----------
    target_column : str
        The name of the column to be scaled.
    iqr : float
        The interquartile range of the target column.
    med : float
        The median of the target column.
  """

  def __init__(self, column):
        self.target_column = column
        self.iqr = None
        self.med = None
        self.is_fitted_ = False  # Track fit status

  def fit(self, X, y=None):
      # Check if column exists
      if self.target_column not in X.columns:
          raise AssertionError(f"CustomRobustTransformer.fit unrecognizable column {self.target_column}.")

      # Extract target column
      col_data = X[self.target_column]
      # Compute median and IQR
      self.med = col_data.median()
      q1 = col_data.quantile(0.25)
      q3 = col_data.quantile(0.75)
      self.iqr = q3 - q1

      self.is_fitted_ = True  # Mark as fitted
      return self  # For chaining

  def transform(self, X):
      if not self.is_fitted_:
          raise NotFittedError(f"This {self.__class__.__name__} instance is not fitted yet. "
                                f"Call 'fit' with appropriate arguments before using this transformer.")

      X = X.copy()

      # Skip if IQR == 0
      if self.iqr == 0 or pd.isna(self.iqr):
          print(f"Skipping transformation for column '{self.target_column}' due to IQR=0")
          return X

      # Apply robust scaling
      X[self.target_column] = (X[self.target_column] - self.med) / self.iqr
      return X
  
gender_mapping = {'Male': 0, 'Female': 1, np.nan: -1} #added for nan. You may want to use a different value
class_mapping = {'Crew': 0, 'C3': 1, 'C2': 2, 'C1': 3, np.nan: -1} #added for nan. You may want to use a different value

#first define the pipeline
titanic_transformer = Pipeline(steps=[
    ('gender', CustomMappingTransformer('Gender', gender_mapping)),
    ('class', CustomMappingTransformer('Class', class_mapping)),
    ('fare', CustomTukeyTransformer(target_column='Fare', fence='outer')),
    ], verbose=True)


customer_transformer = Pipeline(steps=[
    #add drop step below
    ('drop', CustomDropColumnsTransformer(column_list=['ID'], action='drop')),
    ('time spent', CustomTukeyTransformer('Time Spent', 'inner')),
    ], verbose=True)
