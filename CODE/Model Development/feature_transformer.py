import numbers
import warnings
from collections import Counter
import pandas as pd
import numpy as np
import numpy.ma as ma
from scipy import sparse as sp
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin, _OneToOneFeatureMixin
from sklearn.utils.sparsefuncs import _get_median
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import FLOAT_DTYPES
from sklearn.utils._mask import _get_mask
from sklearn.utils import is_scalar_nan
import category_encoders.utils as util
from sklearn.utils.random import check_random_state
from category_encoders.ordinal import OrdinalEncoder
import holidays

from sklearn.utils import check_array, is_scalar_nan
from sklearn.utils.deprecation import deprecated
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import _check_feature_names_in
from sklearn.utils._encode import _encode, _check_unknown, _unique


__all__ = ["OneHotEncoder", "OrdinalEncoder"]

def _get_input_features(input_features, n_features_in):
    if input_features is not None:
        return np.array(input_features, dtype=object)
    return np.array([f"X{i}" for i in range(n_features_in)], dtype=object)


def _check_inputs_dtype(X, missing_values):
    if X.dtype.kind in ("f", "i", "u") and not isinstance(missing_values, numbers.Real):
        raise ValueError(
            "'X' and 'missing_values' types are expected to be"
            " both numerical. Got X.dtype={} and "
            " type(missing_values)={}.".format(X.dtype, type(missing_values))
        )

def _most_frequent(array, extra_value, n_repeat):
    """Compute the most frequent value in a 1d array extended with
    [extra_value] * n_repeat, where extra_value is assumed to be not part
    of the array."""
    # Compute the most frequent value in array only
    if array.size > 0:
        if array.dtype == object:
            # scipy.stats.mode is slow with object dtype array.
            # Python Counter is more efficient
            counter = Counter(array)
            most_frequent_count = counter.most_common(1)[0][1]
            # tie breaking similarly to scipy.stats.mode
            most_frequent_value = min(
                value
                for value, count in counter.items()
                if count == most_frequent_count
            )
        else:
            mode = stats.mode(array)
            most_frequent_value = mode[0][0]
            most_frequent_count = mode[1][0]
    else:
        most_frequent_value = 0
        most_frequent_count = 0

    # Compare to array + [extra_value] * n_repeat
    if most_frequent_count == 0 and n_repeat == 0:
        return np.nan
    elif most_frequent_count < n_repeat:
        return extra_value
    elif most_frequent_count > n_repeat:
        return most_frequent_value
    elif most_frequent_count == n_repeat:
        # tie breaking similarly to scipy.stats.mode
        return min(most_frequent_value, extra_value)


class _BaseImputer(TransformerMixin, BaseEstimator):
    """Base class for all imputers.

    It adds automatically support for `add_indicator`.
    """

    def __init__(self, *, missing_values=np.nan, add_indicator=False):
        self.missing_values = missing_values
        self.add_indicator = add_indicator

    def _fit_indicator(self, X):
        """Fit a MissingIndicator."""
        if self.add_indicator:
            self.indicator_ = MissingIndicator(
                missing_values=self.missing_values, error_on_new=False
            )
            self.indicator_._fit(X, precomputed=True)
        else:
            self.indicator_ = None

    def _transform_indicator(self, X):
        """Compute the indicator mask.'

        Note that X must be the original data as passed to the imputer before
        any imputation, since imputation may be done inplace in some cases.
        """
        if self.add_indicator:
            if not hasattr(self, "indicator_"):
                raise ValueError(
                    "Make sure to call _fit_indicator before _transform_indicator"
                )
            return self.indicator_.transform(X)

    def _concatenate_indicator(self, X_imputed, X_indicator):
        """Concatenate indicator mask with the imputed data."""
        if not self.add_indicator:
            return X_imputed

        hstack = sp.hstack if sp.issparse(X_imputed) else np.hstack
        if X_indicator is None:
            raise ValueError(
                "Data from the missing indicator are not provided. Call "
                "_fit_indicator and _transform_indicator in the imputer "
                "implementation."
            )

        return hstack((X_imputed, X_indicator))

    def _more_tags(self):
        return {"allow_nan": is_scalar_nan(self.missing_values)}



class Debug(BaseEstimator, TransformerMixin):

    def transform(self, X):
        self.X_shape = X.shape
        self.X_df = X
        # print(self.shape)
        # print(self.df)
        return X

    def fit(self, X, y=None):
        
        # assert pd.isnull(X[self.group_cols]).any(axis=None) == False, 'There are missing values in group_cols'
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns)
        else:
            self.feature_names_in_ = None
        self.n_features_in_ = X.shape[1]
        
        return self 
    
    def get_feature_names_out(self, input_features=None):
        # uses input_features if self.feature_names_in_ is None
        if self.feature_names_in_ is None:
            return _get_input_features(input_features, self.n_features_in_)
        return self.feature_names_in_
    
    
class SimpleImputer(_BaseImputer):
    """Imputation transformer for completing missing values.

    Read more in the :ref:`User Guide <impute>`.

    .. versionadded:: 0.20
       `SimpleImputer` replaces the previous `sklearn.preprocessing.Imputer`
       estimator which is now removed.

    Parameters
    ----------
    missing_values : int, float, str, np.nan or None, default=np.nan
        The placeholder for the missing values. All occurrences of
        `missing_values` will be imputed. For pandas' dataframes with
        nullable integer dtypes with missing values, `missing_values`
        should be set to `np.nan`, since `pd.NA` will be converted to `np.nan`.

    strategy : str, default='mean'
        The imputation strategy.

        - If "mean", then replace missing values using the mean along
          each column. Can only be used with numeric data.
        - If "median", then replace missing values using the median along
          each column. Can only be used with numeric data.
        - If "most_frequent", then replace missing using the most frequent
          value along each column. Can be used with strings or numeric data.
          If there is more than one such value, only the smallest is returned.
        - If "constant", then replace missing values with fill_value. Can be
          used with strings or numeric data.

        .. versionadded:: 0.20
           strategy="constant" for fixed value imputation.

    fill_value : str or numerical value, default=None
        When strategy == "constant", fill_value is used to replace all
        occurrences of missing_values.
        If left to the default, fill_value will be 0 when imputing numerical
        data and "missing_value" for strings or object data types.

    verbose : int, default=0
        Controls the verbosity of the imputer.

    copy : bool, default=True
        If True, a copy of `X` will be created. If False, imputation will
        be done in-place whenever possible. Note that, in the following cases,
        a new copy will always be made, even if `copy=False`:

        - If `X` is not an array of floating values;
        - If `X` is encoded as a CSR matrix;
        - If `add_indicator=True`.

    add_indicator : bool, default=False
        If True, a :class:`MissingIndicator` transform will stack onto output
        of the imputer's transform. This allows a predictive estimator
        to account for missingness despite imputation. If a feature has no
        missing values at fit/train time, the feature won't appear on
        the missing indicator even if there are missing values at
        transform/test time.

    Attributes
    ----------
    statistics_ : array of shape (n_features,)
        The imputation fill value for each feature.
        Computing statistics can result in `np.nan` values.
        During :meth:`transform`, features corresponding to `np.nan`
        statistics will be discarded.

    indicator_ : :class:`~sklearn.impute.MissingIndicator`
        Indicator used to add binary indicators for missing values.
        `None` if `add_indicator=False`.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    IterativeImputer : Multivariate imputation of missing values.

    Notes
    -----
    Columns which only contained missing values at :meth:`fit` are discarded
    upon :meth:`transform` if strategy is not `"constant"`.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.impute import SimpleImputer
    >>> imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    >>> imp_mean.fit([[7, 2, 3], [4, np.nan, 6], [10, 5, 9]])
    SimpleImputer()
    >>> X = [[np.nan, 2, 3], [4, np.nan, 6], [10, np.nan, 9]]
    >>> print(imp_mean.transform(X))
    [[ 7.   2.   3. ]
     [ 4.   3.5  6. ]
     [10.   3.5  9. ]]
    """

    def __init__(
        self,
        *,
        missing_values=np.nan,
        strategy="mean",
        fill_value=None,
        verbose=0,
        copy=True,
        add_indicator=False,
    ):
        super().__init__(missing_values=missing_values, add_indicator=add_indicator)
        self.strategy = strategy
        self.fill_value = fill_value
        self.verbose = verbose
        self.copy = copy

    def _validate_input(self, X, in_fit):
        allowed_strategies = ["mean", "median", "most_frequent", "constant"]
        if self.strategy not in allowed_strategies:
            raise ValueError(
                "Can only use these strategies: {0}  got strategy={1}".format(
                    allowed_strategies, self.strategy
                )
            )

        if self.strategy in ("most_frequent", "constant"):
            # If input is a list of strings, dtype = object.
            # Otherwise ValueError is raised in SimpleImputer
            # with strategy='most_frequent' or 'constant'
            # because the list is converted to Unicode numpy array
            if isinstance(X, list) and any(
                isinstance(elem, str) for row in X for elem in row
            ):
                dtype = object
            else:
                dtype = None
        else:
            dtype = FLOAT_DTYPES

        if not is_scalar_nan(self.missing_values):
            force_all_finite = True
        else:
            force_all_finite = "allow-nan"

        try:
            X = self._validate_data(
                X,
                reset=in_fit,
                accept_sparse="csc",
                dtype=dtype,
                force_all_finite=force_all_finite,
                copy=self.copy,
            )
        except ValueError as ve:
            if "could not convert" in str(ve):
                new_ve = ValueError(
                    "Cannot use {} strategy with non-numeric data:\n{}".format(
                        self.strategy, ve
                    )
                )
                raise new_ve from None
            else:
                raise ve

        _check_inputs_dtype(X, self.missing_values)
        if X.dtype.kind not in ("i", "u", "f", "O"):
            raise ValueError(
                "SimpleImputer does not support data with dtype "
                "{0}. Please provide either a numeric array (with"
                " a floating point or integer dtype) or "
                "categorical data represented either as an array "
                "with integer dtype or an array of string values "
                "with an object dtype.".format(X.dtype)
            )

        return X

    def fit(self, X, y=None):
        """Fit the imputer on `X`.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Input data, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = self._validate_input(X, in_fit=True)

        # default fill_value is 0 for numerical input and "missing_value"
        # otherwise
        if self.fill_value is None:
            if X.dtype.kind in ("i", "u", "f"):
                fill_value = 0
            else:
                fill_value = "missing_value"
        else:
            fill_value = self.fill_value

        # fill_value should be numerical in case of numerical input
        if (
            self.strategy == "constant"
            and X.dtype.kind in ("i", "u", "f")
            and not isinstance(fill_value, numbers.Real)
        ):
            raise ValueError(
                "'fill_value'={0} is invalid. Expected a "
                "numerical value when imputing numerical "
                "data".format(fill_value)
            )

        if sp.issparse(X):
            # missing_values = 0 not allowed with sparse data as it would
            # force densification
            if self.missing_values == 0:
                raise ValueError(
                    "Imputation not possible when missing_values "
                    "== 0 and input is sparse. Provide a dense "
                    "array instead."
                )
            else:
                self.statistics_ = self._sparse_fit(
                    X, self.strategy, self.missing_values, fill_value
                )

        else:
            self.statistics_ = self._dense_fit(
                X, self.strategy, self.missing_values, fill_value
            )
            
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns)
        else:
            self.feature_names_in_ = None
        self.n_features_in_ = X.shape[1]

        return self

    def _sparse_fit(self, X, strategy, missing_values, fill_value):
        """Fit the transformer on sparse data."""
        missing_mask = _get_mask(X, missing_values)
        mask_data = missing_mask.data
        n_implicit_zeros = X.shape[0] - np.diff(X.indptr)

        statistics = np.empty(X.shape[1])

        if strategy == "constant":
            # for constant strategy, self.statistcs_ is used to store
            # fill_value in each column
            statistics.fill(fill_value)
        else:
            for i in range(X.shape[1]):
                column = X.data[X.indptr[i] : X.indptr[i + 1]]
                mask_column = mask_data[X.indptr[i] : X.indptr[i + 1]]
                column = column[~mask_column]

                # combine explicit and implicit zeros
                mask_zeros = _get_mask(column, 0)
                column = column[~mask_zeros]
                n_explicit_zeros = mask_zeros.sum()
                n_zeros = n_implicit_zeros[i] + n_explicit_zeros

                if strategy == "mean":
                    s = column.size + n_zeros
                    statistics[i] = np.nan if s == 0 else column.sum() / s

                elif strategy == "median":
                    statistics[i] = _get_median(column, n_zeros)

                elif strategy == "most_frequent":
                    statistics[i] = _most_frequent(column, 0, n_zeros)
        super()._fit_indicator(missing_mask)

        return statistics

    def _dense_fit(self, X, strategy, missing_values, fill_value):
        """Fit the transformer on dense data."""
        missing_mask = _get_mask(X, missing_values)
        masked_X = ma.masked_array(X, mask=missing_mask)

        super()._fit_indicator(missing_mask)

        # Mean
        if strategy == "mean":
            mean_masked = np.ma.mean(masked_X, axis=0)
            # Avoid the warning "Warning: converting a masked element to nan."
            mean = np.ma.getdata(mean_masked)
            mean[np.ma.getmask(mean_masked)] = np.nan

            return mean

        # Median
        elif strategy == "median":
            median_masked = np.ma.median(masked_X, axis=0)
            # Avoid the warning "Warning: converting a masked element to nan."
            median = np.ma.getdata(median_masked)
            median[np.ma.getmaskarray(median_masked)] = np.nan

            return median

        # Most frequent
        elif strategy == "most_frequent":
            # Avoid use of scipy.stats.mstats.mode due to the required
            # additional overhead and slow benchmarking performance.
            # See Issue 14325 and PR 14399 for full discussion.

            # To be able access the elements by columns
            X = X.transpose()
            mask = missing_mask.transpose()

            if X.dtype.kind == "O":
                most_frequent = np.empty(X.shape[0], dtype=object)
            else:
                most_frequent = np.empty(X.shape[0])

            for i, (row, row_mask) in enumerate(zip(X[:], mask[:])):
                row_mask = np.logical_not(row_mask).astype(bool)
                row = row[row_mask]
                most_frequent[i] = _most_frequent(row, np.nan, 0)

            return most_frequent

        # Constant
        elif strategy == "constant":
            # for constant strategy, self.statistcs_ is used to store
            # fill_value in each column
            return np.full(X.shape[1], fill_value, dtype=X.dtype)

    def transform(self, X):
        """Impute all missing values in `X`.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data to complete.

        Returns
        -------
        X_imputed : {ndarray, sparse matrix} of shape \
                (n_samples, n_features_out)
            `X` with imputed values.
        """
        check_is_fitted(self)

        X = self._validate_input(X, in_fit=False)
        statistics = self.statistics_

        if X.shape[1] != statistics.shape[0]:
            raise ValueError(
                "X has %d features per sample, expected %d"
                % (X.shape[1], self.statistics_.shape[0])
            )

        # compute mask before eliminating invalid features
        missing_mask = _get_mask(X, self.missing_values)

        # Delete the invalid columns if strategy is not constant
        if self.strategy == "constant":
            valid_statistics = statistics
            valid_statistics_indexes = None
        else:
            # same as np.isnan but also works for object dtypes
            invalid_mask = _get_mask(statistics, np.nan)
            valid_mask = np.logical_not(invalid_mask)
            valid_statistics = statistics[valid_mask]
            valid_statistics_indexes = np.flatnonzero(valid_mask)

            if invalid_mask.any():
                missing = np.arange(X.shape[1])[invalid_mask]
                if self.verbose:
                    warnings.warn(
                        "Deleting features without observed values: %s" % missing
                    )
                X = X[:, valid_statistics_indexes]

        # Do actual imputation
        if sp.issparse(X):
            if self.missing_values == 0:
                raise ValueError(
                    "Imputation not possible when missing_values "
                    "== 0 and input is sparse. Provide a dense "
                    "array instead."
                )
            else:
                # if no invalid statistics are found, use the mask computed
                # before, else recompute mask
                if valid_statistics_indexes is None:
                    mask = missing_mask.data
                else:
                    mask = _get_mask(X.data, self.missing_values)
                indexes = np.repeat(
                    np.arange(len(X.indptr) - 1, dtype=int), np.diff(X.indptr)
                )[mask]

                X.data[mask] = valid_statistics[indexes].astype(X.dtype, copy=False)
        else:
            # use mask computed before eliminating invalid mask
            if valid_statistics_indexes is None:
                mask_valid_features = missing_mask
            else:
                mask_valid_features = missing_mask[:, valid_statistics_indexes]
            n_missing = np.sum(mask_valid_features, axis=0)
            values = np.repeat(valid_statistics, n_missing)
            coordinates = np.where(mask_valid_features.transpose())[::-1]

            X[coordinates] = values

        X_indicator = super()._transform_indicator(missing_mask)

        return super()._concatenate_indicator(X, X_indicator)

    def inverse_transform(self, X):
        """Convert the data back to the original representation.

        Inverts the `transform` operation performed on an array.
        This operation can only be performed after :class:`SimpleImputer` is
        instantiated with `add_indicator=True`.

        Note that `inverse_transform` can only invert the transform in
        features that have binary indicators for missing values. If a feature
        has no missing values at `fit` time, the feature won't have a binary
        indicator, and the imputation done at `transform` time won't be
        inverted.

        .. versionadded:: 0.24

        Parameters
        ----------
        X : array-like of shape \
                (n_samples, n_features + n_features_missing_indicator)
            The imputed data to be reverted to original data. It has to be
            an augmented array of imputed data and the missing indicator mask.

        Returns
        -------
        X_original : ndarray of shape (n_samples, n_features)
            The original `X` with missing values as it was prior
            to imputation.
        """
        check_is_fitted(self)

        if not self.add_indicator:
            raise ValueError(
                "'inverse_transform' works only when "
                "'SimpleImputer' is instantiated with "
                "'add_indicator=True'. "
                f"Got 'add_indicator={self.add_indicator}' "
                "instead."
            )

        n_features_missing = len(self.indicator_.features_)
        non_empty_feature_count = X.shape[1] - n_features_missing
        array_imputed = X[:, :non_empty_feature_count].copy()
        missing_mask = X[:, non_empty_feature_count:].astype(bool)

        n_features_original = len(self.statistics_)
        shape_original = (X.shape[0], n_features_original)
        X_original = np.zeros(shape_original)
        X_original[:, self.indicator_.features_] = missing_mask
        full_mask = X_original.astype(bool)

        imputed_idx, original_idx = 0, 0
        while imputed_idx < len(array_imputed.T):
            if not np.all(X_original[:, original_idx]):
                X_original[:, original_idx] = array_imputed.T[imputed_idx]
                imputed_idx += 1
                original_idx += 1
            else:
                original_idx += 1

        X_original[full_mask] = self.missing_values
        return X_original
    
    def get_feature_names_out(self, input_features=None):
        # uses input_features if self.feature_names_in_ is None
        if self.feature_names_in_ is None:
            return _get_input_features(input_features, self.n_features_in_)
        return self.feature_names_in_ 
    
class FicoCombiner(_BaseImputer):
    '''    
    Produces 'ficoscore' column, a weighted average of fico_range_low and fico_range_high
    '''
    def __init__(self, source_cols = ['fico_range_low', 'fico_range_high'], weights = [0.5, 0.5], target_col =  "ficoscore"):
        
        super().__init__()
        assert type(source_cols) == list, 'source_cols should be a list of columns'
        assert type(target_col) == str, 'target_col should be a string'
        self.source_cols = source_cols
        self.target_col = target_col
        self.feature_names_in_ = None
        self.weights = weights

    def fit(self, X, y=None):
        
        # assert pd.isnull(X[self.group_cols]).any(axis=None) == False, 'There are missing values in group_cols'
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns)
        else:
            self.feature_names_in_ = None
        self.n_features_in_ = X.shape[1]
        
        return self 
    
    def transform(self, X, y=None):
        
        # make sure that the imputer was fitted
        # check_is_fitted(self, 'impute_map_')
        X = X.copy()
        X[self.target_col] = np.multiply(X[self.source_cols], self.weights).sum(axis=1)
        X.drop(self.source_cols, axis = 1, inplace = True)
        # return X.values
        return X
    
    def get_feature_names_out(self, input_features=None):
        # uses input_features if self.feature_names_in_ is None
        if self.feature_names_in_ is None:
            return _get_input_features(input_features, self.n_features_in_)
        features_out = np.delete(self.feature_names_in_, np.where(self.feature_names_in_ == 'fico_range_low'))
        features_out = np.delete(features_out, np.where(self.feature_names_in_ == 'fico_range_high'))
        return np.append(features_out, 'ficoscore')

class CategoryMerger(_BaseImputer):
    '''    
    1. Redundant items are categorized (e.g. 'NONE', 'ANY', 'OTHER' --> 'OTHER')
    '''
    def __init__(self):
        
        super().__init__()

    def fit(self, X, y=None):
        
        # assert pd.isnull(X[self.group_cols]).any(axis=None) == False, 'There are missing values in group_cols'
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns)
        else:
            self.feature_names_in_ = None
        self.n_features_in_ = X.shape[1]
        
        return self 
    
    def transform(self, X, y=None):

        data = X.copy()
        # Home_ownership - non-informative values - collapse into OTHER
        data['home_ownership'].replace(['NONE', 'ANY'], 'OTHER', inplace=True)
        # return data.values
        return data

    def get_feature_names_out(self, input_features=None):
        # uses input_features if self.feature_names_in_ is None
        if self.feature_names_in_ is None:
            return _get_input_features(input_features, self.n_features_in_)
        return self.feature_names_in_

    
class StringFormatter(_BaseImputer):
    '''    
    1. Cleans inconsistent strings (e.g. emp_length: 10+ years --> 10)
    2. Redundant items are categorized (e.g. 'NONE', 'ANY', 'OTHER' --> 'OTHER')
    
    '''
    def __init__(self):
        
        super().__init__()

    def fit(self, X, y=None):
        
        # assert pd.isnull(X[self.group_cols]).any(axis=None) == False, 'There are missing values in group_cols'
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns)
        else:
            self.feature_names_in_ = None
        self.n_features_in_ = X.shape[1]
        
        return self 
    
    def transform(self, X, y=None):
        
        # make sure that the imputer was fitted
        # check_is_fitted(self, 'impute_map_')
        data = X.copy()
        data['emp_length'].replace(to_replace='10+ years', value='10 years', inplace=True)
        data['emp_length'].replace('< 1 year', '0 years', inplace=True)
        data['emp_length'] = data['emp_length'].apply(lambda s: s if pd.isnull(s) else np.int8(s.split()[0]))
        
        # Clean percentage scores
        data['revol_util'] = data['revol_util'].map(lambda x: str(x).replace('%','')).astype(np.float64)

        # temporary: leave morgage account value as total acc if morgage account is non-negative
        total_acc_avg = data.groupby('total_acc').mean()['mort_acc'].fillna(0)
        data['mort_acc'] = data.apply(lambda x: total_acc_avg[x['total_acc']] if x['mort_acc'] >=0 else x['mort_acc'], axis=1)
 
        # Utilize total length of credit history: provide interaction to account for upward mobility         
        data['last_credit_pull_d'] = pd.to_datetime(data['last_credit_pull_d'])
        data['earliest_cr_line'] = pd.to_datetime(data['earliest_cr_line'])
        data['time_since_earliest_cr_line'] = (data['last_credit_pull_d'] - data['earliest_cr_line']).dt.days
        data.drop(['earliest_cr_line', 'last_credit_pull_d'], axis=1, inplace=True)
        # return data.values
        return data
    
    def get_feature_names_out(self, input_features=None):
        # uses input_features if self.feature_names_in_ is None
        if self.feature_names_in_ is None:
        #     return _get_input_features(input_features, self.n_features_in_).append('time_since_earliest_cr_line')
        # return self.feature_names_in_.append('time_since_earliest_cr_line')
            return _get_input_features(input_features, self.n_features_in_)
        features_out = np.delete(self.feature_names_in_, np.where(self.feature_names_in_ == 'earliest_cr_line'))
        features_out = np.delete(features_out, np.where(self.feature_names_in_ == 'last_credit_pull_d'))
        return np.append(features_out, 'time_since_earliest_cr_line')
    
    
class MissingIndicator(TransformerMixin, BaseEstimator):
    """Binary indicators for missing values.

    Note that this component typically should not be used in a vanilla
    :class:`Pipeline` consisting of transformers and a classifier, but rather
    could be added using a :class:`FeatureUnion` or :class:`ColumnTransformer`.

    Read more in the :ref:`User Guide <impute>`.

    .. versionadded:: 0.20

    Parameters
    ----------
    missing_values : int, float, str, np.nan or None, default=np.nan
        The placeholder for the missing values. All occurrences of
        `missing_values` will be imputed. For pandas' dataframes with
        nullable integer dtypes with missing values, `missing_values`
        should be set to `np.nan`, since `pd.NA` will be converted to `np.nan`.

    features : {'missing-only', 'all'}, default='missing-only'
        Whether the imputer mask should represent all or a subset of
        features.

        - If `'missing-only'` (default), the imputer mask will only represent
          features containing missing values during fit time.
        - If `'all'`, the imputer mask will represent all features.

    sparse : bool or 'auto', default='auto'
        Whether the imputer mask format should be sparse or dense.

        - If `'auto'` (default), the imputer mask will be of same type as
          input.
        - If `True`, the imputer mask will be a sparse matrix.
        - If `False`, the imputer mask will be a numpy array.

    error_on_new : bool, default=True
        If `True`, :meth:`transform` will raise an error when there are
        features with missing values that have no missing values in
        :meth:`fit`. This is applicable only when `features='missing-only'`.

    Attributes
    ----------
    features_ : ndarray of shape (n_missing_features,) or (n_features,)
        The features indices which will be returned when calling
        :meth:`transform`. They are computed during :meth:`fit`. If
        `features='all'`, `features_` is equal to `range(n_features)`.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    SimpleImputer : Univariate imputation of missing values.
    IterativeImputer : Multivariate imputation of missing values.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.impute import MissingIndicator
    >>> X1 = np.array([[np.nan, 1, 3],
    ...                [4, 0, np.nan],
    ...                [8, 1, 0]])
    >>> X2 = np.array([[5, 1, np.nan],
    ...                [np.nan, 2, 3],
    ...                [2, 4, 0]])
    >>> indicator = MissingIndicator()
    >>> indicator.fit(X1)
    MissingIndicator()
    >>> X2_tr = indicator.transform(X2)
    >>> X2_tr
    array([[False,  True],
           [ True, False],
           [False, False]])
    """

    def __init__(
        self,
        *,
        missing_values=np.nan,
        features="missing-only",
        sparse="auto",
        error_on_new=True,
    ):
        self.missing_values = missing_values
        self.features = features
        self.sparse = sparse
        self.error_on_new = error_on_new

    def _get_missing_features_info(self, X):
        '''
        Compute the imputer mask and the indices of the features
        containing missing values.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            The input data with missing values. Note that `X` has been
            checked in :meth:`fit` and :meth:`transform` before to call this
            function.

        Returns
        -------
        imputer_mask : {ndarray, sparse matrix} of shape \
        (n_samples, n_features)
            The imputer mask of the original data.

        features_with_missing : ndarray of shape (n_features_with_missing)
            The features containing missing values.
        '''
        if not self._precomputed:
            imputer_mask = _get_mask(X, self.missing_values)
        else:
            imputer_mask = X

        if sp.issparse(X):
            imputer_mask.eliminate_zeros()

            if self.features == "missing-only":
                n_missing = imputer_mask.getnnz(axis=0)

            if self.sparse is False:
                imputer_mask = imputer_mask.toarray()
            elif imputer_mask.format == "csr":
                imputer_mask = imputer_mask.tocsc()
        else:
            if not self._precomputed:
                imputer_mask = _get_mask(X, self.missing_values)
            else:
                imputer_mask = X

            if self.features == "missing-only":
                n_missing = imputer_mask.sum(axis=0)

            if self.sparse is True:
                imputer_mask = sp.csc_matrix(imputer_mask)

        if self.features == "all":
            features_indices = np.arange(X.shape[1])
        else:
            features_indices = np.flatnonzero(n_missing)

        return imputer_mask, features_indices

    def _validate_input(self, X, in_fit):
        if not is_scalar_nan(self.missing_values):
            force_all_finite = True
        else:
            force_all_finite = "allow-nan"
        X = self._validate_data(
            X,
            reset=in_fit,
            accept_sparse=("csc", "csr"),
            dtype=None,
            force_all_finite=force_all_finite,
        )
        _check_inputs_dtype(X, self.missing_values)
        if X.dtype.kind not in ("i", "u", "f", "O"):
            raise ValueError(
                "MissingIndicator does not support data with "
                "dtype {0}. Please provide either a numeric array"
                " (with a floating point or integer dtype) or "
                "categorical data represented either as an array "
                "with integer dtype or an array of string values "
                "with an object dtype.".format(X.dtype)
            )

        if sp.issparse(X) and self.missing_values == 0:
            # missing_values = 0 not allowed with sparse data as it would
            # force densification
            raise ValueError(
                "Sparse input with missing_values=0 is "
                "not supported. Provide a dense "
                "array instead."
            )

        return X

    def _fit(self, X, y=None, precomputed=False):
        """Fit the transformer on `X`.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Input data, where `n_samples` is the number of samples and
            `n_features` is the number of features.
            If `precomputed=True`, then `X` is a mask of the input data.

        precomputed : bool
            Whether the input data is a mask.

        Returns
        -------
        imputer_mask : {ndarray, sparse matrix} of shape (n_samples, \
        n_features)
            The imputer mask of the original data.
        """
        if precomputed:
            if not (hasattr(X, "dtype") and X.dtype.kind == "b"):
                raise ValueError("precomputed is True but the input data is not a mask")
            self._precomputed = True
        else:
            self._precomputed = False

        # Need not validate X again as it would have already been validated
        # in the Imputer calling MissingIndicator
        if not self._precomputed:
            X = self._validate_input(X, in_fit=True)

        self._n_features = X.shape[1]

        if self.features not in ("missing-only", "all"):
            raise ValueError(
                "'features' has to be either 'missing-only' or "
                "'all'. Got {} instead.".format(self.features)
            )

        if not (
            (isinstance(self.sparse, str) and self.sparse == "auto")
            or isinstance(self.sparse, bool)
        ):
            raise ValueError(
                "'sparse' has to be a boolean or 'auto'. Got {!r} instead.".format(
                    self.sparse
                )
            )

        missing_features_info = self._get_missing_features_info(X)
        self.features_ = missing_features_info[1]

        return missing_features_info[0]

    def fit(self, X, y=None):
        """Fit the transformer on `X`.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Input data, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self._fit(X, y)

        return self

    def transform(self, X):
        """Generate missing values indicator for `X`.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data to complete.

        Returns
        -------
        Xt : {ndarray, sparse matrix} of shape (n_samples, n_features) \
        or (n_samples, n_features_with_missing)
            The missing indicator for input data. The data type of `Xt`
            will be boolean.
        """
        check_is_fitted(self)

        # Need not validate X again as it would have already been validated
        # in the Imputer calling MissingIndicator
        if not self._precomputed:
            X = self._validate_input(X, in_fit=False)
        else:
            if not (hasattr(X, "dtype") and X.dtype.kind == "b"):
                raise ValueError("precomputed is True but the input data is not a mask")

        imputer_mask, features = self._get_missing_features_info(X)

        if self.features == "missing-only":
            features_diff_fit_trans = np.setdiff1d(features, self.features_)
            if self.error_on_new and features_diff_fit_trans.size > 0:
                raise ValueError(
                    "The features {} have missing values "
                    "in transform but have no missing values "
                    "in fit.".format(features_diff_fit_trans)
                )

            if self.features_.size < self._n_features:
                imputer_mask = imputer_mask[:, self.features_]

        return imputer_mask

    def fit_transform(self, X, y=None):
        """Generate missing values indicator for `X`.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data to complete.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        Xt : {ndarray, sparse matrix} of shape (n_samples, n_features) \
        or (n_samples, n_features_with_missing)
            The missing indicator for input data. The data type of `Xt`
            will be boolean.
        """
        imputer_mask = self._fit(X, y)

        if self.features_.size < self._n_features:
            imputer_mask = imputer_mask[:, self.features_]

        return imputer_mask

    def _more_tags(self):
        return {
            "allow_nan": True,
            "X_types": ["2darray", "string"],
            "preserves_dtype": [],
        }

    
from category_encoders.ordinal import OrdinalEncoder
import category_encoders.utils as util
from sklearn.utils.random import check_random_state

class WOEEncoder(util.BaseEncoder, util.SupervisedTransformerMixin):
    """Weight of Evidence coding for categorical features.
    Supported targets: binomial. For polynomial target support, see PolynomialWrapper.
    Parameters
    ----------
    verbose: int
        integer indicating verbosity of the output. 0 for none.
    cols: list
        a list of columns to encode, if None, all string columns will be encoded.
    drop_invariant: bool
        boolean for whether or not to drop columns with 0 variance.
    return_df: bool
        boolean for whether to return a pandas DataFrame from transform (otherwise it will be a numpy array).
    handle_missing: str
        options are 'return_nan', 'error' and 'value', defaults to 'value', which will assume WOE=0.
    handle_unknown: str
        options are 'return_nan', 'error' and 'value', defaults to 'value', which will assume WOE=0.
    randomized: bool,
        adds normal (Gaussian) distribution noise into training data in order to decrease overfitting (testing data are untouched).
    sigma: float
        standard deviation (spread or "width") of the normal distribution.
    regularization: float
        the purpose of regularization is mostly to prevent division by zero.
        When regularization is 0, you may encounter division by zero.

    """
    prefit_ordinal = True
    encoding_relation = util.EncodingRelation.ONE_TO_ONE

    def __init__(self, verbose=0, cols=None, drop_invariant=False, return_df=True,
                 handle_unknown='value', handle_missing='value', random_state=None, randomized=False, sigma=0.05, regularization=1.0):
        super().__init__(verbose=verbose, cols=cols, drop_invariant=drop_invariant, return_df=return_df,
                         handle_unknown=handle_unknown, handle_missing=handle_missing)
        self.ordinal_encoder = None
        self._sum = None
        self._count = None
        self.random_state = random_state
        self.randomized = randomized
        self.sigma = sigma
        self.regularization = regularization
        self.feature_names_in_ = None

    def _fit(self, X, y, **kwargs):
        # The label must be binary with values {0,1}
        unique = y.unique()
        if len(unique) != 2:
            raise ValueError("The target column y must be binary. But the target contains " + str(len(unique)) + " unique value(s).")
        if y.isnull().any():
            raise ValueError("The target column y must not contain missing values.")
        if np.max(unique) < 1:
            raise ValueError("The target column y must be binary with values {0, 1}. Value 1 was not found in the target.")
        if np.min(unique) > 0:
            raise ValueError("The target column y must be binary with values {0, 1}. Value 0 was not found in the target.")

        self.ordinal_encoder = OrdinalEncoder(
            verbose=self.verbose,
            cols=self.cols,
            handle_unknown='value',
            handle_missing='value'
        )
        self.ordinal_encoder = self.ordinal_encoder.fit(X)
        X_ordinal = self.ordinal_encoder.transform(X)

        # Training
        self.mapping = self._train(X_ordinal, y)
        
        self.n_features_in_ = X.shape[1]

    def _transform(self, X, y=None):
        X = self.ordinal_encoder.transform(X)

        if self.handle_unknown == 'error':
            if X[self.cols].isin([-1]).any().any():
                raise ValueError('Unexpected categories found in dataframe')

        # Loop over columns and replace nominal values with WOE
        X = self._score(X, y)
        return X

    def _train(self, X, y):
        # Initialize the output
        mapping = {}

        # Calculate global statistics
        self._sum = y.sum()
        self._count = y.count()

        for switch in self.ordinal_encoder.category_mapping:
            col = switch.get('col')
            values = switch.get('mapping')
            # Calculate sum and count of the target for each unique value in the feature col
            stats = y.groupby(X[col]).agg(['sum', 'count'])  # Count of x_{i,+} and x_i

            # Create a new column with regularized WOE.
            # Regularization helps to avoid division by zero.
            # Pre-calculate WOEs because logarithms are slow.
            nominator = (stats['sum'] + self.regularization) / (self._sum + 2*self.regularization)
            denominator = ((stats['count'] - stats['sum']) + self.regularization) / (self._count - self._sum + 2*self.regularization)
            woe = np.log(nominator / denominator)

            # Ignore unique values. This helps to prevent overfitting on id-like columns.
            woe[stats['count'] == 1] = 0

            if self.handle_unknown == 'return_nan':
                woe.loc[-1] = np.nan
            elif self.handle_unknown == 'value':
                woe.loc[-1] = 0

            if self.handle_missing == 'return_nan':
                woe.loc[values.loc[np.nan]] = np.nan
            elif self.handle_missing == 'value':
                woe.loc[-2] = 0

            # Store WOE for transform() function
            mapping[col] = woe

        return mapping

    def _score(self, X, y):
        for col in self.cols:
            # Score the column
            X[col] = X[col].map(self.mapping[col])

            # Randomization is meaningful only for training data -> we do it only if y is present
            if self.randomized and y is not None:
                random_state_generator = check_random_state(self.random_state)
                X[col] = (X[col] * random_state_generator.normal(1., self.sigma, X[col].shape[0]))

        return X
    
    def get_feature_names_out(self, input_features=None):
        # uses input_features if self.feature_names_in_ is None
        if self.feature_names_in_ is None:
            return _get_input_features(input_features, self.n_features_in_)
        return self.feature_names_in_


class _BaseEncoder(TransformerMixin, BaseEstimator):
    """
    Base class for encoders that includes the code to categorize and
    transform the input features.
    """

    def _check_X(self, X, force_all_finite=True):
        """
        Perform custom check_array:
        - convert list of strings to object dtype
        - check for missing values for object dtype data (check_array does
          not do that)
        - return list of features (arrays): this list of features is
          constructed feature by feature to preserve the data types
          of pandas DataFrame columns, as otherwise information is lost
          and cannot be used, e.g. for the `categories_` attribute.
        """
        if not (hasattr(X, "iloc") and getattr(X, "ndim", 0) == 2):
            # if not a dataframe, do normal check_array validation
            X_temp = check_array(X, dtype=None, force_all_finite=force_all_finite)
            if not hasattr(X, "dtype") and np.issubdtype(X_temp.dtype, np.str_):
                X = check_array(X, dtype=object, force_all_finite=force_all_finite)
            else:
                X = X_temp
            needs_validation = False
        else:
            # pandas dataframe, do validation later column by column, in order
            # to keep the dtype information to be used in the encoder.
            needs_validation = force_all_finite

        n_samples, n_features = X.shape
        X_columns = []

        for i in range(n_features):
            Xi = self._get_feature(X, feature_idx=i)
            Xi = check_array(
                Xi, ensure_2d=False, dtype=None, force_all_finite=needs_validation
            )
            X_columns.append(Xi)

        return X_columns, n_samples, n_features

    def _get_feature(self, X, feature_idx):
        if hasattr(X, "iloc"):
            # pandas dataframes
            return X.iloc[:, feature_idx]
        # numpy arrays, sparse arrays
        return X[:, feature_idx]

    def _fit(
        self, X, handle_unknown="error", force_all_finite=True, return_counts=False
    ):
        self._check_n_features(X, reset=True)
        self._check_feature_names(X, reset=True)
        X_list, n_samples, n_features = self._check_X(
            X, force_all_finite=force_all_finite
        )
        self.n_features_in_ = n_features

        if self.categories != "auto":
            if len(self.categories) != n_features:
                raise ValueError(
                    "Shape mismatch: if categories is an array,"
                    " it has to be of shape (n_features,)."
                )

        self.categories_ = []
        category_counts = []

        for i in range(n_features):
            Xi = X_list[i]

            if self.categories == "auto":
                result = _unique(Xi, return_counts=return_counts)
                if return_counts:
                    cats, counts = result
                    category_counts.append(counts)
                else:
                    cats = result
            else:
                cats = np.array(self.categories[i], dtype=Xi.dtype)
                if Xi.dtype.kind not in "OUS":
                    sorted_cats = np.sort(cats)
                    error_msg = (
                        "Unsorted categories are not supported for numerical categories"
                    )
                    # if there are nans, nan should be the last element
                    stop_idx = -1 if np.isnan(sorted_cats[-1]) else None
                    if np.any(sorted_cats[:stop_idx] != cats[:stop_idx]) or (
                        np.isnan(sorted_cats[-1]) and not np.isnan(sorted_cats[-1])
                    ):
                        raise ValueError(error_msg)

                if handle_unknown == "error":
                    diff = _check_unknown(Xi, cats)
                    if diff:
                        msg = (
                            "Found unknown categories {0} in column {1}"
                            " during fit".format(diff, i)
                        )
                        raise ValueError(msg)
                if return_counts:
                    category_counts.append(_get_counts(Xi, cats))

            self.categories_.append(cats)

        output = {"n_samples": n_samples}
        if return_counts:
            output["category_counts"] = category_counts
        return output

    def _transform(
        self, X, handle_unknown="error", force_all_finite=True, warn_on_unknown=False
    ):
        self._check_feature_names(X, reset=False)
        self._check_n_features(X, reset=False)
        X_list, n_samples, n_features = self._check_X(
            X, force_all_finite=force_all_finite
        )

        X_int = np.zeros((n_samples, n_features), dtype=int)
        X_mask = np.ones((n_samples, n_features), dtype=bool)

        columns_with_unknown = []
        for i in range(n_features):
            Xi = X_list[i]
            diff, valid_mask = _check_unknown(Xi, self.categories_[i], return_mask=True)

            if not np.all(valid_mask):
                if handle_unknown == "error":
                    msg = (
                        "Found unknown categories {0} in column {1}"
                        " during transform".format(diff, i)
                    )
                    raise ValueError(msg)
                else:
                    if warn_on_unknown:
                        columns_with_unknown.append(i)
                    # Set the problematic rows to an acceptable value and
                    # continue `The rows are marked `X_mask` and will be
                    # removed later.
                    X_mask[:, i] = valid_mask
                    # cast Xi into the largest string type necessary
                    # to handle different lengths of numpy strings
                    if (
                        self.categories_[i].dtype.kind in ("U", "S")
                        and self.categories_[i].itemsize > Xi.itemsize
                    ):
                        Xi = Xi.astype(self.categories_[i].dtype)
                    elif self.categories_[i].dtype.kind == "O" and Xi.dtype.kind == "U":
                        # categories are objects and Xi are numpy strings.
                        # Cast Xi to an object dtype to prevent truncation
                        # when setting invalid values.
                        Xi = Xi.astype("O")
                    else:
                        Xi = Xi.copy()

                    Xi[~valid_mask] = self.categories_[i][0]
            # We use check_unknown=False, since _check_unknown was
            # already called above.
            X_int[:, i] = _encode(Xi, uniques=self.categories_[i], check_unknown=False)
        if columns_with_unknown:
            warnings.warn(
                "Found unknown categories in columns "
                f"{columns_with_unknown} during transform. These "
                "unknown categories will be encoded as all zeros",
                UserWarning,
            )

        return X_int, X_mask

    def _more_tags(self):
        return {"X_types": ["categorical"]}


    

class SeasonalityExtractor(_BaseEncoder):
    '''
    Class used for imputing missing values in a pd.DataFrame using either mean or median of a group.
    
    Parameters
    ----------    
    group_cols : list
        List of columns used for calculating the aggregated value 
    target : str
        The name of the column to impute
    metric : str
        The metric to be used for remplacement, can be one of ['mean', 'median']

    Returns
    -------
    X : array-like
        The array with imputed values in the target column
    '''
    def __init__(self, expand_date_column = "application_date", expand_sin_cos = True, expand_days = True, time_features = ["hour", "day", "week", "month"], day_categories = ["Holiday", "Weekend"], special_day_fall_in_range = 3):
        
        super().__init__()
        
        assert isinstance(expand_date_column, str), 'source_cols should be a string of the datetime feature names'
        self.expand_date_column = expand_date_column
        # self.df = df ### ILLEGAL
        # self.datetime = pd.to_datetime(self.df[self.expand_date_column])
        
        self.allowed_time_features = ["minute", "hour", "day", "week", "month"]
        self.day_categories = ["Holiday", "Weekend", "Birthday", "TaxSeason"]
        # 'isFederalHoliday', 'isTaxSeason', 'isWeekend
        
        # self.datetime_object = {'minute': self.datetime.dt.minute, 
        #                         'hour': self.datetime.dt.hour, 
        #                         'day': self.datetime.dt.day - 1, 
        #                         'week': self.datetime.dt.weekday, 
        #                         'month': self.datetime.dt.month - 1} 

        # self.period = {'minute': 60, 'hour': 24, 'day': 30, 'week': 7, 'month': 12}  
        
        self.expand_sin_cos = expand_sin_cos
        self.expand_days = expand_days
        
        self.time_features = time_features
        assert isinstance(self.time_features, str) or isinstance(self.time_features, list), "Attribute time_features must be a string or a list"
        
        if isinstance(self.time_features, str):
            assert self.time_features in self.allowed_time_features, "Attribute time_features must contain any of the following: 'minute', 'hour', 'day', 'week', 'month'"
            self.time_features = [self.time_features]
         
        self.day_categories = day_categories
        assert isinstance(self.day_categories, str) or isinstance(self.day_categories, list), "Attribute day_categories must be a string or a list of the following: Holiday, Weekend, Birthday, TaxSeason"
        
        if isinstance(self.day_categories, str):
            assert self.day_categories in self.allowed_day_categories, "Attribute day_categories must contain any of the following: Holiday, Weekend, etc."
            self.day_categories = [self.day_categories]
            
        self.special_day_fall_in_range = special_day_fall_in_range
        
        self.feature_names_out = None

    
    def fit(self, X, y=None):
        
        # assert pd.isnull(X[self.group_cols]).any(axis=None) == False, 'There are missing values in group_cols'
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns)
        else:
            self.feature_names_in_ = None
        self.n_features_in_ = X.shape[1]
        
        return self 
    
    
    def transform(self, X, y=None):
        
        X = X.copy()
        
        self.datetime = pd.to_datetime(X[self.expand_date_column])
        self.datetime_object = {
            'minute': self.datetime.dt.minute, 
            'hour': self.datetime.dt.hour, 
            'day': self.datetime.dt.day - 1, 
            'week': self.datetime.dt.weekday, 
            'month': self.datetime.dt.month - 1
        } 
        self.period = {'minute': 60, 'hour': 24, 'day': 30, 'week': 7, 'month': 12}  
        
        if self.expand_sin_cos:
            
            for feature in self.time_features:
                X[feature + "_cos"] = np.cos(self.datetime_object[feature]*(2.*np.pi / self.period[feature]))
                # self.feature_names_out.append(feature + "_cos")
                X[feature + "_sin"] = np.sin(self.datetime_object[feature]*(2.*np.pi / self.period[feature]))
                # self.feature_names_out.append(feature + "_sin")

        if self.expand_days:
            for day_category in self.day_categories:
                if day_category == 'Weekend':
                    X["is_" + day_category] = self.datetime.apply(lambda datetime: int(datetime.weekday() in [5,6]))
                if day_category == 'Holiday':
                    X["is_" + day_category] = self.datetime.apply(lambda datetime: int(datetime in holidays.US()))
                # self.feature_names_out.append("is_" + day_category)

        X.drop(self.expand_date_column, axis = 1, inplace = True)
        
        self.feature_names_out = X.columns

        return X        
        # return X.values
        # return self.df
    
    def get_feature_names_out(self, input_features=None):
        # uses input_features if self.feature_names_in_ is None
        # if self.feature_names_in_ is None:
        #     return _get_input_features(input_features, self.n_features_in_)
        # return self.feature_names_in_
        return self.feature_names_out
    
    
from sklearn.pipeline import TransformerMixin
from imblearn.pipeline import Pipeline
from sklearn.neighbors import LocalOutlierFactor

class OutlierExtractor(TransformerMixin):
    def __init__(self, **kwargs):
        """
        Create a transformer to remove outliers. A threshold is set for selection
        criteria, and further arguments are passed to the LocalOutlierFactor class

        Keyword Args:
            neg_conf_val (float): The threshold for excluding samples with a lower
               negative outlier factor.

        Returns:
            object: to be used as a transformer method as part of Pipeline()
        """

        self.threshold = kwargs.pop('neg_conf_val', -10.0)

        self.kwargs = kwargs

    def transform(self, X, y):
        """
        Uses LocalOutlierFactor class to subselect data based on some threshold

        Returns:
            ndarray: subsampled data

        Notes:
            X should be of shape (n_samples, n_features)
        """
        X = np.asarray(X)
        y = np.asarray(y)
        lcf = LocalOutlierFactor(**self.kwargs)
        lcf.fit(X)
        return (X[lcf.negative_outlier_factor_ > self.threshold, :],
                y[lcf.negative_outlier_factor_ > self.threshold])

    def fit(self, *args, **kwargs):
        return self

    
if __name__ == "__main__":
    
    pass