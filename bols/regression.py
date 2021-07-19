import functools
import math
import numpy as np
import scipy
import warnings

from collections import namedtuple
from tabulate import tabulate


class BOLS(object):
    '''
    A batch OLS regression that allows you to iterate over your training data.
    Is considerably more memory effecient than a naive implementation of OLS.
    Expects a vector of your dependent variable y as well as a column-ordered
    design matrix with your independent variables X.
    X needs to have the same shape for each iteration/update. Does not
    calculate intercepts, i.e. data has to be already centered  or you have to
    add a dummy column to your data. Naturally supports multi-processing as the
    heavy lifting is done with numpy.

    Inspired by the answer of Chris Taylor:
    http://stackoverflow.com/questions/13148052/linear-regression-in-numpy-with-very-large-matrices-how-to-save-memory

    The statistical parts taken from here:
    http://dept.stat.lsa.umich.edu/~kshedden/Courses/Stat401/Notes/401-multreg.pdf
    https://web.archive.org/web/20161014123710/http://dept.stat.lsa.umich.edu/~kshedden/Courses/Stat401/Notes/401-multreg.pdf
    t-test of coefficients:
    http://reliawiki.org/index.php/Simple_Linear_Regression_Analysis
    F test:
    http://facweb.cs.depaul.edu/sjost/csc423/documents/f-test-reg.htm

    And for the test example for calculating the parameter variance see here:
    http://stats.stackexchange.com/questions/27916/standard-errors-for-multiple-regression-coefficients

    '''
    def __init__(self, shape=None, names=None):
        '''
        shape: An optional "numpy" tuple of the shape of each matrix X. If not
        provided, it will be inferred from the first X
        names: A list of optional variable names
        '''
        self.nobs = 0
        self.errors = list()
        self.df_resid = 0
        if shape is not None:
            self.shape = shape
            self.df = shape[1]
        else:
            self.shape = None
        self.XtX = None
        self.Xty = None
        if names is not None:
            self.names = names
        else:
            self.names = None
        self.total_sum_of_squares = 0

    def __calculate_OLS_intermediate__(self, y, X):
        '''
        Calculates X'X and X'y.
        y: Dependent variable
        X: Independent variables
        '''
        XtX = np.dot(X.transpose(), X)
        Xty = np.dot(X.transpose(), y)
        return XtX, Xty

    def batch(self, ys, Xs, calculate_errors=True):
        '''
        ys: An iterable that yields your independent variables y
        Xs: An iterable that yields your design matrices X
        calculate_errors: Wehther to calculate the residulas or not
        '''
        for y, X in zip(ys, Xs):
            self.update(y, X)
        self.fit()
        if calculate_errors:
            for y, X in zip(ys, Xs):
                self.calculate_errors(y, X)

    def get_statistics(self, verbose=False):
        '''
        Returns and optionally prints the test statistics of the regression
        verbose: Whether to print the test statistics.
        '''
        self.__calculate_statistics()
        self.__test_parameters()
        if self.names is None:
            names = list(range(len(self.beta_p_value)))

        rows = list(zip(names, self.beta, self.se_beta,
                    self.standardized_parameters, self.beta_p_value))
        if verbose:
            print('OLS Regression Results')
            print('')
            print('F: {:>12}'.format(round(self.F, 3)))
            print('P>|F|: {:>8}'.format(round(self.F_p_value, 3)))
            print('')
            print(tabulate(rows,
                  headers=['Variable', 'Coef.',
                           'Standard Error', 't', 'P>|t|'],
                  floatfmt=(".3f", ".3f", ".3f", ".3f", ".3f")))

        result = namedtuple('result', ['names', 'F', 'F_p_value', 'R2', 'RMSE',
                            'beta', 'se_beta', 'beta_p_value'])
        return result(names, self.F, self.F_p_value, self.R2, self.RMSE,
                      self.beta, self.se_beta, self.beta_p_value)

    def update(self, y, X):
        '''
        Updates X'X and X'y
        y: Dependent variable
        X: Independent variables
        '''
        if self.shape is None:
            self.shape = X.shape
            self.df = X.shape[1]
        assert X.shape == self.shape, (
                'X.shape %s. does not match self.shape %s'
                % (X.shape, self.shape))
        self.nobs += len(X)
        self.df_resid = self.nobs - self.df
        XtX_i, Xty_i = self.__calculate_OLS_intermediate__(y, X)
        if self.XtX is None and self.Xty is None:
            self.XtX = XtX_i
            self.Xty = Xty_i
        else:
            self.XtX = np.add(self.XtX, XtX_i)
            self.Xty = np.add(self.Xty, Xty_i)
        self.total_sum_of_squares += np.dot(y.transpose(), y)

    def fit(self):
        '''
        Calculates the parameter vector beta
        '''
        # We are solving: (X'X) * beta = X'Y
        self.beta = np.linalg.solve(self.XtX, self.Xty)

    def calculate_errors(self, y, X):
        '''
        Calculates the errors/residuals
        y: Dependent variable
        X: Independent variables
        '''
        e = y - np.dot(X, self.beta)
        self.errors.extend(e)

    def __calculate_statistics(self):
        '''
        Calculates the R^2, RMSE, and the standard errors of the parameter
        estimates of beta.
        '''
        if self.nobs != len(self.errors):
            warnings.warn(
                'Number of errors does not match number of observations.')
        self.residual_sum_of_squares = np.dot(
                np.asarray(self.errors).transpose(), np.asarray(self.errors))
        residual_sum_of_squares = self.residual_sum_of_squares
        error_variance = residual_sum_of_squares/(self.nobs - self.df)
        self.R2 = 1 - residual_sum_of_squares/self.total_sum_of_squares
        self.RMSE = math.sqrt(residual_sum_of_squares/self.nobs)
        # Calculating the inverse might not be the best numerically
        # but it works for now
        self.se_beta = np.sqrt(np.multiply(error_variance,
                               np.diagonal(np.linalg.inv(self.XtX))))
        SSM = self.total_sum_of_squares - self.residual_sum_of_squares
        self.F = (SSM/self.df) / (self.residual_sum_of_squares/self.df_resid)

    def __test_parameters(self):
        '''
        Test the F value and using the null hypothesis that F = 0.
        Test for each predictor variable X_i the null hypothesis beta_i = 0
        against the alternative hypothesis beta_i != 0
        '''
        self.F_p_value = scipy.stats.f.sf(self.F, self.df, self.df_resid)
        self.standardized_parameters = self.beta/self.se_beta
        t_distribution = functools.partial(scipy.stats.t.sf, df=self.nobs - 2)
        self.beta_p_value = np.apply_along_axis(
            t_distribution, 0, np.absolute(self.standardized_parameters)) * 2

    def predict(self, X):
        return np.dot(X, self.beta)
