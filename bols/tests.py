import numpy as np
import unittest
import statsmodels.api as sm

from bols import BOLS
from sklearn.linear_model import LinearRegression


data_y = np.random.random_sample((10000,))
data_x0 = np.random.random_sample((10000,))
data_x1 = np.random.random_sample((10000,))
data = np.column_stack((data_y, data_x0, data_x1))

y_a = data[0:5000, 0]
y_b = data[5000:10000, 0]
data_a = data[0:5000, 1:]
data_b = data[5000:10000, 1:]
model_a = BOLS()
model_a.batch([y_a, y_b], [data_a, data_b])
model_a.get_statistics()


class TestBOLS(unittest.TestCase):

    def test_batches(self):
        model_b = BOLS()
        model_b.batch([data[:, 0]], [data[:, 1:]])
        model_b.get_statistics()
        self.assertTrue(np.isclose(model_a.R2, model_b.R2))
        self.assertTrue(np.isclose(model_a.beta, model_b.beta).all())
        self.assertTrue(np.isclose(model_a.F, model_b.F))
        self.assertTrue(np.isclose(model_a.F_p_value, model_b.F_p_value))
        self.assertTrue(np.isclose(model_a.beta, model_b.beta).all())
        self.assertTrue(np.isclose(model_a.se_beta, model_b.se_beta).all())
        self.assertTrue(np.isclose(model_a.beta_p_value,
                        model_b.beta_p_value).all())
        self.assertTrue(model_a.df == model_b.df)
        self.assertTrue(model_a.df_resid == model_b.df_resid)
        self.assertTrue(np.isclose(model_a.residual_sum_of_squares,
                        model_b.residual_sum_of_squares))

    def test_sk_regression(self):
        sk_ols = LinearRegression(fit_intercept=False)
        sk_ols.fit(y=data[:, 0], X=data[:, 1:])
        assert np.isclose(model_a.beta, sk_ols.coef_).all()

    def test_sm_model(self):
        sm_ols = sm.OLS(data[:, 0], data[:, 1:])
        sm_ols = sm_ols.fit()
        self.assertTrue(np.isclose(sm_ols.rsquared, model_a.R2))
        self.assertTrue(np.isclose(sm_ols.params, model_a.beta).all())
        self.assertTrue(np.isclose(sm_ols.fvalue, model_a.F))
        self.assertTrue(np.isclose(sm_ols.f_pvalue, model_a.F_p_value))
        self.assertTrue(np.isclose(sm_ols.tvalues,
                        model_a.beta / model_a.se_beta).all())
        self.assertTrue(np.isclose(sm_ols.pvalues, model_a.beta_p_value).all())
        self.assertTrue(sm_ols.df_model == model_a.df)
        self.assertTrue(sm_ols.df_resid == model_a.df_resid)
        self.assertTrue(np.isclose(sm_ols.ssr,
                        model_a.residual_sum_of_squares))


if __name__ == '__main__':
    unittest.main()
