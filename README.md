# Batch Ordinary Least Squares regression

An OLS regression that allows you to iterate over your training data in 
batches. Useful when a normal implementation of linear regression does not fit 
into memory as this library is considerably more memory efficient than the 
standard implementation.
Expects a vector of your dependent variable y as well as a column-ordered 
design matrix with your independent variables X.
X needs to have the same shape for each iteration/update. Does not calculate
intercepts, i.e. data has to be already centered  or you have to add a dummy
column to your data. Naturally supports multi-processing as the heavy 
lifting is done with numpy. 
Inspired by the answer of Chris Taylor on [Stackoverlfow](http://stackoverflow.com/questions/13148052/linear-regression-in-numpy-with-very-large-matrices-how-to-save-memory).


## Installation
The library can be installed straight from [PyPI](https://pypi.org/).

```
pip install bols
``` 

The only dependencies are `numpy` and `scipy` and the library should work 
with all Python versions >= 3.6.

## Usage 

First generate some data.

```python
>>> import numpy as np

>>> data_y = np.random.random_sample((15000,))
>>> data_x0 = np.random.random_sample((15000,))
>>> data_x1 = np.random.random_sample((15000,))
>>> data = np.column_stack((data_y, data_x0, data_x1))
>>> y_a = data[0:5000, 0]
>>> y_b = data[5000:10000, 0]
>>> y_c = data[10000:15000, 0]
>>> data_a = data[0:5000, 1:]
>>> data_b = data[5000:10000, 1:]
>>> data_c = data[10000:15000, 1:]
```

Then you can just fit a model. You need to pass an iterable of both your 
dependent and independent variables or in other words an iterable over your
batches. The only limitation is that batches need to be of the same size.

```python 
>>> from bols import BOLS
>>> model = BOLS()
>>> model.batch([y_a, y_b], [data_a, data_b]) 
``` 

We can then also use the fitted model to predict unseen data.

```python
>>> model.predict(data_c)
array([0.27206   , 0.42766053, 0.63881539, ..., 0.39375078, 0.44824941,
       0.4866372 ])
```

Alternatively, we can also update our model with new batches in the future.

```python
>>> model.batch([y_c], [data_c])
```

We can also get a bunch of useful statistics about the regression with 
`model.get_statistics(verbose=True)` where `verbose` determines whether the 
method just returns the statistics or prints them as well.

```python
>>> model.get_statistics(verbose=True)
OLS Regression Results

F:    13564.635
P>|F|:      0.0

  Variable    Coef.    Standard Error       t    P>|t|
----------  -------  ----------------  ------  -------
         0    0.429             0.007  58.225    0.000
         1    0.433             0.007  58.926    0.000

result(names=[0, 1], F=13564.634855542236, F_p_value=0.0, R2=0.6439835739923647, RMSE=0.3463917044171118, beta=array([0.42915076, 0.43304183]), se_beta=array([0.00737053, 0.00734891]), beta_p_value=array([0., 0.]))
```
Even though our data is purely random the regression and the coefficients are 
both statistically significant.
It is up to the user to make sure linear regression is an appropriate model
for the data by for example examining the residuals (`model.errors`). 

## Tests

The package is tested against both the implementations of linear regressions by 
`sklearn` and `statsmodels`. Those two packages thus become additional 
dependencies for running the tests.

## Development

Using `nix-shell default.nix` drops you in a development shell with all 
dependencies already installed.
