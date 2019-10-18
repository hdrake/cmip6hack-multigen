import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.tests import assert_allclose

from xskillscore.core.deterministic import (
    _preprocess_dims, _preprocess_weights, mae, mse, pearson_r, pearson_r_p_value, rmse)
from xskillscore.core.np_deterministic import (
    _mae, _mse, _pearson_r, _pearson_r_p_value, _rmse)


AXES = ('time', 'lat', 'lon', ('lat', 'lon'), ('time', 'lat', 'lon'))

@pytest.fixture
def a():
    dates = pd.date_range('1/1/2000', '1/3/2000', freq='D')
    lats = np.arange(4)
    lons = np.arange(5)
    data = np.random.rand(len(dates), len(lats), len(lons))
    return xr.DataArray(data,
                        coords=[dates, lats, lons],
                        dims=['time', 'lat', 'lon'])


@pytest.fixture
def b():
    dates = pd.date_range('1/1/2000', '1/3/2000', freq='D')
    lats = np.arange(4)
    lons = np.arange(5)
    data = np.random.rand(len(dates), len(lats), len(lons))
    return xr.DataArray(data,
                        coords=[dates, lats, lons],
                        dims=['time', 'lat', 'lon'])


@pytest.fixture
def weights_ones():
    """
    Weighting array of all ones, i.e. no weighting.
    """
    dates = pd.date_range('1/1/2000', '1/3/2000', freq='D')
    lats = np.arange(4)
    lons = np.arange(5)
    data = np.ones((len(dates), len(lats), len(lons)))
    return xr.DataArray(data,
                        coords=[dates, lats, lons],
                        dims=['time', 'lat', 'lon'])


@pytest.fixture
def weights_latitude():
    """
    Weighting array by cosine of the latitude.
    """
    dates = pd.date_range('1/1/2000', '1/3/2000', freq='D')
    lats = np.arange(4)
    lons = np.arange(5)
    cos = np.abs(np.cos(lats))
    data = np.tile(cos, (len(dates), len(lons), 1)).reshape(len(dates), len(lats), len(lons))
    return xr.DataArray(data,
                        coords=[dates, lats, lons],
                        dims=['time', 'lat', 'lon'])


@pytest.fixture
def a_dask():
    dates = pd.date_range('1/1/2000', '1/3/2000', freq='D')
    lats = np.arange(4)
    lons = np.arange(5)
    data = np.random.rand(len(dates), len(lats), len(lons))
    return xr.DataArray(data,
                        coords=[dates, lats, lons],
                        dims=['time', 'lat', 'lon']).chunk()


@pytest.fixture
def b_dask(b):
    dates = pd.date_range('1/1/2000', '1/3/2000', freq='D')
    lats = np.arange(4)
    lons = np.arange(5)
    data = np.random.rand(len(dates), len(lats), len(lons))
    return xr.DataArray(data,
                        coords=[dates, lats, lons],
                        dims=['time', 'lat', 'lon']).chunk()


@pytest.fixture
def weights_ones_dask(b):
    """
    Weighting array of all ones, i.e. no weighting.
    """
    dates = pd.date_range('1/1/2000', '1/3/2000', freq='D')
    lats = np.arange(4)
    lons = np.arange(5)
    data = np.ones((len(dates), len(lats), len(lons)))
    return xr.DataArray(data,
                        coords=[dates, lats, lons],
                        dims=['time', 'lat', 'lon']).chunk()


@pytest.fixture
def weights_latitude_dask():
    """
    Weighting array by cosine of the latitude.
    """
    dates = pd.date_range('1/1/2000', '1/3/2000', freq='D')
    lats = np.arange(4)
    lons = np.arange(5)
    cos = np.abs(np.cos(lats))
    data = np.tile(cos, (len(dates), len(lons), 1)).reshape(len(dates), len(lats), len(lons))
    return xr.DataArray(data,
                        coords=[dates, lats, lons],
                        dims=['time', 'lat', 'lon']).chunk()


def adjust_weights(weight, dim, weights_ones, weights_latitude):
    """
    Adjust the weights test data to only span the core dimension
    that the function is being applied over.
    """
    drop_dims = [i for i in weights_ones.dims if i not in dim]
    drop_dims = {k: 0 for k in drop_dims}
    if weight:
        weights_arg = weights_latitude.isel(drop_dims)
        weights_np = weights_latitude.isel(drop_dims)
    else:
        weights_arg = None
        weights_np = weights_ones.isel(drop_dims)
    return weights_arg, weights_np


@pytest.mark.parametrize('dim', AXES)
@pytest.mark.parametrize('weight', [True, False])
def test_pearson_r_xr(a, b, dim, weight, weights_ones, weights_latitude):
    # Generates subsetted weights to pass in as arg to main function and for the numpy testing.
    weights_arg, weights_np = adjust_weights(weight, dim, weights_ones, weights_latitude)

    actual = pearson_r(a, b, dim, weights=weights_arg)
    assert actual.chunks is None

    dim, _ = _preprocess_dims(dim)
    if len(dim) > 1:
        new_dim = '_'.join(dim)
        _a = a.stack(**{new_dim: dim})
        _b = b.stack(**{new_dim: dim})
        _weights_np = weights_np.stack(**{new_dim: dim})
    else:
        new_dim = dim[0]
        _a = a
        _b = b
        _weights_np = weights_np
    _weights_np = _preprocess_weights(_a, dim, new_dim, _weights_np)

    axis = _a.dims.index(new_dim)
    res = _pearson_r(_a.values, _b.values, _weights_np.values, axis)
    expected = actual.copy()
    expected.values = res
    assert_allclose(actual, expected)


@pytest.mark.parametrize('dim', AXES)
@pytest.mark.parametrize('weight', [True, False])
def test_pearson_r_xr_dask(a_dask, b_dask, dim, weight, weights_ones_dask, weights_latitude_dask):
    # Generates subsetted weights to pass in as arg to main function and for the numpy testing.
    weights_arg, weights_np = adjust_weights(weight, dim, weights_ones_dask, weights_latitude_dask)

    actual = pearson_r(a_dask, b_dask, dim, weights=weights_arg)
    assert actual.chunks is not None

    dim, _ = _preprocess_dims(dim)
    if len(dim) > 1:
        new_dim = '_'.join(dim)
        _a_dask = a_dask.stack(**{new_dim: dim})
        _b_dask = b_dask.stack(**{new_dim: dim})
        _weights_np = weights_np.stack(**{new_dim: dim})
    else:
        new_dim = dim[0]
        _a_dask = a_dask
        _b_dask = b_dask
        _weights_np = weights_np
    _weights_np = _preprocess_weights(_a_dask, dim, new_dim, _weights_np)

    axis = _a_dask.dims.index(new_dim)
    res = _pearson_r(_a_dask.values, _b_dask.values, _weights_np.values, axis)
    expected = actual.copy()
    expected.values = res
    assert_allclose(actual, expected)


@pytest.mark.parametrize('dim', AXES)
@pytest.mark.parametrize('weight', [True, False])
def test_pearson_r_p_value_xr(a, b, dim, weight, weights_ones, weights_latitude):
    # Generates subsetted weights to pass in as arg to main function and for the numpy testing.
    weights_arg, weights_np = adjust_weights(weight, dim, weights_ones, weights_latitude)

    actual = pearson_r_p_value(a, b, dim, weights=weights_arg)
    assert actual.chunks is None

    dim, _ = _preprocess_dims(dim)
    if len(dim) > 1:
        new_dim = '_'.join(dim)
        _a = a.stack(**{new_dim: dim})
        _b = b.stack(**{new_dim: dim})
        _weights_np = weights_np.stack(**{new_dim: dim})
    else:
        new_dim = dim[0]
        _a = a
        _b = b
        _weights_np = weights_np
    _weights_np = _preprocess_weights(_a, dim, new_dim, _weights_np)

    axis = _a.dims.index(new_dim)
    res = _pearson_r_p_value(_a.values, _b.values, _weights_np.values, axis)
    expected = actual.copy()
    expected.values = res
    assert_allclose(actual, expected)


@pytest.mark.parametrize('dim', AXES)
@pytest.mark.parametrize('weight', [True, False])
def test_pearson_r_p_value_xr_dask(a_dask, b_dask, dim, weight, weights_ones_dask, weights_latitude_dask):
    # Generates subsetted weights to pass in as arg to main function and for the numpy testing.
    weights_arg, weights_np = adjust_weights(weight, dim, weights_ones_dask, weights_latitude_dask)

    actual = pearson_r_p_value(a_dask, b_dask, dim, weights=weights_arg)
    assert actual.chunks is not None

    dim, _ = _preprocess_dims(dim)
    if len(dim) > 1:
        new_dim = '_'.join(dim)
        _a_dask = a_dask.stack(**{new_dim: dim})
        _b_dask = b_dask.stack(**{new_dim: dim})
        _weights_np = weights_np.stack(**{new_dim: dim})
    else:
        new_dim = dim[0]
        _a_dask = a_dask
        _b_dask = b_dask
        _weights_np = weights_np
    _weights_np = _preprocess_weights(_a_dask, dim, new_dim, _weights_np)

    axis = _a_dask.dims.index(new_dim)
    res = _pearson_r_p_value(_a_dask.values, _b_dask.values, _weights_np.values, axis)
    expected = actual.copy()
    expected.values = res
    assert_allclose(actual, expected)


@pytest.mark.parametrize('dim', AXES)
@pytest.mark.parametrize('weight', [True, False])
def test_rmse_r_xr(a, b, dim, weight, weights_ones, weights_latitude):
    # Generates subsetted weights to pass in as arg to main function and for the numpy testing.
    weights_arg, weights_np = adjust_weights(weight, dim, weights_ones, weights_latitude)

    actual = rmse(a, b, dim, weights=weights_arg)
    assert actual.chunks is None

    dim, axis = _preprocess_dims(dim)
    _a = a
    _b = b
    _weights_np = _preprocess_weights(_a, dim, dim, weights_np)
    axis = tuple(a.dims.index(d) for d in dim)
    res = _rmse(_a.values, _b.values, _weights_np.values, axis)
    expected = actual.copy()
    expected.values = res
    assert_allclose(actual, expected)


@pytest.mark.parametrize('dim', AXES)
@pytest.mark.parametrize('weight', [True, False])
def test_rmse_r_xr_dask(a_dask, b_dask, dim, weight, weights_ones_dask, weights_latitude_dask):
    # Generates subsetted weights to pass in as arg to main function and for the numpy testing.
    weights_arg, weights_np = adjust_weights(weight, dim, weights_ones_dask, weights_latitude_dask)

    actual = rmse(a_dask, b_dask, dim, weights=weights_arg)
    assert actual.chunks is not None

    dim, axis = _preprocess_dims(dim)
    _a_dask = a_dask
    _b_dask = b_dask
    _weights_np = _preprocess_weights(_a_dask, dim, dim, weights_np)
    axis = tuple(a_dask.dims.index(d) for d in dim)
    res = _rmse(_a_dask.values, _b_dask.values, _weights_np.values, axis)
    expected = actual.copy()
    expected.values = res
    assert_allclose(actual, expected)


@pytest.mark.parametrize('dim', AXES)
@pytest.mark.parametrize('weight', [True, False])
def test_mse_r_xr(a, b, dim, weight, weights_ones, weights_latitude):
    # Generates subsetted weights to pass in as arg to main function and for the numpy testing.
    weights_arg, weights_np = adjust_weights(weight, dim, weights_ones, weights_latitude)

    actual = mse(a, b, dim, weights=weights_arg)
    assert actual.chunks is None

    dim, axis = _preprocess_dims(dim)
    _a = a
    _b = b
    _weights_np = _preprocess_weights(_a, dim, dim, weights_np)
    axis = tuple(a.dims.index(d) for d in dim)
    res = _mse(_a.values, _b.values, _weights_np.values, axis)
    expected = actual.copy()
    expected.values = res
    assert_allclose(actual, expected)


@pytest.mark.parametrize('dim', AXES)
@pytest.mark.parametrize('weight', [True, False])
def test_mse_r_xr_dask(a_dask, b_dask, dim, weight, weights_ones_dask, weights_latitude_dask):
    # Generates subsetted weights to pass in as arg to main function and for the numpy testing.
    weights_arg, weights_np = adjust_weights(weight, dim, weights_ones_dask, weights_latitude_dask)

    actual = mse(a_dask, b_dask, dim, weights=weights_arg)
    assert actual.chunks is not None

    dim, axis = _preprocess_dims(dim)
    _a_dask = a_dask
    _b_dask = b_dask
    _weights_np = _preprocess_weights(_a_dask, dim, dim, weights_np)
    axis = tuple(a_dask.dims.index(d) for d in dim)
    res = _mse(_a_dask.values, _b_dask.values, _weights_np.values, axis)
    expected = actual.copy()
    expected.values = res
    assert_allclose(actual, expected)


@pytest.mark.parametrize('dim', AXES)
@pytest.mark.parametrize('weight', [True, False])
def test_mae_r_xr(a, b, dim, weight, weights_ones, weights_latitude):
    # Generates subsetted weights to pass in as arg to main function and for the numpy testing.
    weights_arg, weights_np = adjust_weights(weight, dim, weights_ones, weights_latitude)

    actual = mae(a, b, dim, weights=weights_arg)
    assert actual.chunks is None

    dim, axis = _preprocess_dims(dim)
    _a = a
    _b = b
    _weights_np = _preprocess_weights(_a, dim, dim, weights_np)
    axis = tuple(a.dims.index(d) for d in dim)
    res = _mae(_a.values, _b.values, _weights_np.values, axis)
    expected = actual.copy()
    expected.values = res
    assert_allclose(actual, expected)


@pytest.mark.parametrize('dim', AXES)
@pytest.mark.parametrize('weight', [True, False])
def test_mae_r_xr_dask(a_dask, b_dask, dim, weight, weights_ones_dask, weights_latitude_dask):
    # Generates subsetted weights to pass in as arg to main function and for the numpy testing.
    weights_arg, weights_np = adjust_weights(weight, dim, weights_ones_dask, weights_latitude_dask)

    actual = mae(a_dask, b_dask, dim, weights=weights_arg)
    assert actual.chunks is not None

    dim, axis = _preprocess_dims(dim)
    _a_dask = a_dask
    _b_dask = b_dask
    _weights_np = _preprocess_weights(_a_dask, dim, dim, weights_np)
    axis = tuple(a_dask.dims.index(d) for d in dim)
    res = _mae(_a_dask.values, _b_dask.values, _weights_np.values, axis)
    expected = actual.copy()
    expected.values = res
    assert_allclose(actual, expected)
