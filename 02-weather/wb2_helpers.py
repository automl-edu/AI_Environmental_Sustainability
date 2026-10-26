"""Shared helpers for the WeatherBench 2 analysis notebooks (January 2020, 64x32 grid).

All comparisons align forecast and ERA5 by coordinates (time, latitude, longitude), never by array position.
"""
import os
import warnings
from functools import lru_cache

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

warnings.filterwarnings('ignore')

DATA_DIR = os.environ.get('WB2_DATA_DIR', 'data')
FIG_DIR = 'figures'
os.makedirs(FIG_DIR, exist_ok=True)

LEADS = {3: 72, 5: 120, 7: 168}                      # days -> hours
VARS = {'2m_temperature': '2-m temperature (K)',
        '10m_wind_speed': '10-m wind speed (m/s)',
        'geopotential': 'Z500 (m²/s²)'}
SHORT = {'2m_temperature': 'T2m', '10m_wind_speed': 'Wind10m', 'geopotential': 'Z500'}

DET = {'HRES': 'hres', 'GraphCast': 'graphcast', 'Pangu': 'pangu', 'NeuralGCM': 'neuralgcm'}
ENS = {'IFS ENS': ('ifs_ens', 'number'), 'GenCast': ('gencast', 'sample'),
       'NeuralGCM ENS': ('neuralgcm_ens', 'realization')}
PHYSICS = {'HRES', 'IFS ENS'}
COLORS = {'HRES': '#222222', 'IFS ENS': '#777777', 'GraphCast': '#1f77b4', 'GenCast': '#17becf',
          'Pangu': '#ff7f0e', 'NeuralGCM': '#2ca02c', 'NeuralGCM ENS': '#98df8a'}
STYLE = {m: ('--' if m in PHYSICS else '-') for m in COLORS}

CITIES = {'Hannover': (52.4, 9.7), 'Madrid': (40.4, -3.7), 'Reykjavik': (64.1, -21.9),
          'New York': (40.7, -74.0), 'Tokyo': (35.7, 139.7), 'Nairobi': (-1.3, 36.8)}


# ─────────────────────────── loading ───────────────────────────
@lru_cache(maxsize=None)
def open_local(name):
    path = os.path.join(DATA_DIR, f'{name}_jan2020.zarr')
    if not os.path.exists(path):
        raise FileNotFoundError(f'Missing {path}. Ask the instructor for the prepared data/ folder.')
    return xr.open_zarr(path, consolidated=False, decode_timedelta=False)


def era5():
    return open_local('era5')


def lead_hours(ds):
    v = ds.prediction_timedelta.values
    if np.issubdtype(v.dtype, np.timedelta64):
        return (v / np.timedelta64(1, 'h')).astype(float)
    units = str(ds.prediction_timedelta.attrs.get('units', 'hours')).lower()
    for key, f in [('nanosecond', 1 / 3.6e12), ('second', 1 / 3600), ('minute', 1 / 60), ('day', 24), ('hour', 1)]:
        if units.startswith(key):
            return v.astype(float) * f
    return v.astype(float)


def _truth(variable):
    tr = era5()[variable]
    if 'level' in tr.dims:
        tr = tr.sel(level=500)
    return tr.drop_vars([c for c in ('level',) if c in tr.coords])


def get_forecast(name, variable, lead_h):
    """Forecast of dataset `name` at one lead time plus matching ERA5 truth, aligned by coordinates."""
    ds = open_local(name)
    if variable not in ds:
        return None, None
    hrs = lead_hours(ds)
    idx = np.where(np.isclose(hrs, lead_h))[0]
    if len(idx) == 0:
        return None, None
    fc = ds[variable].isel(prediction_timedelta=int(idx[0]))
    if 'level' in fc.dims:
        fc = fc.sel(level=500)
    fc = fc.drop_vars([c for c in ('prediction_timedelta', 'level') if c in fc.coords])
    fc = fc.assign_coords(time=fc.time.values.astype('datetime64[ns]') + np.timedelta64(int(lead_h), 'h'))
    tr = _truth(variable)
    common = np.intersect1d(fc.time.values, tr.time.values)
    fc, tr = fc.sel(time=common), tr.sel(time=common)
    tr = tr.sel(latitude=fc.latitude, longitude=fc.longitude, method='nearest')
    assert np.abs(tr.latitude.values - fc.latitude.values).max() < 0.5, 'latitude grids do not match'
    tr = tr.assign_coords(latitude=fc.latitude, longitude=fc.longitude)
    fc.name = tr.name = variable
    return fc.load(), tr.load()


def det_forecast(model, variable, lead_h):
    return get_forecast(DET[model], variable, lead_h)


def ens_forecast(model, variable, lead_h, n_members=None, seed=0):
    name, mdim = ENS[model]
    fc, tr = get_forecast(name, variable, lead_h)
    if fc is not None and n_members and fc.sizes[mdim] > n_members:
        fc = subensemble(fc, mdim, n_members, seed)
    return fc, tr


# ─────────────────────────── metrics ───────────────────────────
def wmean(da, dims=('latitude', 'longitude')):
    return da.weighted(np.cos(np.deg2rad(da.latitude))).mean(dims)


def rmse(fc, tr):
    return float(np.sqrt(wmean((fc - tr) ** 2).mean('time')))


def bias(fc, tr):
    return float(wmean(fc - tr).mean('time'))


def crps(fc, tr, member_dim, fair=False):
    """Ensemble CRPS (lower = better). fair=True removes the penalty for small ensembles."""
    x = np.sort(fc.transpose(*tr.dims, member_dim).values, axis=-1)
    y = tr.values[..., None]
    M = x.shape[-1]
    k = np.arange(1, M + 1)
    spread_term = (x * (2 * k - M - 1)).sum(-1)
    denom = M * (M - 1) if (fair and M > 1) else M ** 2
    score = np.abs(x - y).mean(-1) - spread_term / denom
    return float(wmean(xr.DataArray(score, dims=tr.dims, coords=tr.coords)).mean('time'))


def spread_skill(fc, tr, member_dim):
    M = fc.sizes[member_dim]
    spread = np.sqrt(float(wmean(fc.var(member_dim, ddof=1)).mean('time')))
    skill = rmse(fc.mean(member_dim), tr)
    return float(np.sqrt((M + 1) / M) * spread / skill)


def subensemble(fc, member_dim, k, seed=0):
    idx = np.sort(np.random.default_rng(seed).choice(fc.sizes[member_dim], size=k, replace=False))
    return fc.isel({member_dim: idx})


def rank_histogram(fc, tr, member_dim, n_points=40000, seed=0):
    M = fc.sizes[member_dim]
    x = fc.transpose(*tr.dims, member_dim).values.reshape(-1, M)
    y = tr.values.reshape(-1)
    w = np.broadcast_to(np.cos(np.deg2rad(tr.latitude.values)), tr.shape).reshape(-1)
    idx = np.random.default_rng(seed).choice(len(y), size=min(n_points, len(y)), replace=False, p=w / w.sum())
    ranks = (x[idx] < y[idx, None]).sum(1)
    return np.bincount(ranks, minlength=M + 1) / len(idx)


# ─────────────────────────── climatology, anomalies, realism ───────────────────────────
def climatology(variable):
    return _truth(variable).sel(time=slice('2020-01-01', '2020-01-31')).mean('time')


def anomaly(da):
    c = climatology(da.name).sel(latitude=da.latitude, longitude=da.longitude, method='nearest')
    return da - c.assign_coords(latitude=da.latitude, longitude=da.longitude)


def activity_ratio(fc, tr):
    """Strength of the forecast's weather patterns relative to reality (1 = as strong, < 1 = smoother)."""
    act = lambda d: float(np.sqrt(wmean(anomaly(d) ** 2).mean('time')))
    return act(fc) / act(tr)


def zonal_spectrum(da, lat_min=30, lat_max=70):
    """Mean power per zonal wavenumber in a latitude band (deviation from the zonal mean)."""
    band = da.sel(latitude=(da.latitude >= lat_min) & (da.latitude <= lat_max))
    x = band.transpose(..., 'longitude').values
    x = x - x.mean(-1, keepdims=True)
    p = np.abs(np.fft.rfft(x, axis=-1)) ** 2
    return p.reshape(-1, p.shape[-1]).mean(0)[1:]          # drop wavenumber 0


# ─────────────────────────── regions and points ───────────────────────────
REGIONS = {
    'Global': lambda lat, lon: np.ones_like(lat, dtype=bool),
    'Tropics (20°S–20°N)': lambda lat, lon: np.abs(lat) <= 20,
    'Northern extratropics': lambda lat, lon: lat > 20,
    'Southern extratropics': lambda lat, lon: lat < -20,
    'Europe': lambda lat, lon: (lat >= 35) & (lat <= 72) & ((lon >= 345) | (lon <= 45)),
}


def regional_rmse(fc, tr, region):
    lat2, lon2 = xr.broadcast(fc.latitude, fc.longitude)
    mask = xr.DataArray(REGIONS[region](lat2.values, lon2.values), coords=lat2.coords)
    err2 = ((fc - tr) ** 2).where(mask)
    w = np.cos(np.deg2rad(fc.latitude)) * mask
    return float(np.sqrt((err2.weighted(w).mean(('latitude', 'longitude'))).mean('time')))


def point(da, city):
    lat, lon = CITIES[city]
    return da.sel(latitude=lat, longitude=lon % 360, method='nearest')


# ─────────────────────────── plotting ───────────────────────────
MAP_PROJ = ccrs.Robinson()


def map_subplots(nrows=1, ncols=1, **kw):
    """Like plt.subplots, but every panel is a cartopy map."""
    return plt.subplots(nrows, ncols, subplot_kw={'projection': MAP_PROJ}, **kw)


def _pm180(da):
    return da.assign_coords(longitude=((da.longitude + 180) % 360) - 180).sortby('longitude')


def plot_map(ax, da, title, cmap='RdBu_r', vmin=None, vmax=None, label='', city=None):
    """Draw a lat/lon field on a cartopy axis created with map_subplots."""
    d = _pm180(da).transpose('latitude', 'longitude')
    im = ax.pcolormesh(d.longitude, d.latitude, d, cmap=cmap, vmin=vmin, vmax=vmax, shading='nearest',
                       transform=ccrs.PlateCarree())
    ax.set_global()
    ax.coastlines(linewidth=0.6)
    ax.gridlines(linewidth=0.3, color='gray', alpha=0.5)
    if city:
        lat, lon = CITIES[city]
        ax.plot(lon, lat, 'k*', ms=9, transform=ccrs.PlateCarree())
    ax.set_title(title, fontsize=9)
    plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.85).set_label(label, fontsize=8)


def save(fig, number):
    fig.savefig(os.path.join(FIG_DIR, f'fig_{number}.png'), dpi=130, bbox_inches='tight')
    plt.show()


def available():
    """Which variables each dataset provides in our subset."""
    rows = {}
    for label, name in list(DET.items()) + [(k, v[0]) for k, v in ENS.items()]:
        ds = open_local(name)
        rows[label] = {VARS[v]: ('✅' if v in ds else '—') for v in VARS}
    return pd.DataFrame(rows)
