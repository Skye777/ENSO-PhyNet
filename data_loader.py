import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.realpath(".."))
sys.path.append('/home/dl/Desktop/vit/')

import xarray as xr
import numpy as np
from einops import rearrange
from torch.utils.data import Dataset, ConcatDataset
from sklearn.preprocessing import MinMaxScaler
from plot_helper import plot_grid

from multivar.configs import args

LLAT, RLAT, LLON, RLON = [-30, 30, 120, 280]
HB_PAD = 10
theta = 10

def normalization(data, feature_range=(0, 1)):
    data_max = data.max()
    data_min = data.min()
    # print('MAX:', data_max, ',MIN:', data_min)

    data_scaled = ((data-data_min) / (data_max-data_min)) * (feature_range[1]-feature_range[0])+feature_range[0]
    return data_scaled


def load_gfdl_data():
    LLAT, RLAT, LLON, RLON = [52, 112, 120, 280]
    h, w = args['input_size']

    sst = xr.open_dataset(os.path.join(
        args['gfdl_dir'] + 'meta_data/', 'sst.nc'))['sst'][:, LLAT:RLAT, LLON:RLON].fillna(0)

    Up_Tb = xr.open_dataset(os.path.join(
        args['gfdl_dir'] + 'process_field/anomaly_term', 'Up_Tb.nc'))['Up_Tb'].fillna(0)

    Vb_Tp = xr.open_dataset(os.path.join(
        args['gfdl_dir'] + 'process_field/anomaly_term', 'Vb_Tp.nc'))['Vb_Tp'].fillna(0)

    Vp_Tb = xr.open_dataset(os.path.join(
        args['gfdl_dir'] + 'process_field/anomaly_term', 'Vp_Tb.nc'))['Vp_Tb'].fillna(0)

    Qzz_Wb_Tp = xr.open_dataset(os.path.join(
        args['gfdl_dir'] + 'process_field/anomaly_term', 'Qzz_Wb_Tp.nc'))['Qzz_Wb_Tp'].fillna(0)
    
    Qzz_Wp_Tb = xr.open_dataset(os.path.join(
        args['gfdl_dir'] + 'process_field/anomaly_term', 'Qzz_Wp_Tb.nc'))['Qzz_Wp_Tb'].fillna(0)
    
    Qw_Wb_Tp = xr.open_dataset(os.path.join(
        args['gfdl_dir'] + 'process_field/anomaly_term', 'Qw_Wb_Tp.nc'))['Qw_Wb_Tp'].fillna(0)
    
    hb_term = np.stack([Up_Tb, Vb_Tp, Vp_Tb, Qzz_Wb_Tp, Qzz_Wp_Tb, Qw_Wb_Tp], axis=0)
    hb_term = np.where(hb_term > theta, theta, hb_term)
    hb_term = np.where(hb_term < -theta, -theta, hb_term)
    hb_term = np.pad(hb_term, ((0, 0), (0, 0), (HB_PAD, HB_PAD), (0, 0)))

    scaler = MinMaxScaler()
    sst = np.reshape(scaler.fit_transform(np.reshape(sst.values, (-1, h * w))), (-1, h, w))
    hb_term = [np.reshape(scaler.fit_transform(np.reshape(hb_term[i], (-1, h * w))), (-1, h, w)) for i in range(len(hb_term))]

    data = np.concatenate((np.expand_dims(sst, axis=-1), np.stack(hb_term, axis=-1)), axis=-1)

    return data


def load_e3sm_data():
    h, w = args['input_size']

    sst = xr.open_dataset(os.path.join(
        args['E3SM_dir'], 'sst.nc'))['tos'].loc['1850-01-16':'2014-12-16'].fillna(0)
    lc = sst.coords['lon']
    la = sst.coords['lat']
    sst = sst.loc[dict(lon=lc[(lc >= LLON) & (lc <= RLON)], lat=la[(la >= LLAT) & (la <= RLAT)])]
    
    Up_Tb = xr.open_dataset(os.path.join(
        args['E3SM_dir'] + 'process_field/anomaly_term', 'Up_Tb.nc'))['Up_Tb'].loc['1850-01-16':'2014-12-16'].fillna(0)

    Vb_Tp = xr.open_dataset(os.path.join(
        args['E3SM_dir'] + 'process_field/anomaly_term', 'Vb_Tp.nc'))['Vb_Tp'].loc['1850-01-16':'2014-12-16'].fillna(0)

    Vp_Tb = xr.open_dataset(os.path.join(
        args['E3SM_dir'] + 'process_field/anomaly_term', 'Vp_Tb.nc'))['Vp_Tb'].loc['1850-01-16':'2014-12-16'].fillna(0)

    Qzz_Wb_Tp = xr.open_dataset(os.path.join(
        args['E3SM_dir'] + 'process_field/anomaly_term', 'Qzz_Wb_Tp.nc'))['Qzz_Wb_Tp'].loc['1850-01-16':'2014-12-16'].fillna(0)
    
    Qzz_Wp_Tb = xr.open_dataset(os.path.join(
        args['E3SM_dir'] + 'process_field/anomaly_term', 'Qzz_Wp_Tb.nc'))['Qzz_Wp_Tb'].loc['1850-01-16':'2014-12-16'].fillna(0)
    
    Qw_Wb_Tp = xr.open_dataset(os.path.join(
        args['E3SM_dir'] + 'process_field/anomaly_term', 'Qw_Wb_Tp.nc'))['Qw_Wb_Tp'].loc['1850-01-16':'2014-12-16'].fillna(0)
    
    hb_term = np.stack([Up_Tb, Vb_Tp, Vp_Tb, Qzz_Wb_Tp, Qzz_Wp_Tb, Qw_Wb_Tp], axis=0)
    hb_term = np.where(hb_term > theta, theta, hb_term)
    hb_term = np.where(hb_term < -theta, -theta, hb_term)
    hb_term = np.pad(hb_term, ((0, 0), (0, 0), (HB_PAD, HB_PAD), (0, 0)))

    scaler = MinMaxScaler()
    sst = np.reshape(scaler.fit_transform(np.reshape(sst.values, (-1, h * w))), (-1, h, w))
    hb_term = [np.reshape(scaler.fit_transform(np.reshape(hb_term[i], (-1, h * w))), (-1, h, w)) for i in range(len(hb_term))]

    data = np.concatenate((np.expand_dims(sst, axis=-1), np.stack(hb_term, axis=-1)), axis=-1)

    return data


def load_noresm2_data():
    h, w = args['input_size']

    sst = xr.open_dataset(os.path.join(
        args['NorESM2_dir'], 'sst.nc'))['tos'].loc['1850-01-16':'2014-12-16'].fillna(0)
    lc = sst.coords['lon']
    la = sst.coords['lat']
    sst = sst.loc[dict(lon=lc[(lc > LLON) & (lc <= RLON)], lat=la[(la >= LLAT) & (la <= RLAT)])]
    
    Up_Tb = xr.open_dataset(os.path.join(
        args['NorESM2_dir'] + 'process_field/anomaly_term', 'Up_Tb.nc'))['Up_Tb'].loc['1850-01-16':'2014-12-16'].fillna(0)

    Vb_Tp = xr.open_dataset(os.path.join(
        args['NorESM2_dir'] + 'process_field/anomaly_term', 'Vb_Tp.nc'))['Vb_Tp'].loc['1850-01-16':'2014-12-16'].fillna(0)

    Vp_Tb = xr.open_dataset(os.path.join(
        args['NorESM2_dir'] + 'process_field/anomaly_term', 'Vp_Tb.nc'))['Vp_Tb'].loc['1850-01-16':'2014-12-16'].fillna(0)

    Qzz_Wb_Tp = xr.open_dataset(os.path.join(
        args['NorESM2_dir'] + 'process_field/anomaly_term', 'Qzz_Wb_Tp.nc'))['Qzz_Wb_Tp'].loc['1850-01-16':'2014-12-16'].fillna(0)
    
    Qzz_Wp_Tb = xr.open_dataset(os.path.join(
        args['NorESM2_dir'] + 'process_field/anomaly_term', 'Qzz_Wp_Tb.nc'))['Qzz_Wp_Tb'].loc['1850-01-16':'2014-12-16'].fillna(0)
    
    Qw_Wb_Tp = xr.open_dataset(os.path.join(
        args['NorESM2_dir'] + 'process_field/anomaly_term', 'Qw_Wb_Tp.nc'))['Qw_Wb_Tp'].loc['1850-01-16':'2014-12-16'].fillna(0)
    
    hb_term = np.stack([Up_Tb, Vb_Tp, Vp_Tb, Qzz_Wb_Tp, Qzz_Wp_Tb, Qw_Wb_Tp], axis=0)
    hb_term = np.where(hb_term > theta, theta, hb_term)
    hb_term = np.where(hb_term < -theta, -theta, hb_term)
    hb_term = np.pad(hb_term, ((0, 0), (0, 0), (HB_PAD, HB_PAD), (0, 0)))

    scaler = MinMaxScaler()
    sst = np.reshape(scaler.fit_transform(np.reshape(sst.values, (-1, h * w))), (-1, h, w))
    hb_term = [np.reshape(scaler.fit_transform(np.reshape(hb_term[i], (-1, h * w))), (-1, h, w)) for i in range(len(hb_term))]

    data = np.concatenate((np.expand_dims(sst, axis=-1), np.stack(hb_term, axis=-1)), axis=-1)

    return data


def load_cesm2_data():
    h, w = args['input_size']

    sst = xr.open_dataset(os.path.join(
        args['CESM2_dir'], 'sst.nc'))['tos'].loc['1850-01-15':'2014-12-15'].fillna(0)
    lc = sst.coords['lon']
    la = sst.coords['lat']
    sst = sst.loc[dict(lon=lc[(lc >= LLON) & (lc <= RLON)], lat=la[(la >= LLAT) & (la <= RLAT)])]
    
    Up_Tb = xr.open_dataset(os.path.join(
        args['CESM2_dir'] + 'process_field/anomaly_term', 'Up_Tb.nc'))['Up_Tb'].loc['1850-01-16':'2014-12-16'].fillna(0)

    Vb_Tp = xr.open_dataset(os.path.join(
        args['CESM2_dir'] + 'process_field/anomaly_term', 'Vb_Tp.nc'))['Vb_Tp'].loc['1850-01-16':'2014-12-16'].fillna(0)

    Vp_Tb = xr.open_dataset(os.path.join(
        args['CESM2_dir'] + 'process_field/anomaly_term', 'Vp_Tb.nc'))['Vp_Tb'].loc['1850-01-16':'2014-12-16'].fillna(0)

    Qzz_Wb_Tp = xr.open_dataset(os.path.join(
        args['CESM2_dir'] + 'process_field/anomaly_term', 'Qzz_Wb_Tp.nc'))['Qzz_Wb_Tp'].loc['1850-01-16':'2014-12-16'].fillna(0)
    
    Qzz_Wp_Tb = xr.open_dataset(os.path.join(
        args['CESM2_dir'] + 'process_field/anomaly_term', 'Qzz_Wp_Tb.nc'))['Qzz_Wp_Tb'].loc['1850-01-16':'2014-12-16'].fillna(0)
    
    Qw_Wb_Tp = xr.open_dataset(os.path.join(
        args['CESM2_dir'] + 'process_field/anomaly_term', 'Qw_Wb_Tp.nc'))['Qw_Wb_Tp'].loc['1850-01-16':'2014-12-16'].fillna(0)
    
    hb_term = np.stack([Up_Tb, Vb_Tp, Vp_Tb, Qzz_Wb_Tp, Qzz_Wp_Tb, Qw_Wb_Tp], axis=0)
    hb_term = np.where(hb_term > theta, theta, hb_term)
    hb_term = np.where(hb_term < -theta, -theta, hb_term)
    hb_term = np.pad(hb_term, ((0, 0), (0, 0), (HB_PAD, HB_PAD), (0, 0)))

    scaler = MinMaxScaler()
    sst = np.reshape(scaler.fit_transform(np.reshape(sst.values, (-1, h * w))), (-1, h, w))
    hb_term = [np.reshape(scaler.fit_transform(np.reshape(hb_term[i], (-1, h * w))), (-1, h, w)) for i in range(len(hb_term))]

    data = np.concatenate((np.expand_dims(sst, axis=-1), np.stack(hb_term, axis=-1)), axis=-1)

    return data


def load_cams_data():
    h, w = args['input_size']

    sst = xr.open_dataset(os.path.join(
        args['CAMS_dir'], 'sst.nc'))['tos'].loc['1850-01-16':'2014-12-16'].fillna(0)
    lc = sst.coords['lon']
    la = sst.coords['lat']
    sst = sst.loc[dict(lon=lc[(lc > LLON) & (lc <= RLON)], lat=la[(la >= LLAT) & (la <= RLAT)])]
    
    Up_Tb = xr.open_dataset(os.path.join(
        args['CAMS_dir'] + 'process_field/anomaly_term', 'Up_Tb.nc'))['Up_Tb'].loc['1850-01-16':'2014-12-16'].fillna(0)

    Vb_Tp = xr.open_dataset(os.path.join(
        args['CAMS_dir'] + 'process_field/anomaly_term', 'Vb_Tp.nc'))['Vb_Tp'].loc['1850-01-16':'2014-12-16'].fillna(0)

    Vp_Tb = xr.open_dataset(os.path.join(
        args['CAMS_dir'] + 'process_field/anomaly_term', 'Vp_Tb.nc'))['Vp_Tb'].loc['1850-01-16':'2014-12-16'].fillna(0)

    Qzz_Wb_Tp = xr.open_dataset(os.path.join(
        args['CAMS_dir'] + 'process_field/anomaly_term', 'Qzz_Wb_Tp.nc'))['Qzz_Wb_Tp'].loc['1850-01-16':'2014-12-16'].fillna(0)
    
    Qzz_Wp_Tb = xr.open_dataset(os.path.join(
        args['CAMS_dir'] + 'process_field/anomaly_term', 'Qzz_Wp_Tb.nc'))['Qzz_Wp_Tb'].loc['1850-01-16':'2014-12-16'].fillna(0)
    
    Qw_Wb_Tp = xr.open_dataset(os.path.join(
        args['CAMS_dir'] + 'process_field/anomaly_term', 'Qw_Wb_Tp.nc'))['Qw_Wb_Tp'].loc['1850-01-16':'2014-12-16'].fillna(0)
    
    hb_term = np.stack([Up_Tb, Vb_Tp, Vp_Tb, Qzz_Wb_Tp, Qzz_Wp_Tb, Qw_Wb_Tp], axis=0)
    hb_term = np.where(hb_term > theta, theta, hb_term)
    hb_term = np.where(hb_term < -theta, -theta, hb_term)
    hb_term = np.pad(hb_term, ((0, 0), (0, 0), (HB_PAD, HB_PAD), (0, 0)))

    scaler = MinMaxScaler()
    sst = np.reshape(scaler.fit_transform(np.reshape(sst.values, (-1, h * w))), (-1, h, w))
    hb_term = [np.reshape(scaler.fit_transform(np.reshape(hb_term[i], (-1, h * w))), (-1, h, w)) for i in range(len(hb_term))]

    data = np.concatenate((np.expand_dims(sst, axis=-1), np.stack(hb_term, axis=-1)), axis=-1)

    return data


def load_access_data():
    h, w = args['input_size']

    sst = xr.open_dataset(os.path.join(
        args['ACCESS_dir'], 'sst.nc'))['tos'].loc['1850-01-16':'2014-12-16'].fillna(0)
    lc = sst.coords['lon']
    la = sst.coords['lat']
    sst = sst.loc[dict(lon=lc[(lc > LLON) & (lc <= RLON)], lat=la[(la >= LLAT) & (la <= RLAT)])]
    
    Up_Tb = xr.open_dataset(os.path.join(
        args['ACCESS_dir'] + 'process_field/anomaly_term', 'Up_Tb.nc'))['Up_Tb'].loc['1850-01-16':'2014-12-16'].fillna(0)

    Vb_Tp = xr.open_dataset(os.path.join(
        args['ACCESS_dir'] + 'process_field/anomaly_term', 'Vb_Tp.nc'))['Vb_Tp'].loc['1850-01-16':'2014-12-16'].fillna(0)

    Vp_Tb = xr.open_dataset(os.path.join(
        args['ACCESS_dir'] + 'process_field/anomaly_term', 'Vp_Tb.nc'))['Vp_Tb'].loc['1850-01-16':'2014-12-16'].fillna(0)

    Qzz_Wb_Tp = xr.open_dataset(os.path.join(
        args['ACCESS_dir'] + 'process_field/anomaly_term', 'Qzz_Wb_Tp.nc'))['Qzz_Wb_Tp'].loc['1850-01-16':'2014-12-16'].fillna(0)
    
    Qzz_Wp_Tb = xr.open_dataset(os.path.join(
        args['ACCESS_dir'] + 'process_field/anomaly_term', 'Qzz_Wp_Tb.nc'))['Qzz_Wp_Tb'].loc['1850-01-16':'2014-12-16'].fillna(0)
    
    Qw_Wb_Tp = xr.open_dataset(os.path.join(
        args['ACCESS_dir'] + 'process_field/anomaly_term', 'Qw_Wb_Tp.nc'))['Qw_Wb_Tp'].loc['1850-01-16':'2014-12-16'].fillna(0)
    
    hb_term = np.stack([Up_Tb, Vb_Tp, Vp_Tb, Qzz_Wb_Tp, Qzz_Wp_Tb, Qw_Wb_Tp], axis=0)
    hb_term = np.where(hb_term > theta, theta, hb_term)
    hb_term = np.where(hb_term < -theta, -theta, hb_term)
    hb_term = np.pad(hb_term, ((0, 0), (0, 0), (HB_PAD, HB_PAD), (0, 0)))

    scaler = MinMaxScaler()
    sst = np.reshape(scaler.fit_transform(np.reshape(sst.values, (-1, h * w))), (-1, h, w))
    hb_term = [np.reshape(scaler.fit_transform(np.reshape(hb_term[i], (-1, h * w))), (-1, h, w)) for i in range(len(hb_term))]

    data = np.concatenate((np.expand_dims(sst, axis=-1), np.stack(hb_term, axis=-1)), axis=-1)

    return data


def load_cesm2_waccm_data():
    h, w = args['input_size']

    sst = xr.open_dataset(os.path.join(
        args['CESM2_WACCM_dir'], 'sst.nc'))['tos'].loc['1850-01-15':'2014-12-15'].fillna(0)
    lc = sst.coords['lon']
    la = sst.coords['lat']
    sst = sst.loc[dict(lon=lc[(lc > LLON) & (lc <= RLON)], lat=la[(la >= LLAT) & (la <= RLAT)])]
    
    Up_Tb = xr.open_dataset(os.path.join(
        args['CESM2_WACCM_dir'] + 'process_field/anomaly_term', 'Up_Tb.nc'))['Up_Tb'].loc['1850-01-15':'2014-12-15'].fillna(0)

    Vb_Tp = xr.open_dataset(os.path.join(
        args['CESM2_WACCM_dir'] + 'process_field/anomaly_term', 'Vb_Tp.nc'))['Vb_Tp'].loc['1850-01-15':'2014-12-15'].fillna(0)

    Vp_Tb = xr.open_dataset(os.path.join(
        args['CESM2_WACCM_dir'] + 'process_field/anomaly_term', 'Vp_Tb.nc'))['Vp_Tb'].loc['1850-01-15':'2014-12-15'].fillna(0)

    Qzz_Wb_Tp = xr.open_dataset(os.path.join(
        args['CESM2_WACCM_dir'] + 'process_field/anomaly_term', 'Qzz_Wb_Tp.nc'))['Qzz_Wb_Tp'].loc['1850-01-15':'2014-12-15'].fillna(0)
    
    Qzz_Wp_Tb = xr.open_dataset(os.path.join(
        args['CESM2_WACCM_dir'] + 'process_field/anomaly_term', 'Qzz_Wp_Tb.nc'))['Qzz_Wp_Tb'].loc['1850-01-15':'2014-12-15'].fillna(0)
    
    Qw_Wb_Tp = xr.open_dataset(os.path.join(
        args['CESM2_WACCM_dir'] + 'process_field/anomaly_term', 'Qw_Wb_Tp.nc'))['Qw_Wb_Tp'].loc['1850-01-15':'2014-12-15'].fillna(0)
    
    hb_term = np.stack([Up_Tb, Vb_Tp, Vp_Tb, Qzz_Wb_Tp, Qzz_Wp_Tb, Qw_Wb_Tp], axis=0)
    hb_term = np.where(hb_term > theta, theta, hb_term)
    hb_term = np.where(hb_term < -theta, -theta, hb_term)
    hb_term = np.pad(hb_term, ((0, 0), (0, 0), (HB_PAD, HB_PAD), (0, 0)))

    scaler = MinMaxScaler()
    sst = np.reshape(scaler.fit_transform(np.reshape(sst.values, (-1, h * w))), (-1, h, w))
    hb_term = [np.reshape(scaler.fit_transform(np.reshape(hb_term[i], (-1, h * w))), (-1, h, w)) for i in range(len(hb_term))]

    data = np.concatenate((np.expand_dims(sst, axis=-1), np.stack(hb_term, axis=-1)), axis=-1)

    return data


def load_fio_data():
    h, w = args['input_size']

    sst = xr.open_dataset(os.path.join(
        args['FIO_dir'], 'sst.nc'))['tos'].loc['1850-01-16':'2014-12-16'].fillna(0)
    lc = sst.coords['lon']
    la = sst.coords['lat']
    sst = sst.loc[dict(lon=lc[(lc > LLON) & (lc <= RLON)], lat=la[(la >= LLAT) & (la <= RLAT)])]
    
    Up_Tb = xr.open_dataset(os.path.join(
        args['FIO_dir'] + 'process_field/anomaly_term', 'Up_Tb.nc'))['Up_Tb'].loc['1850-01-16':'2014-12-16'].fillna(0)

    Vb_Tp = xr.open_dataset(os.path.join(
        args['FIO_dir'] + 'process_field/anomaly_term', 'Vb_Tp.nc'))['Vb_Tp'].loc['1850-01-16':'2014-12-16'].fillna(0)

    Vp_Tb = xr.open_dataset(os.path.join(
        args['FIO_dir'] + 'process_field/anomaly_term', 'Vp_Tb.nc'))['Vp_Tb'].loc['1850-01-16':'2014-12-16'].fillna(0)

    Qzz_Wb_Tp = xr.open_dataset(os.path.join(
        args['FIO_dir'] + 'process_field/anomaly_term', 'Qzz_Wb_Tp.nc'))['Qzz_Wb_Tp'].loc['1850-01-16':'2014-12-16'].fillna(0)
    
    Qzz_Wp_Tb = xr.open_dataset(os.path.join(
        args['FIO_dir'] + 'process_field/anomaly_term', 'Qzz_Wp_Tb.nc'))['Qzz_Wp_Tb'].loc['1850-01-16':'2014-12-16'].fillna(0)
    
    Qw_Wb_Tp = xr.open_dataset(os.path.join(
        args['FIO_dir'] + 'process_field/anomaly_term', 'Qw_Wb_Tp.nc'))['Qw_Wb_Tp'].loc['1850-01-16':'2014-12-16'].fillna(0)
    
    hb_term = np.stack([Up_Tb, Vb_Tp, Vp_Tb, Qzz_Wb_Tp, Qzz_Wp_Tb, Qw_Wb_Tp], axis=0)
    hb_term = np.where(hb_term > theta, theta, hb_term)
    hb_term = np.where(hb_term < -theta, -theta, hb_term)
    hb_term = np.pad(hb_term, ((0, 0), (0, 0), (HB_PAD, HB_PAD), (0, 0)))

    scaler = MinMaxScaler()
    sst = np.reshape(scaler.fit_transform(np.reshape(sst.values, (-1, h * w))), (-1, h, w))
    hb_term = [np.reshape(scaler.fit_transform(np.reshape(hb_term[i], (-1, h * w))), (-1, h, w)) for i in range(len(hb_term))]

    data = np.concatenate((np.expand_dims(sst, axis=-1), np.stack(hb_term, axis=-1)), axis=-1)

    return data


def load_soda_field():

    h, w = args['input_size']

    sst = xr.open_dataset(os.path.join(
        args['soda_dir'] + 'total_field/', 'sst.nc'))['sst'].loc['1871-01-16':'2022-12-16'].fillna(0)
    lc = sst.coords['longitude']
    la = sst.coords['latitude']
    sst = sst.loc[dict(longitude=lc[(lc >= LLON) & (lc <= RLON)], latitude=la[(la >= LLAT) & (la <= RLAT)])]
    
    Up_Tb = xr.open_dataset(os.path.join(
        args['soda_dir'] + 'process_field/anomaly_term', 'Up_Tb.nc'))['Up_Tb'].loc['1871-01-15':'2005-12-15'].fillna(0)

    Vb_Tp = xr.open_dataset(os.path.join(
        args['soda_dir'] + 'process_field/anomaly_term', 'Vb_Tp.nc'))['Vb_Tp'].loc['1871-01-15':'2005-12-15'].fillna(0)

    Vp_Tb = xr.open_dataset(os.path.join(
        args['soda_dir'] + 'process_field/anomaly_term', 'Vp_Tb.nc'))['Vp_Tb'].loc['1871-01-15':'2005-12-15'].fillna(0)

    Qzz_Wb_Tp = xr.open_dataset(os.path.join(
        args['soda_dir'] + 'process_field/anomaly_term', 'Qzz_Wb_Tp.nc'))['Qzz_Wb_Tp'].loc['1871-01-15':'2005-12-15'].fillna(0)
    
    Qzz_Wp_Tb = xr.open_dataset(os.path.join(
        args['soda_dir'] + 'process_field/anomaly_term', 'Qzz_Wp_Tb.nc'))['Qzz_Wp_Tb'].loc['1871-01-15':'2005-12-15'].fillna(0)
    
    Qw_Wb_Tp = xr.open_dataset(os.path.join(
        args['soda_dir'] + 'process_field/anomaly_term', 'Qw_Wb_Tp.nc'))['Qw_Wb_Tp'].loc['1871-01-15':'2005-12-15'].fillna(0)
    
    hb_term = np.stack([Up_Tb, Vb_Tp, Vp_Tb, Qzz_Wb_Tp, Qzz_Wp_Tb, Qw_Wb_Tp], axis=0)
    hb_term = np.where(hb_term > theta, theta, hb_term)
    hb_term = np.where(hb_term < -theta, -theta, hb_term)
    hb_term = np.pad(hb_term, ((0, 0), (0, 0), (HB_PAD, HB_PAD), (0, 0)))

    scaler = MinMaxScaler()
    sst = np.reshape(scaler.fit_transform(np.reshape(sst.values, (-1, h * w))), (-1, h, w))
    hb_term = [np.reshape(scaler.fit_transform(np.reshape(hb_term[i], (-1, h * w))), (-1, h, w)) for i in range(len(hb_term))]

    data = np.concatenate((np.expand_dims(sst[:1620], axis=-1), np.stack(hb_term, axis=-1)), axis=-1)

    return data


def load_godas_field():

    h, w = args['input_size']

    sst = xr.open_dataset(os.path.join(
        args['soda_dir'] + 'total_field/', 'sst.nc'))['sst'].loc['1871-01-16':'2022-12-16'].fillna(0)
    lc = sst.coords['longitude']
    la = sst.coords['latitude']
    sst = sst.loc[dict(longitude=lc[(lc >= LLON) & (lc <= RLON)], latitude=la[(la >= LLAT) & (la <= RLAT)])]

    Up_Tb = xr.open_dataset(os.path.join(
        args['godas_dir'] + 'process_field/anomaly_term', 'Up_Tb.nc'))['Up_Tb'].loc['1980-01-01':'2022-12-01'].fillna(0)

    Vb_Tp = xr.open_dataset(os.path.join(
        args['godas_dir'] + 'process_field/anomaly_term', 'Vb_Tp.nc'))['Vb_Tp'].loc['1980-01-01':'2022-12-01'].fillna(0)

    Vp_Tb = xr.open_dataset(os.path.join(
        args['godas_dir'] + 'process_field/anomaly_term', 'Vp_Tb.nc'))['Vp_Tb'].loc['1980-01-01':'2022-12-01'].fillna(0)

    Qzz_Wb_Tp = xr.open_dataset(os.path.join(
        args['godas_dir'] + 'process_field/anomaly_term', 'Qzz_Wb_Tp.nc'))['Qzz_Wb_Tp'].loc['1980-01-01':'2022-12-01'].fillna(0)
    
    Qzz_Wp_Tb = xr.open_dataset(os.path.join(
        args['godas_dir'] + 'process_field/anomaly_term', 'Qzz_Wp_Tb.nc'))['Qzz_Wp_Tb'].loc['1980-01-01':'2022-12-01'].fillna(0)
    
    Qw_Wb_Tp = xr.open_dataset(os.path.join(
        args['godas_dir'] + 'process_field/anomaly_term', 'Qw_Wb_Tp.nc'))['Qw_Wb_Tp'].loc['1980-01-01':'2022-12-01'].fillna(0)
    
    hb_term = np.stack([Up_Tb, Vb_Tp, Vp_Tb, Qzz_Wb_Tp, Qzz_Wp_Tb, Qw_Wb_Tp], axis=0)
    hb_term = np.where(hb_term > theta, theta, hb_term)
    hb_term = np.where(hb_term < -theta, -theta, hb_term)
    hb_term = np.pad(hb_term, ((0, 0), (0, 0), (HB_PAD, HB_PAD), (0, 0)))

    scaler = MinMaxScaler()
    sst = np.reshape(scaler.fit_transform(np.reshape(sst.values, (-1, h * w))), (-1, h, w))
    hb_term = [np.reshape(scaler.fit_transform(np.reshape(hb_term[i], (-1, h * w))), (-1, h, w)) for i in range(len(hb_term))]

    data = np.concatenate((np.expand_dims(sst[1308:], axis=-1), np.stack(hb_term, axis=-1)), axis=-1)

    return data


# def load_base_data():
#     gfdl_data = load_gfdl_data()
#     soda_data = load_soda_data()
    
#     train_data = gfdl_data
#     val_data = soda_data
#     # print(train_data.shape)
#     # print(val_data.shape)

#     train_dataset = EnsoDataset(train_data)
#     valid_dataset = EnsoDataset(val_data)
#     return train_dataset, valid_dataset


def load_target_data():
    gfdl_data = load_gfdl_data()
    e3sm_data = load_e3sm_data()
    # noresm2_data =  load_noresm2_data()
    cesm_data = load_cesm2_data()
    # cams_data = load_cams_data()
    # cesm_waccm_data = load_cesm2_waccm_data()
    fio_data = load_fio_data()
    soda_data = load_soda_field()
    godas_data = load_godas_field()

    train_data_0 = gfdl_data
    # train_data_1 = cesm_waccm_data
    train_data_2 = fio_data
    train_data_3 = cesm_data
    # train_data_4 = cams_data 
    # train_data_5 = noresm2_data
    train_data_6 = e3sm_data
    train_data_7 = soda_data
    train_data_8 = godas_data[:312]
    val_data = godas_data[360:]
    # print(val_data.shape)

    train_dataset_0 = EnsoDataset(train_data_0)
    # train_dataset_1 = EnsoDataset(train_data_1)
    train_dataset_2 = EnsoDataset(train_data_2)
    train_dataset_3 = EnsoDataset(train_data_3)
    # train_dataset_4 = EnsoDataset(train_data_4)
    # train_dataset_5 = EnsoDataset(train_data_5)
    train_dataset_6 = EnsoDataset(train_data_6)
    train_dataset_7 = EnsoDataset(train_data_7)
    train_dataset_8 = EnsoDataset(train_data_8)

    valid_dataset = EnsoDataset(val_data)
    train_dataset = ConcatDataset([train_dataset_0, train_dataset_2, train_dataset_3, 
                                   train_dataset_6, train_dataset_7, train_dataset_8])
    return train_dataset, valid_dataset
    

class EnsoDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.i_len = args['T_in']
        self.o_len = args['T_out']
        self.lead = args['step']

    def __len__(self):
        return self.data.shape[0] - self.i_len - self.o_len - self.lead + 1

    def __getitem__(self, idx):
        inputs = self.data[idx:idx + self.i_len, ...]
        output = self.data[idx + self.i_len + self.lead:idx + self.i_len + self.lead + self.o_len, ..., 0]
        if self.o_len == 1:
            output = output.squeeze(0)

        return inputs, output


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import time

    start = time.process_time()
    train_set, val_set = load_target_data()

    train_loader = DataLoader(train_set, batch_size=args['batch_size'])
    val_loader = DataLoader(val_set, batch_size=args['batch_size'])
    for step, (sst, sst_target) in enumerate(val_loader):
        # if step%40==0:
        print(step, sst.shape, sst_target.shape)
            
    elapsed = (time.process_time() - start)
    print(elapsed)
    