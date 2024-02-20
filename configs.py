import torch
import os

args = {

    'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
    # 'device': 'cpu',
    'model_name': 'stmafno',

    # 'double_skip': True,
    # 'checkpoint_activations': False,
    # 'patch_size': 4,
    # 'hidden_size': 384,
    # 'num_layers': 6, 
    # 'mlp_ratio': 4,

    'ntrain': 110*12,
    'nval': 30*12,
    'train_sample': 321*4,
    'val_sample': 81*4,

    'batch_size': 8,

    'epochs': 1000,
    'learning_rate': 0.0001,
    'scheduler_step': 20,
    'scheduler_gamma': 0.9,

    'd_size': 384,
    'warmup':2000, 

    'T_in': 6,
    'T_out': 18,
    'step': 0,

    'input_size': (60, 160),
    'patch_size': (4, 4),

    'lat_nino_relative': (25, 35),
    'lon_nino_relative': (70, 120),

    # 'data_file': ['sst.nc', 't300.nc'],
    # 'var_name': ['sst', 'temp'],
    # 'Field': 'total',

    'pretrain': False,
    'ckpt': '',
    
    'data_dir': [i for i in ['/home/dl/Desktop/vit/data/TEST/', r'D:/Python/vit-pytorch/data/Transfer/']
                 if os.path.exists(i)][0],
    'soda_dir': [i for i in ['/home/dl/Desktop/vit/data/Transfer/', r'D:/Python/vit-pytorch/data/Transfer/']
                 if os.path.exists(i)][0],
    'godas_dir': [i for i in ['/home/dl/Desktop/vit/data/TEST/', r'D:/Python/vit-pytorch/data/Transfer/']
                 if os.path.exists(i)][0],
    'gfdl_dir': [i for i in ['/media/dl/Skye_Cui/vit/data/GFDL/', r'D:/Python/vit-pytorch/data/Transfer/']
                 if os.path.exists(i)][0],      
    'E3SM_dir': [i for i in ['/media/dl/Elements/E3SM-1-1/', r'D:/Python/vit-pytorch/GFDL_data/']
                 if os.path.exists(i)][0],
    'NorESM2_dir': [i for i in ['/media/dl/Elements/NorESM2-MM/', r'D:/Python/vit-pytorch/GFDL_data/']
                 if os.path.exists(i)][0],
    'CESM2_dir': [i for i in ['/media/dl/Elements/CESM2/', r'D:/Python/vit-pytorch/GFDL_data/']
                 if os.path.exists(i)][0],
    # 'CAMS_dir': [i for i in ['/media/dl/Elements/CAMS/', r'D:/Python/vit-pytorch/GFDL_data/']
    #              if os.path.exists(i)][0],
    'CESM2_WACCM_dir': [i for i in ['/media/dl/Elements/CESM2-WACCM/', r'D:/Python/vit-pytorch/GFDL_data/']
                 if os.path.exists(i)][0],
    'FIO_dir': [i for i in ['/media/dl/Elements/FIO/', r'D:/Python/vit-pytorch/GFDL_data/']
                 if os.path.exists(i)][0],
    'ACCESS_dir': [i for i in ['/media/dl/Elements/ACCESS/', r'D:/Python/vit-pytorch/GFDL_data/']
                 if os.path.exists(i)][0],
    'save_dir': [i for i in ['/home/dl/Desktop/vit/ckpt/', r'D:/Python/vit-pytorch/ckpt/']
                 if os.path.exists(i)][0],
    'delivery_model': '/home/dl/Desktop/vit/ckpt/99/stmafno.pth',

}

