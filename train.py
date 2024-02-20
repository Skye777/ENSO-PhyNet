import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.realpath(".."))
sys.path.append('/home/dl/Desktop/vit/')

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from FNO.utilities3 import *
from timeit import default_timer
from multivar.configs import args
# from FNO.Adam import Adam
from multivar.data_loader import load_target_data
from torch.utils.data import DataLoader
from multivar.stmafno_norm import *
from copy import deepcopy
import re


torch.manual_seed(42) #3407
# np.random.seed(0)

device = args['device']
model = AFNONet(img_size=args['input_size'], patch_size=args['patch_size'], in_chans=args['T_in'], out_chans=args['T_out']).to(device)

print(count_params(model))
myloss = LpLoss(size_average=False)

ninoweight = torch.from_numpy(
            np.array([1.5] * 4 + [2] * 7 + [3] * 7 + [4] * 6)
            * np.log(np.arange(24) + 1)
        ).to(device)
ninoweight = ninoweight[:args['T_out']]
def calscore(y_pred, y_true):
    # compute Nino score
    with torch.no_grad():
        pred = y_pred - y_pred.mean(dim=0, keepdim=True)
        true = y_true - y_true.mean(dim=0, keepdim=True)
        cor = (pred * true).sum(dim=0) / (
            torch.sqrt(torch.sum(pred ** 2, dim=0) * torch.sum(true ** 2, dim=0))
            + 1e-6
        )
        acc = (ninoweight * cor).sum()
        rmse = torch.mean((y_pred - y_true) ** 2, dim=0).sqrt().sum()
        sc = 2 / 3.0 * acc - rmse
    return sc.item()

# else:
optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'])

# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = 10, T_mult = 5, eta_min = 1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args['scheduler_step'], gamma=args['scheduler_gamma'])

train_set, val_set = load_target_data()
train_loader = DataLoader(train_set, batch_size=args["batch_size"])
val_loader = DataLoader(val_set, batch_size=args["batch_size"])

batch_size = args['batch_size']
ntrain = args['train_sample']
nval = args['val_sample']
best_loss = -float('inf')
ckpt = args['ckpt']

if args["pretrain"] and os.path.exists(args['save_dir']):
    model.load_state_dict(
        torch.load(
            os.path.join(args['save_dir'], args["ckpt"]), map_location=device
        )
    )
    print("load model from:", args['save_dir'])
    ckpt = args['model_name'] + f'.pth'

if ckpt == '':
    # ckpt = args['model_name'] + f'_{0}.pth'
    ckpt = args['model_name'] + f'.pth'

for ep in range(args['epochs']):
    # total_epoch = int(re.findall("\d+", ckpt)[0])
    # ckpt = ckpt.replace(f'_{total_epoch}', f'_{total_epoch + 1}')
    print("==========" * 8)
    print("\n-->epoch: {0}".format(ep))
    # ---------train
    model.train()
    t1 = default_timer()
    train_mse = 0
    train_l2 = 0
    # train_mae = 0
    for j, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        im, _, _,  = model(x)

        nino_y = y[
            :, 
            :, 
            args['lat_nino_relative'][0]: args['lat_nino_relative'][1], 
            args['lon_nino_relative'][0]: args['lon_nino_relative'][1]+30,
        ].mean(dim=[2, 3])
        nino_out = im[
            :, 
            :, 
            args['lat_nino_relative'][0]: args['lat_nino_relative'][1], 
            args['lon_nino_relative'][0]: args['lon_nino_relative'][1]+30,
        ].mean(dim=[2, 3])
        mse = F.mse_loss(im, y, reduction='mean')
        # mae = F.l1_loss(im, y, reduction='mean')
        l2 = myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))
        loss_nino = nino_loss(nino_out, nino_y)
        
        combine_loss = combien_loss(loss_nino, l2)
        combine_loss.backward()

        optimizer.step()
        train_mse += mse.item()
        train_l2 += l2.item()
        # train_mae += mae.item()

        # -----------Intensive verification
        if (ep + 1 >= 11) and (j + 1) % 100 == 0:
            model.eval()
            test_mse = 0
            test_l2 = 0
            nino_true = []
            nino_pred = []
            # test_mae = 0
            with torch.no_grad():
                for xx, yy in val_loader:
                    xx = xx.to(device)
                    yy = yy.to(device)

                    im, _, _, = model(xx)

                    nino_yy = yy[
                        :, 
                        :, 
                        args['lat_nino_relative'][0]: args['lat_nino_relative'][1], 
                        args['lon_nino_relative'][0]: args['lon_nino_relative'][1]+30,
                    ].mean(dim=[2, 3])
                    nino_out = im[
                        :, 
                        :, 
                        args['lat_nino_relative'][0]: args['lat_nino_relative'][1], 
                        args['lon_nino_relative'][0]: args['lon_nino_relative'][1]+30,
                    ].mean(dim=[2, 3])
                    nino_true.append(nino_yy)
                    nino_pred.append(nino_out)
                    mse = F.mse_loss(im, yy, reduction='mean')
                    # mae = F.l1_loss(im, yy, reduction='mean')
                    test_l2 += myloss(im.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()
                    test_mse += mse.item()
                    # test_mae += mae.item()
                nino_true = torch.cat(nino_true, dim=0)
                nino_pred = torch.cat(nino_pred, dim=0)
                ninosc = calscore(nino_pred, nino_true)
            test_mse /= len(val_loader)
            test_l2 /= len(train_loader)
            # train_mae /= len(train_loader)
            # test_mae /= len(val_loader)

            print(
                "-->Evaluation... \nscore:{:.3f} \ntest_mse:{:.5f} ".format(
                ninosc, test_mse
                )
            )

            if ninosc > best_loss:
                best_loss = ninosc
                best_model = deepcopy(model.state_dict())
                torch.save(best_model, os.path.join(args['save_dir'],  ckpt))
                print('Model saved successfully:{}'.format(os.path.join(args['save_dir'], ckpt)))

    scheduler.step()

    # ----------after one epoch-----------
    model.eval()
    test_mse = 0
    test_l2 = 0
    nino_true = []
    nino_pred = []
    # test_mae = 0
    with torch.no_grad():
        for xx, yy in val_loader:
            xx = xx.to(device)
            yy = yy.to(device)

            im, _, _, = model(xx)

            nino_yy = yy[
                :, 
                :, 
                args['lat_nino_relative'][0]: args['lat_nino_relative'][1], 
                args['lon_nino_relative'][0]: args['lon_nino_relative'][1]+30,
            ].mean(dim=[2, 3])
            nino_out = im[
                :, 
                :, 
                args['lat_nino_relative'][0]: args['lat_nino_relative'][1], 
                args['lon_nino_relative'][0]: args['lon_nino_relative'][1]+30,
            ].mean(dim=[2, 3])
            nino_true.append(nino_yy)
            nino_pred.append(nino_out)
            mse = F.mse_loss(im, yy, reduction='mean')
            # mae = F.l1_loss(im, yy, reduction='mean')
            test_l2 += myloss(im.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()
            test_mse += mse.item()
            # test_mae += mae.item()
        nino_true = torch.cat(nino_true, dim=0)
        nino_pred = torch.cat(nino_pred, dim=0)
        ninosc = calscore(nino_pred, nino_true)
    train_mse /= len(train_loader)
    test_mse /= len(val_loader)
    train_l2 /= len(train_loader)
    test_l2 /= len(train_loader)
    # train_mae /= len(train_loader)
    # test_mae /= len(val_loader)

    t2 = default_timer()
    print(
        "\n-->epoch{} end... \nduration:{:.1f} \nscore:{:.3f} \ntrain_mse:{:.5f} \ntest_mse:{:.5f}".format(
                    ep, t2-t1, ninosc, train_mse, test_mse
                )
    )

    if ninosc > best_loss:
        best_loss = ninosc
        best_model = deepcopy(model.state_dict())
        torch.save(best_model, os.path.join(args['save_dir'],  ckpt))
        print('Model saved successfully:{}'.format(os.path.join(args['save_dir'], ckpt)))


