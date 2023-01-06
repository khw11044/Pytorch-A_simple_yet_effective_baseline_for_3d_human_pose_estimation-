# 학습 안됨


import torch
import torch.nn
import torch.optim
import numpy as np
from torch.utils import data
from utils.data import H36MDataset
import torch.optim as optim
import networks.model as model
from utils.print_losses import print_losses
from types import SimpleNamespace
from tqdm import tqdm
from utils.functions import *
import datetime

#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

config = SimpleNamespace()

config.learning_rate = 0.0001
config.BATCH_SIZE = 512     
config.N_epochs = 100

# weights for the different losses
config.weight_rep = 1
config.weight_view = 1
config.weight_camera = 0.1

data_folder = './data/'

config.datafile = data_folder + 'alphapose_h36m.pickle'

config.save_model_name =  'models/model_lifter.pt'
config.checkpoint = 'models/chekcpoints'

def loss_mpjpe(gt_p3d, pred_p3d):
    # the weighted reprojection loss as defined in Equation 5

    # normalize by scale
    scale_gt_p3d = torch.sqrt(gt_p3d[:, 0:48].square().sum(axis=1, keepdim=True) / 48)
    gt_p3d_scaled = gt_p3d[:, 0:48]/scale_gt_p3d

    # only the u,v coordinates are used and depth is ignored
    # this is a simple weak perspective projection
    scale_pred_p3d = torch.sqrt(pred_p3d[:, 0:48].square().sum(axis=1, keepdim=True) / 48)
    pred_p3d_scaled = pred_p3d[:, 0:48]/scale_pred_p3d
    # loss = ((p2d_scaled - p3d_scaled).abs().reshape(-1, 2, 16).sum(axis=1) * confs).sum() / (p2d_scaled.shape[0] * p2d_scaled.shape[1])

    # np.mean( np.sqrt( np.sum( (gt_p3d_scaled - pred_p3d_scaled)**2, axis=2) ) )

    # loss = torch.mean( torch.sqrt(torch.square(gt_p3d_scaled - pred_p3d_scaled).sum(axis=2)) )

    # 1 순위
    loss = ((gt_p3d_scaled - pred_p3d_scaled).square().reshape(-1, 3, 16).sum(axis=1) ).mean()
    # 2 순위
    # loss = ((gt_p3d_scaled - pred_p3d_scaled).square().reshape(-1, 3, 16).sum(axis=1) ).sum() / (gt_p3d_scaled.shape[0] * gt_p3d_scaled.shape[1])
    
    return loss


def train(config, len_dataset, train_loader, model_skel_morph, model, optimizer, epoch, losses,losses_mean):
    return step('train', config, len_dataset, train_loader, model_skel_morph, model, optimizer, epoch, losses,losses_mean)


def test(config, len_dataset, test_loader, model_skel_morph, model):
    with torch.no_grad():
        return step('test', config, len_dataset, test_loader, model_skel_morph, model)


def step(split, config, len_dataset, dataLoader, model_skel_morph, model, optimizer=None, epoch=None, losses=None,losses_mean=None):
    if split == 'train':
        model.train()
    else:
        model.eval()
        # mpjpes = []
        # pmpjpes = []
        # pcks = []

    for iter, sample in enumerate(tqdm(dataLoader, 0)):

        # not the most elegant way to extract the dictionary
        poses_2d = {key:sample[key] for key in all_cams}
        poses_2d_gt = {key:sample[key+'_2dgt'] for key in all_cams}
        poses_3d_gt = {key:sample[key+'_3dgt'] for key in all_cams} 

        inp_poses = torch.zeros((poses_2d['cam0'].shape[0] * len(all_cams), 32)).cuda()
        # inp_confidences = torch.zeros((poses_2d['cam0'].shape[0] * len(all_cams), 16)).cuda()

        gt_2dpose = torch.zeros((poses_2d_gt['cam0'].shape[0] * len(all_cams), 32)).cuda()
        gt_3dpose = torch.zeros((poses_3d_gt['cam0'].shape[0] * len(all_cams), 16,3)).cuda()

        # poses_2d is a dictionary. It needs to be reshaped to be propagated through the model.
        cnt = 0
        for b in range(poses_2d['cam0'].shape[0]):
            for c_idx, cam in enumerate(poses_2d):
                inp_poses[cnt] = poses_2d[cam][b]
                # inp_confidences[cnt] = sample['confidences'][cam_names[c_idx]][b]
                gt_2dpose[cnt] = poses_2d_gt[cam][b]
                gt_3dpose[cnt] = poses_3d_gt[cam][b]
                    
                cnt += 1

        # morph the poses using the skeleton morphing network
        # inp_poses = model_skel_morph(inp_poses)

        # predict 3d poses
        pred = model(gt_2dpose)
        
# ----------------------------------------------------------------------------------------

        if split == 'test':
            pred_3dpose = pred.reshape(-1,3,16).permute(0,2,1)
            pred_3dpose = pred_3dpose.cpu().detach().numpy()
            poses_3dgt = gt_3dpose.cpu().detach().numpy()

            poses_3dgt, pose_norm, _ = regular_normalized3d(poses_3dgt) 
            pred_3dpose, _, _ = regular_normalized3d(pred_3dpose) 

            poses_3dgt = poses_3dgt * pose_norm
            pred_3dpose = pred_3dpose * pose_norm
            

            mpjpe = np.mean( np.sqrt( np.sum( (poses_3dgt - pred_3dpose)**2, axis=2) ) )
            
            diff = np.sqrt(np.square(poses_3dgt - pred_3dpose).sum(axis=2))
            pck = 100 * len(np.where(diff < 150)[0]) / (diff.shape[0] * diff.shape[1])

            pred_3dpose = pred_3dpose.reshape(-1,4,16,3)
            poses_3dgt = poses_3dgt.reshape(-1,4,16,3)

            pmpjpes = []
            for k in range(len(poses_3dgt)):
                pmpjpe = []
                for v in range(len(all_cams)):
                    mpjpes = np.mean(np.sqrt(np.sum((pred_3dpose[k][v] - poses_3dgt[k][v])**2, axis=1)))
                    pmpjpe.append(mpjpes)
                pmpjpe = min(pmpjpe)
                pmpjpes.append(pmpjpe)
            pmpjpe = np.mean(pmpjpes)

            print('mpjpe,  pmpjpe,  pck')
            print('{:.2f} / {:.2f} / {:.2f}'.format( mpjpe, pmpjpe, pck ))
            now = datetime.datetime.now()
            nowTime = now.strftime('%H:%M:%S')   # 12:11:32

            return mpjpe, pmpjpe, pck, nowTime


        elif split == 'train':
            gt = gt_3dpose.permute(0,2,1).reshape(-1,48)
            # mpjpe loss
            losses.loss = loss_mpjpe(gt, pred)

            optimizer.zero_grad()
            losses.loss.backward()
            optimizer.step()

            for key, value in losses.__dict__.items():
                if key not in losses_mean.__dict__.keys():
                    losses_mean.__dict__[key] = []

                losses_mean.__dict__[key].append(value.item())

            # print progress every 100 iterations
            if not iter % 1000:
                # print the losses to the console
                # print_losses(epoch, iter, len(my_dataset) / config.BATCH_SIZE, losses_mean.__dict__, print_keys=not(iter % 1000))
                print_losses(config.N_epochs, epoch, iter, len_dataset / config.BATCH_SIZE, losses_mean.__dict__, print_keys=not(iter % 1000))
                

                # this line is important for logging!
                losses_mean = SimpleNamespace()


        # scheduler.step()



if __name__ == '__main__':

    print('training start')

    # loading the H36M dataset
    train_dataset = H36MDataset(config.datafile, normalize_2d=True, subjects=[5,6,7,8])
    train_loader = data.DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)

    test_dataset = H36MDataset(config.datafile, normalize_2d=True, subjects=[9,11])
    test_loader = data.DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    # load the skeleton morphing model as defined in Section 4.2
    # for another joint detector it needs to be retrained -> train_skeleton_morph.py
    model_skel_morph = torch.load('models/model_skeleton_morph_S1_gh.pt')
    model_skel_morph.eval()

    # loading the lifting network
    model = model.Lifter().cuda()

    params = list(model.parameters())

    optimizer = optim.Adam(params, lr=config.learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)

    losses = SimpleNamespace()
    losses_mean = SimpleNamespace()

    cam_names = ['54138969', '55011271', '58860488', '60457274']
    all_cams = ['cam0', 'cam1', 'cam2', 'cam3']

    mpjpes = []
    pmpjpes = []
    pcks = []
    nowTimes = []

    for epoch in range(config.N_epochs):

        train(config,len(train_dataset),train_loader,model_skel_morph,model,optimizer,epoch, losses,losses_mean)

        print('test')

        # mpjpe, pmpjpe, pck = test(config,len(train_loader),train_loader,model_skel_morph,model)
        mpjpe, pmpjpe, pck, nowTime = test(config,len(test_dataset),test_loader,model_skel_morph,model)

        mpjpes.append(mpjpe)
        pmpjpes.append(pmpjpe)
        pcks.append(pck)
        nowTimes.append(nowTime)

        # save the new trained model every epoch
        torch.save(model, config.save_model_name)

        scheduler.step()

    print('mpjpes',mpjpes)

    print()
    print('---'*10)
    print()

    print('pmpjpes',pmpjpes)

    print()
    print('---'*10)
    print()

    print('pcks',pcks)

    print()
    print('---'*10)
    print()

    print('nowTimes',nowTimes)

    print(config.save_model_name)
    print('done')