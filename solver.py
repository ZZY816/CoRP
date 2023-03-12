import torch
from torch.optim import Adam

from loss import IoU_loss
import numpy as np
import cv2
from loader.dataset import get_loader
from os.path import join
from utils import mkdir, write_doc, get_time
from loader.data_loader_for_sal import Data
from loader.data_loader_for_sal import Config
import network
from torch.utils.data import DataLoader
from itertools import cycle
from torch import nn
from torch.nn import init


class Solver(object):
    def __init__(self, backbone):
        self.CoRP = network.CoRP(backbone=backbone).cuda()
        self.backbone = backbone

    def weights_init(self, module):
        if isinstance(module, nn.Conv2d):
            init.normal_(module.weight, 0, 0.01)
            if module.bias is not None:
                init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            init.constant_(module.weight, 1)
            init.constant_(module.bias, 0)

    def train(self, roots, init_epoch, end_epoch, learning_rate, batch_size, weight_decay, ckpt_root, doc_path,
              num_thread, pin,  milestones, sal_root, fix_seed=False):
        # Define Adam optimizer.

        backbone_params = list(map(id, self.CoRP.encoder.parameters()))
        decoder_params = filter(lambda p: id(p) not in backbone_params,
                                self.CoRP.parameters())

        lr_d = 10 if self.backbone == 'vgg16' else 1
        backbone_lr = learning_rate / lr_d

        optimizer = Adam([{'params': decoder_params, 'lr': learning_rate},
                          {'params': self.CoRP.encoder.parameters(), 'lr': backbone_lr}],
                         weight_decay=weight_decay)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
        # Load ".pth" to initialize model.
        if init_epoch != 0:
            # From the existed checkpoint file.
            ckpt = torch.load(join(ckpt_root, 'Weights_{}.pth'.format(init_epoch)))
            self.CoRP.load_state_dict(ckpt['state_dict'])
            optimizer.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])


        # Define training dataloader.
        train_dataloader = get_loader(roots=roots, request=('img', 'gt'), shuffle=True, batch_size=batch_size,
                                      data_aug=True, num_thread=num_thread, pin=pin, fix_seed=fix_seed)
        cfg = Config(mode='train', datapath=sal_root)
        # cfg = Config(mode='train', datapath='./Dataset/COCOSAL')
        All_data = Data(cfg)

        def _init_fn(worker_id):
            np.random.seed(int(666) + worker_id)
        if fix_seed:
            train_sal_dataloader = DataLoader(All_data, collate_fn=All_data.collate, batch_size=8, shuffle=True,
                                              num_workers=8, worker_init_fn=_init_fn)
        else:
            train_sal_dataloader = DataLoader(All_data, collate_fn=All_data.collate, batch_size=8, shuffle=True,
                                              num_workers=8)
        
        # Train.
        self.CoRP.train()
        for epoch in range(init_epoch + 1, end_epoch):
            start_time = get_time()
            loss_sum = 0.0
            count = 0
            #for i in range(len(train_dataloader)):
            for i, data in enumerate(zip(train_dataloader, cycle(train_sal_dataloader))):
                self.CoRP.zero_grad()

                # Obtain a batch of data.
                img, gt  = data[0]['img'], data[0]['gt']
                img, gt = img.cuda(), gt.cuda()

                sal_img, sal_gt = data[1][0].float().cuda(), data[1][1].float().cuda()

                if len(img) == 1:
                    # Skip this iteration when training batchsize is 1 due to Batch Normalization. 
                    continue
                
                # Forward.
                preds_list, preds_sal = self.CoRP(image_group=img,
                                        sal=sal_img,

                                        is_training=True,
                                                   gt=gt)
                
                # Compute IoU loss.
                loss = 0.9 * IoU_loss(preds_list, gt) + 0.1 * IoU_loss(preds_sal, sal_gt)

                # Backward.
                loss.backward()
                optimizer.step()
                loss_sum = loss_sum + loss.detach().item()
                count += 1
                loss_m = loss_sum/count

                if count % 20 == 0:
                    print('epoch:', epoch, 'lr:', optimizer.state_dict()['param_groups'][0]['lr'], 'loss_mean:', loss_m)
            scheduler.step()
            # Save the checkpoint file (".pth") after each epoch.
            mkdir(ckpt_root)
            torch.save({'optimizer': optimizer.state_dict(),
                        'state_dict': self.CoRP.state_dict()}, join(ckpt_root, 'Weights_{}.pth'.format(epoch)))
            
            # Compute average loss over the training dataset approximately.
            loss_mean = loss_sum / len(train_dataloader)
            end_time = get_time()

            # Record training information (".txt").
            content = 'CkptIndex={}:    TrainLoss={}    LR={}    Time={}\n'.format(epoch, loss_mean, learning_rate,
                                                                                   end_time - start_time)
            write_doc(doc_path, content)
    
    def test(self, roots, ckpt_path, pred_root, num_thread, batch_size, original_size, pin):
        with torch.no_grad():            
            # Load the specified checkpoint file(".pth").
            state_dict = torch.load(ckpt_path)['state_dict']
            self.CoRP.load_state_dict(state_dict)
            self.CoRP.eval()
            
            # Get names of the test datasets.
            datasets = roots.keys()

            # Test CoRP on each dataset.
            for dataset in datasets:
                # Define test dataloader for the current test dataset.
                test_dataloader = get_loader(roots=roots[dataset], 
                                             request=('img',  'file_name', 'group_name', 'size'),
                                             shuffle=False,
                                             data_aug=False, 
                                             num_thread=num_thread, 
                                             batch_size=batch_size, 
                                             pin=pin)

                # Create a folder for the current test dataset for saving predictions.
                mkdir(pred_root)
                cur_dataset_pred_root = join(pred_root, dataset)
                mkdir(cur_dataset_pred_root)

                for data_batch in test_dataloader:
                    # Obtain a batch of data.
                    img= data_batch['img'].cuda()

                    # Forward.
                    preds = self.CoRP(image_group=img,
                                       is_training=False)
                    
                    # Create a folder for the current batch according to its "group_name" for saving predictions.
                    group_name = data_batch['group_name'][0]
                    cur_group_pred_root = join(cur_dataset_pred_root, group_name)
                    mkdir(cur_group_pred_root)

                    # preds.shape: [N, 1, H, W]->[N, H, W, 1]
                    preds = preds.permute(0, 2, 3, 1).cpu().numpy()

                    # Make paths where predictions will be saved.
                    pred_paths = list(map(lambda file_name: join(cur_group_pred_root, file_name + '.png'), data_batch['file_name']))

                    # For each prediction:
                    for i, pred_path in enumerate(pred_paths):
                        # Resize the prediction to the original size when "original_size == True".
                        H, W = data_batch['size'][0][i].item(), data_batch['size'][1][i].item()

                        pred = cv2.resize(preds[i], (W, H)) if original_size else preds[i]

                        # Save the prediction.

                        cv2.imwrite(pred_path, np.array(pred * 255))
