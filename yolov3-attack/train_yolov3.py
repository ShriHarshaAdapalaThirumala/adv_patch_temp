import argparse
import os
import torch
from tqdm import tqdm
import pdb

from utils import setup_seed
from dataset import Kitti, get_dataloader
from model import PointPillars, Darknet
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable

# Fix GPU use
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = str(2)


def test(args):
    setup_seed()
    train_dataset = Kitti(data_root=args.data_root,
                          split='train_rdsq')
    val_dataset = Kitti(data_root=args.data_root,
                        split='val_rdsq')
    train_dataloader = get_dataloader(dataset=train_dataset, 
                                      batch_size=16, 
                                      num_workers=4,
                                      shuffle=False)
    val_dataloader = get_dataloader(dataset=val_dataset, 
                                    batch_size=16, 
                                    num_workers=4,
                                    shuffle=False)

    for epoch in range(6):
        for i, data_dict in enumerate(train_dataloader):
            #batched_img = data_dict['batched_img']
            batched_gt_bboxes_img = data_dict['batched_gt_bboxes_img']
            print(batched_gt_bboxes_img[0][0])
            #batched_labels = data_dict['batched_labels']
            #imgs = torch.stack(batched_img,dim=0)
            break

def main(args):
    setup_seed()
    train_dataset = Kitti(data_root=args.data_root,
                          split='train_rdsq')
    val_dataset = Kitti(data_root=args.data_root,
                        split='val_rdsq')
    train_dataloader = get_dataloader(dataset=train_dataset, 
                                      batch_size=args.batch_size, 
                                      num_workers=args.num_workers,
                                      shuffle=True)
    val_dataloader = get_dataloader(dataset=val_dataset, 
                                    batch_size=args.batch_size, 
                                    num_workers=args.num_workers,
                                    shuffle=False)

    model_config_path = args.model_config_path
    weights_path = "yolo_pre/darknet53.conv.74"
    if not args.no_cuda:
        model = Darknet(model_config_path).cuda()
    else:
        model = Darknet(model_config_path)
    model.load_weights(weights_path) #load pretrained darknet-53 on imagenet
    model.train()

    accumulated_batches = 4
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

    saved_logs_path = args.saved_path
    os.makedirs(saved_logs_path, exist_ok=True)
    writer = SummaryWriter(saved_logs_path)
    saved_ckpt_path = args.saved_path

    for epoch in range(args.max_epoch):
        if True:        #freeze pretrained darknet
            if epoch < 20:
                for i, (name, p) in enumerate(model.named_parameters()):
                    if int(name.split('.')[1]) < 75:  # if layer < 75
                        p.requires_grad = False
            elif epoch >= 20:
                for i, (name, p) in enumerate(model.named_parameters()):
                    if int(name.split('.')[1]) < 75:  # if layer < 75
                        p.requires_grad = True
        
        optimizer.zero_grad() 
        for i, data_dict in enumerate(tqdm(train_dataloader)):
            if not args.no_cuda:
                # move the tensors to the cuda
                for key in data_dict:
                    for j, item in enumerate(data_dict[key]):
                        if torch.is_tensor(item):
                            data_dict[key][j] = data_dict[key][j].cuda()

            batched_img = data_dict['batched_img']
            batched_gt_bboxes_img = data_dict['batched_gt_bboxes_img']
            batched_labels = data_dict['batched_labels']
            imgs = torch.stack(batched_img,dim=0)
            #print(imgs.shape)
            targets = []
            max_objects = 50
            for b in range(len(batched_gt_bboxes_img)):
                labels = torch.cat([batched_labels[b].unsqueeze(-1),batched_gt_bboxes_img[b]],dim=-1)
                filled_labels = torch.zeros((max_objects, 5)).to(labels)
                if labels is not None:
                    filled_labels[range(len(labels))[:max_objects]] = labels[:max_objects] 
                targets.append(filled_labels)
            targets = torch.stack(targets,dim=0)
            #print(targets.shape)

            imgs = Variable(imgs)
            targets = Variable(targets, requires_grad=False)
            loss = model(imgs, targets)

            loss.backward()
            #optimizer.step()
                    # accumulate gradient for x batches before optimizing
            if ((i + 1) % accumulated_batches == 0) or (i == len(train_dataloader) - 1):
                optimizer.step()
                optimizer.zero_grad()
            
            model.seen += imgs.size(0)
            #break
        if (epoch + 1) % args.ckpt_freq_epoch == 0:
            model.save_weights("%s/%d.weights" % (saved_ckpt_path, epoch))
        '''
        if epoch % 2 == 0:
            continue
        model.eval()
        with torch.no_grad():
            for i, data_dict in enumerate(tqdm(val_dataloader)):
                if not args.no_cuda:
                    # move the tensors to the cuda
                    for key in data_dict:
                        for j, item in enumerate(data_dict[key]):
                            if torch.is_tensor(item):
                                data_dict[key][j] = data_dict[key][j].cuda()
                
                batched_img = data_dict['batched_img']
                batched_gt_bboxes_img = data_dict['batched_gt_bboxes_img']
                batched_labels = data_dict['batched_labels']
                imgs = torch.stack(batched_img,dim=0)
                
                imgs = Variable(imgs)
                outputs = model(imgs)
            break 
        model.train()
        '''
        #break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--data_root', default='/media/14T/yi/KITTI', 
                        help='your data root for kitti')
    parser.add_argument('--saved_path', default='logs/yolov3_newresize_rdsq_fixbug')
    parser.add_argument('--model_config_path', default='yolo_pre/yolov3-kitti.cfg')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--nclasses', type=int, default=3)
    parser.add_argument('--init_lr', type=float, default=0.00025)
    parser.add_argument('--max_epoch', type=int, default=140)
    parser.add_argument('--log_freq', type=int, default=8)
    parser.add_argument('--ckpt_freq_epoch', type=int, default=20)
    parser.add_argument('--no_cuda', action='store_true',
                        help='whether to use cuda')
    args = parser.parse_args()

    main(args)
    #test(args)
