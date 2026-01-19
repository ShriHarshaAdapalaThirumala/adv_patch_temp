import argparse
import os
import torch
from tqdm import tqdm
import pdb
import numpy as np
import cv2

from utils import setup_seed,iou_img_bbox
from dataset import Kitti, get_dataloader
from model import PointPillars, Darknet
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import torch.nn.functional as F

# Fix GPU use
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

def non_max_suppression(prediction, num_classes, conf_thres=0.8, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        conf_mask = (image_pred[:, 4] >= conf_thres).squeeze()
        image_pred = image_pred[conf_mask]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5 : 5 + num_classes], 1, keepdim=True)
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)
        # Iterate through all predicted classes
        unique_labels = detections[:, -1].cpu().unique()
        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()
        for c in unique_labels:
            # Get the detections with the particular class
            detections_class = detections[detections[:, -1] == c]
            # Sort the detections by maximum objectness confidence
            _, conf_sort_index = torch.sort(detections_class[:, 4], descending=True)
            detections_class = detections_class[conf_sort_index]
            # Perform non-maximum suppression
            max_detections = []
            while detections_class.size(0):
                # Get detection with highest confidence and save as max detection
                max_detections.append(detections_class[0].unsqueeze(0))
                # Stop if we're at the last detection
                if len(detections_class) == 1:
                    break
                # Get the IOUs for all boxes with lower confidence
                ious = iou_img_bbox(max_detections[-1], detections_class[1:])
                # Remove detections with IoU >= NMS threshold
                detections_class = detections_class[1:][ious < nms_thres]

            max_detections = torch.cat(max_detections).data
            # Add max detections to outputs
            output[image_i] = (
                max_detections if output[image_i] is None else torch.cat((output[image_i], max_detections))
            )

    return output

def get_adv_loss(prediction,gt_bbox_img):
    box_corner = prediction.new(prediction.shape)
    box_corner[:, 0] = prediction[:, 0] - prediction[:, 2] / 2
    box_corner[:, 1] = prediction[:, 1] - prediction[:, 3] / 2
    box_corner[:, 2] = prediction[:, 0] + prediction[:, 2] / 2
    box_corner[:, 3] = prediction[:, 1] + prediction[:, 3] / 2
    prediction[:, :4] = box_corner[:, :4]    #(x1, y1, x2, y2, object_conf, conf_each_class...)
    #we only care about object confidence and ignore the class label
    ious = iou_img_bbox(gt_bbox_img,prediction[:,:4]) #output 1-dim array, each value is the iou of each prediction
    mask = ious>0
    if torch.sum(mask)>0:
        adv_loss,idx = torch.max(prediction[mask][:,4],dim=0)
        out_bbox = prediction[mask][idx:idx+1,:4]
    else:
        adv_loss = 0.0
        out_bbox = [[]]
    return adv_loss,out_bbox

def adv_img_hide(gt_bbox,N_iter,padded_h, padded_w):
    #input: gt_bbox, bbox of area for changing pixels
    bbox_x1,bbox_y1,bbox_x2,bbox_y2 = [int(x) for x in gt_bbox.tolist()]
    center_xs = torch.randint(bbox_x1,bbox_x2,size=(N_iter,))
    center_ys = torch.randint(bbox_y1,bbox_y2,size=(N_iter,))
    size_ws = torch.randint(6,12,size=(N_iter,))
    size_hs = torch.randint(6,12,size=(N_iter,))
    patches_x1,patches_y1,patches_x2,patches_y2 = center_xs-size_ws//2,center_ys-size_hs//2,center_xs+size_ws//2,center_ys+size_hs//2

    adv_imgs,adv_masks = [], []
    for iter_ in range(N_iter):
        patch_w,patch_h = (patches_x2-patches_x1)[iter_],(patches_y2-patches_y1)[iter_]
        x1,y1,x2,y2 = patches_x1[iter_],patches_y1[iter_],patches_x2[iter_],patches_y2[iter_]
        #patch_w,patch_h = (bbox_x2-bbox_x1),(bbox_y2-bbox_y1)  #assume patch cover the whole bbox
        #x1,y1,x2,y2 = bbox_x1,bbox_y1,bbox_x2,bbox_y2        
        pad_top,pad_left,pad_bottom,pad_right = y1,x1,padded_h-y2,padded_w-x2 

        mask = torch.ones((3,patch_h,patch_w))
        adv_patch = torch.rand((3,patch_h,patch_w))*1.0 #random patch
        
        pad1 = (0, 0, pad_top, pad_bottom, 0, 0)
        pad2 = (pad_left, pad_right, 0, 0, 0, 0)
        adv_img = F.pad(adv_patch,pad1,'constant',value=0)
        adv_img = F.pad(adv_img,pad2,'constant',value=0)
        adv_mask = F.pad(mask,pad1,'constant',value=0)
        adv_mask = F.pad(adv_mask,pad2,'constant',value=0)
        adv_imgs.append(adv_img)
        adv_masks.append(adv_mask)
    return adv_imgs,adv_masks

def main(args):
    #setup_seed()
    val_dataset = Kitti(data_root=args.data_root,
                        split='val')

    model_config_path = args.model_config_path
    ckpt = args.ckpt
    if not args.no_cuda:
        model = Darknet(model_config_path).cuda()
    else:
        model = Darknet(model_config_path)
    model.load_weights(ckpt) #load pretrained darknet-53 on imagenet

    saved_path = args.saved_path
    os.makedirs(saved_path, exist_ok=True)

    frame_id = 9
    obj_id = 4
    data_dict = val_dataset[frame_id]
    img = torch.from_numpy(data_dict['img'])
    gt_bboxes_img_ = torch.from_numpy(data_dict['gt_bboxes_img'])
    labels = torch.from_numpy(data_dict['gt_labels'])
    _, padded_h, padded_w = img.shape
    gt_bboxes_img = torch.zeros_like(gt_bboxes_img_)
    gt_bboxes_img[:, 0] = (gt_bboxes_img_[:,0] - gt_bboxes_img_[:,2] / 2)*padded_w
    gt_bboxes_img[:, 1] = (gt_bboxes_img_[:,1] - gt_bboxes_img_[:,3] / 2)*padded_h
    gt_bboxes_img[:, 2] = (gt_bboxes_img_[:,0] + gt_bboxes_img_[:,2] / 2)*padded_w
    gt_bboxes_img[:, 3] = (gt_bboxes_img_[:,1] + gt_bboxes_img_[:,3] / 2)*padded_h

    N_iter = 1000
    adv_imgs,adv_masks = adv_img_hide(gt_bboxes_img[obj_id],N_iter,padded_h,padded_w)
    cv2.imwrite(os.path.join(saved_path, 'patch.png'),adv_imgs[0].permute(1,2,0).numpy()*255)

    model.eval()
    with torch.no_grad():
        loss_list = []
        for iter_ in range(N_iter):
            img_a = img*(1-adv_masks[iter_])+adv_imgs[iter_]*adv_masks[iter_]
            if not args.no_cuda:
                batched_img = [img_a.float().cuda(),]
            else:
                batched_img = [img_a.float(),]
            imgs = torch.stack(batched_img,dim=0)
            imgs = Variable(imgs)
            outputs = model(imgs) #B,N_,5+num_class {x1, y1, x2, y2, object_conf, conf_each_class...}
            #outputs = non_max_suppression(outputs,num_classes=80) #B,N,7 {x1, y1, x2, y2, object_conf, class_score, class_pred}
            adv_loss,out_bbox = get_adv_loss(outputs[0].cpu(),gt_bboxes_img[obj_id:obj_id+1]) #adv_loss, maximum_bbox related to the target object
            print(adv_loss)
            loss_list.append(adv_loss)
            if True:
                #print(outputs[0].shape)
                img_a = img_a.permute(1,2,0).numpy()*255
                for box in gt_bboxes_img[obj_id:obj_id+1].numpy():
                    box = box.astype(int)
                    cv2.rectangle(img_a, (box[0],box[1]), (box[2],box[3]), (0, 255, 0))
                for box in out_bbox.numpy():
                    box = box.astype(int)
                    cv2.rectangle(img_a, (box[0],box[1]), (box[2],box[3]), (255, 0, 0))
                cv2.imwrite(os.path.join(saved_path, 'image.png'),img_a)
        print(min(loss_list))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--data_root', default='/media/14T/yi/KITTI', 
                        help='your data root for kitti')
    parser.add_argument('--ckpt', default='logs/yolov3_custom/99.weights', help='your checkpoint for kitti')
    parser.add_argument('--model_config_path', default='yolo_pre/yolov3-kitti.cfg')
    parser.add_argument('--saved_path', default='results/yolov3_custom/')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--nclasses', type=int, default=3)
    parser.add_argument('--no_cuda', action='store_true',
                        help='whether to use cuda')
    args = parser.parse_args()

    main(args)
