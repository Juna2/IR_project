import os
import sys
import time
import h5py
import torch
import skimage
from utils import general
from utils import register
from utils import interpretation
import numpy as np
import albumentations as albu
import sklearn.metrics as metrics

from tqdm import tqdm
from torch.utils.data import Dataset


def get_mean_std(train_loader):
# train set으로 normalization을 위한 mean, std 구하기
    print('Calculating mean std of trainset')
    for ch_num in range(3):
        image_list = []
        print('channel {}...'.format(ch_num))
        for (img, mask), label, use_mask in tqdm(train_loader):
            image_list.append(img[:,ch_num,:,:])

        image_ch = np.concatenate(image_list, axis=0)
        print('image_ch :', image_ch.shape)

        print('channel {} mean : {}'.format(ch_num, image_ch.mean()))
        print('channel {} std : {}'.format(ch_num, image_ch.std()))
        print()

def get_masked_img(images, R):
    w = 20
    sigma = 0.5
    R_abs = torch.nn.functional.interpolate(torch.abs(R), (224))
    mask = 1 - torch.sigmoid(w*(R_abs-sigma))
    masked_img = images * mask
    masked_img_new_graph = masked_img.detach().requires_grad_()
    return masked_img_new_graph, masked_img, mask


def get_masked_img_no_grad_skip(images, R):
    w = 20
    sigma = 0.5
    R_abs = torch.nn.functional.interpolate(torch.abs(R), (224))
    mask = 1 - torch.sigmoid(w*(R_abs-sigma))
    masked_img = images * mask
    return masked_img, mask

# def get_mask(gcam, sigma=.5, w=8):
#     gcam = (gcam - F.min(gcam).data)/(F.max(gcam) - F.min(gcam)).data
#     mask = F.squeeze(F.sigmoid(w * (gcam - sigma)))
#     return mask


def thr_max_norm(R):
    R = torch.nn.functional.threshold(R, threshold=0, value=0)
    R_max,_ = R.max(2, keepdim=True)
    R_max,_ = R_max.max(3, keepdim=True)
    R = R / (R_max + 1e-8)
    return R

def max_norm(R):
    R_abs = torch.abs(R)
    R_max,_ = R_abs.max(2, keepdim=True)
    R_max,_ = R_max.max(3, keepdim=True)
    R = R / (R_max + 1e-8)
    return R


def save_or_not(iou_path, key, args):
    if not args.IoU_compact_save:
        saved_img_num = None
        if os.path.exists(iou_path):
            with h5py.File(iou_path, 'r') as f:
                if key in f:
                    saved_img_num = f[key].shape[0]
                    if saved_img_num < args.interp_save_num:
                        return True
                    else:
                        return False
            return True
        else:
            return True
    return False


@register.IoU_arg_setting
def get_IoU_all_pos_autograd_compl(data_loader, model, criterion, args): 

    total_IoU = []
    iou_path = os.path.join(args.result_path, 'IoU.hdf5')
    model.eval()
    threshold_list = ['0.05', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9']
    
    # mask size를 input size와 같게 키워줘야 함.
    ori_mask_size = args.mask_data_size
    data_loader.dataset.mask_size = None # None으로 설정하면 mask가 저장되어 있을 때와 동일한 size로 나옴
    
    # label이 가리키는 score의 반대 score에서도 lrp를 내리기 위한 장치
    status_list = None
    if args.label_oppo:
        status_list = [False, True]
    else:
        status_list = [False]
        
    for status in status_list:
        # args.label_oppo_status가 True면 label의 반대에서 lrp를 내린다.
        args.label_oppo_status = status
        inter_from = None
        if status:
            inter_from = 'oppo'
        else:
            inter_from = 'ori'
            
        img_cul_num = 0 # val img들에 번호를 매겨 다른 setting의 R들과 비교하기 편리하게 하기 위함
        for (data, label, use_mask) in tqdm(data_loader):
            img, mask = data[0].type(torch.FloatTensor).cuda(args.gpu), data[1].type(torch.FloatTensor).cuda(args.gpu)
            label = label.cuda(args.gpu)

            img_idx = np.arange(label.shape[0]) + img_cul_num
            
            activation = {}
            def save_activation(name):
                def hook(model, input, output):
                    activation[name] = output
                return hook

            if 'vgg' in args.model:
                layer, sub_layer = args.target_layer.split('_')
                hook_handler = model.__dict__['_modules'][layer][int(sub_layer)].register_forward_hook(save_activation(args.target_layer))
            else:
                raise Exception('wrong model name')

            # output과 relevance map을 얻음
            output = model(img)
            target_layer_output = activation[args.target_layer]
            R_ori = interpretation.__dict__[args.interpreter](label, output, target_layer_output, args, \
                model=model, input_img=img, activation=activation)

            # output으로 pred를 얻어내고 true or false 여부를 알아냄
            # (interpretation은 label이 가리키는 score에 대해서 구함)
            pred = torch.argmax(output, dim=1)
            true_false = (pred == label).type(torch.cuda.LongTensor)
            
            # R의 양수만 남기고 normalize한 것으로 IoU구함
            R_4_IoU = thr_max_norm(R_ori) # IoU 계산할 때 사용
            R_4_train = max_norm(R_ori) # Visualize할 때 사용

            # true_pos인 relevance map만 골라냄
            true_pos = (true_false * pred).type(torch.BoolTensor) # true_false는 pred가 맞았나 틀렸나를 의미
            R_true_pos = R_4_IoU[true_pos].cpu().detach().numpy()
            img_idx_true_pos = img_idx[true_pos]
            if R_true_pos.shape[0] != 0:
                
                # relevance map 저장
                if save_or_not(iou_path, 'true_pos/{}/R_4_train'.format(inter_from), args):
                    general.save_hdf5(iou_path,
                                    'true_pos/{}/R_4_train'.format(inter_from),
                                    R_4_train[true_pos].cpu().detach().numpy()) ####
                    general.save_hdf5(iou_path,
                                    'true_pos/{}/R_4_IoU'.format(inter_from),
                                    R_true_pos) ####
                                
                # relevance map 224x224로 resize
#                     resize = albu.Resize(R_true_pos.shape[0], R_true_pos.shape[1], args.mask_data_size, args.mask_data_size)
#                     R_true_pos = resize(image=R_true_pos)['image']
                R_true_pos = skimage.transform.resize(R_true_pos, \
                                        (R_true_pos.shape[0], R_true_pos.shape[1], args.mask_data_size, args.mask_data_size))
        
                # true_pos인 seg label만 골라내기
                mask_true_pos = mask.cpu().detach().numpy()[true_pos]
                
                # seg label을 binary로 바꾸기
                mask_true_pos = (mask_true_pos > 0).astype(np.int)
                
                # seg label을 저장
                if save_or_not(iou_path, 'true_pos/{}/mask'.format(inter_from), args):
                    general.save_hdf5(iou_path, 'true_pos/{}/mask'.format(inter_from), mask_true_pos)

                ##############################################################################
                IoU_list = []
                for threshold in threshold_list:
                    # relevance map을 binary로 바꾸기
                    R_true_pos_pos = (R_true_pos > float(threshold)).astype(np.int) ####
                        
                    # relevance map과 seg label로 IoU 계산하기
                    IS = R_true_pos_pos * mask_true_pos
                    union = (mask_true_pos + R_true_pos_pos) - IS
                    IoU = (np.sum(IS.reshape(IS.shape[0], -1), axis=1) / (np.sum(union.reshape(union.shape[0], -1), axis=1) + 1e-8)) * 100
                    IoU_list.append(IoU)
                IoU_arr = np.stack(IoU_list, axis=1) # ex) (8, 10) threshold가 10가지라고 했을 때

                if inter_from == 'ori':
                    total_IoU.append(IoU_arr)
                ##############################################################################
                
                img_true_pos = img.cpu().detach().numpy()[true_pos]
                output_true_pos = output.cpu().detach().numpy()[true_pos]
                label_true_pos = label.cpu().detach().numpy()[true_pos]
                if save_or_not(iou_path, 'true_pos/{}/img'.format(inter_from), args):
                    general.save_hdf5(iou_path, 'true_pos/{}/img'.format(inter_from), img_true_pos)
                    # general.save_hdf5(iou_path, 'true_pos/{}/union'.format(inter_from), union) ####
                    # general.save_hdf5(iou_path, 'true_pos/{}/IS'.format(inter_from), IS) ####
                    general.save_hdf5(iou_path, 'true_pos/{}/img_idx'.format(inter_from), img_idx_true_pos)
                    general.save_hdf5(iou_path, 'true_pos/{}/output'.format(inter_from), output_true_pos)
                    general.save_hdf5(iou_path, 'true_pos/{}/label'.format(inter_from), label_true_pos)
                general.save_hdf5(iou_path, 'true_pos/{}/IoU'.format(inter_from), IoU_arr) ####
            
            



            # label은 tumor인데 pred는 liver로 한 경우의 interpretation을 골라내기 위함
            # (interpretation은 label이 가리키는 score에 대해서 구함) 
            false_neg = ((1 - true_false) * (1 - pred)).type(torch.BoolTensor) # true_false는 pred가 맞았나 틀렸나를 의미
            R_false_neg = R_4_IoU[false_neg].cpu().detach().numpy()
            img_idx_false_neg = img_idx[false_neg]
            if R_false_neg.shape[0] != 0:
                if save_or_not(iou_path, 'false_neg/{}/R_4_train'.format(inter_from), args):
                    general.save_hdf5(iou_path, 
                                        'false_neg/{}/R_4_train'.format(inter_from), 
                                        R_4_train[false_neg].cpu().detach().numpy()) ####
                    general.save_hdf5(iou_path,
                                        'false_neg/{}/R_4_IoU'.format(inter_from),
                                        R_false_neg) ####
                
                # relevance map 224x224로 resize
                # resize = albu.Resize(R_false_neg.shape[0], R_false_neg.shape[1], args.mask_data_size, args.mask_data_size)
                # R_false_neg = resize(image=R_false_neg)['image']
                R_false_neg = skimage.transform.resize(R_false_neg, \
                                        (R_false_neg.shape[0], R_false_neg.shape[1], args.mask_data_size, args.mask_data_size))
        
                # false_neg인 seg label만 골라내기
                mask_false_neg = mask.cpu().detach().numpy()[false_neg]
                
                # seg label을 binary로 바꾸기
                mask_false_neg = (mask_false_neg > 0).astype(np.int)
                
                # seg label을 저장
                if save_or_not(iou_path, 'false_neg/{}/mask'.format(inter_from), args):
                    general.save_hdf5(iou_path, 'false_neg/{}/mask'.format(inter_from), mask_false_neg)

                #####################################################################################
                IoU_list = []
                for threshold in threshold_list:
                    # relevance map을 binary로 바꾸기
                    R_false_neg_pos = (R_false_neg > float(threshold)).astype(np.int)
                
                    # relevance map과 seg label로 IoU 계산하기
                    IS = R_false_neg_pos * mask_false_neg
                    union = (mask_false_neg + R_false_neg_pos) - IS
                    IoU = (np.sum(IS.reshape(IS.shape[0], -1), axis=1) / (np.sum(union.reshape(union.shape[0], -1), axis=1) + 1e-8)) * 100
                    IoU_list.append(IoU)
                IoU_arr = np.stack(IoU_list, axis=1) # ex) (8, 10) threshold가 10가지라고 했을 때

                if inter_from == 'ori':
                    total_IoU.append(IoU_arr)
                #####################################################################################
                
                img_false_neg = img.cpu().detach().numpy()[false_neg]
                output_false_neg = output.cpu().detach().numpy()[false_neg]
                label_false_neg = label.cpu().detach().numpy()[false_neg]
                if save_or_not(iou_path, 'false_neg/{}/img'.format(inter_from), args):
                    general.save_hdf5(iou_path, 'false_neg/{}/img'.format(inter_from), img_false_neg)
                    # general.save_hdf5(iou_path, 'false_neg/{}/union'.format(inter_from), union)
                    # general.save_hdf5(iou_path, 'false_neg/{}/IS'.format(inter_from), IS)
                    general.save_hdf5(iou_path, 'false_neg/{}/img_idx'.format(inter_from), img_idx_false_neg)
                    general.save_hdf5(iou_path, 'false_neg/{}/output'.format(inter_from), output_false_neg)
                    general.save_hdf5(iou_path, 'false_neg/{}/label'.format(inter_from), label_false_neg)
                general.save_hdf5(iou_path, 'false_neg/{}/IoU'.format(inter_from), IoU_arr)
                
            # label은 liver인데 pred는 tumor로 한 경우의 interpretation을 골라내기 위함
            # (interpretation은 label이 가리키는 score에 대해서 구함)
            false_pos = ((1 - true_false) * pred).type(torch.BoolTensor) # true_false는 pred가 맞았나 틀렸나를 의미
            R_false_pos = R_4_IoU[false_pos].cpu().detach().numpy()
            img_idx_false_pos = img_idx[false_pos]
            if R_false_pos.shape[0] != 0:
                if save_or_not(iou_path, 'false_pos/{}/R_4_train'.format(inter_from), args):
                    general.save_hdf5(iou_path, 
                                        'false_pos/{}/R_4_train'.format(inter_from), 
                                        R_4_train[false_pos].cpu().detach().numpy()) ####
                    general.save_hdf5(iou_path,
                                        'false_pos/{}/R_4_IoU'.format(inter_from),
                                        R_false_pos) ####

                img_false_pos = img.cpu().detach().numpy()[false_pos]
                mask_false_pos = mask.cpu().detach().numpy()[false_pos]
                output_false_pos = output.cpu().detach().numpy()[false_pos]
                label_false_pos = label.cpu().detach().numpy()[false_pos]
                if save_or_not(iou_path, 'false_pos/{}/img'.format(inter_from), args):
                    general.save_hdf5(iou_path, 'false_pos/{}/img'.format(inter_from), img_false_pos)
                    general.save_hdf5(iou_path, 'false_pos/{}/mask'.format(inter_from), mask_false_pos)
                    general.save_hdf5(iou_path, 'false_pos/{}/img_idx'.format(inter_from), img_idx_false_pos)
                    general.save_hdf5(iou_path, 'false_pos/{}/output'.format(inter_from), output_false_pos)
                    general.save_hdf5(iou_path, 'false_pos/{}/label'.format(inter_from), label_false_pos)

            # label은 liver인데 pred도 liver로 한 경우의 interpretation을 골라내기 위함
            # (interpretation은 label이 가리키는 score에 대해서 구함)
            true_neg = (true_false * (1 - pred)).type(torch.BoolTensor) # true_false는 pred가 맞았나 틀렸나를 의미
            R_true_neg = R_4_IoU[true_neg].cpu().detach().numpy()
            img_idx_true_neg = img_idx[true_neg]
            if R_true_neg.shape[0] != 0:
                if save_or_not(iou_path, 'true_neg/{}/R_4_train'.format(inter_from), args):
                    general.save_hdf5(iou_path, 
                                        'true_neg/{}/R_4_train'.format(inter_from), 
                                        R_4_train[true_neg].cpu().detach().numpy()) ####
                    general.save_hdf5(iou_path,
                                        'true_neg/{}/R_4_IoU'.format(inter_from),
                                        R_true_neg) ####

                img_true_neg = img.cpu().detach().numpy()[true_neg]
                mask_true_neg = mask.cpu().detach().numpy()[true_neg]
                output_true_neg = output.cpu().detach().numpy()[true_neg]
                label_true_neg = label.cpu().detach().numpy()[true_neg]
                if save_or_not(iou_path, 'true_neg/{}/img'.format(inter_from), args):
                    general.save_hdf5(iou_path, 'true_neg/{}/img'.format(inter_from), img_true_neg)
                    general.save_hdf5(iou_path, 'true_neg/{}/mask'.format(inter_from), mask_true_neg)
                    general.save_hdf5(iou_path, 'true_neg/{}/img_idx'.format(inter_from), img_idx_true_neg)
                    general.save_hdf5(iou_path, 'true_neg/{}/output'.format(inter_from), output_true_neg)
                    general.save_hdf5(iou_path, 'true_neg/{}/label'.format(inter_from), label_true_neg)

            img_cul_num += label.shape[0]

            hook_handler.remove()
            del activation
            ############################################################################
        
    print('IoU.hdf5 saved')
    total_IoU = np.concatenate(total_IoU, axis=0)
    print('total_IoU :', total_IoU.shape)
    IoU_mean = total_IoU.mean(axis=0) ####
    print('IoU_mean :', IoU_mean.shape)

    setting_dict = general.get_setting(args.result_path)
    ############################################################
    for num, threshold in enumerate(threshold_list):
        setting_dict['IoU_{}'.format(threshold)] = ['{:.2f}'.format(IoU_mean[num])]
    setting_dict['best_threshold'] = [threshold_list[IoU_mean.argmax()]]
    setting_dict['best_IoU'] = ['{:.2f}'.format(IoU_mean.max())]
    ############################################################
    general.write_setting(setting_dict, args.result_path)
    
    data_loader.dataset.mask_size = ori_mask_size



@register.IoU_arg_setting
def get_IoU_all_pos(data_loader, model, criterion, args): 

    total_IoU = []
    iou_path = os.path.join(args.result_path, 'IoU.hdf5')
    model.eval()
    threshold_list = ['0.05', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9']
    
    # mask size를 input size와 같게 키워줘야 함.
    ori_mask_size = args.mask_data_size
    data_loader.dataset.mask_size = None # None으로 설정하면 mask가 저장되어 있을 때와 동일한 size로 나옴
    
    with torch.no_grad():
        
        # label이 가리키는 score의 반대 score에서도 lrp를 내리기 위한 장치
        status_list = None
        if args.label_oppo:
            status_list = [False, True]
        else:
            status_list = [False]
            
        for status in status_list:
            # args.label_oppo_status가 True면 label의 반대에서 lrp를 내린다.
            args.label_oppo_status = status
            inter_from = None
            if status:
                inter_from = 'oppo'
            else:
                inter_from = 'ori'
                
            img_cul_num = 0 # val img들에 번호를 매겨 다른 setting의 R들과 비교하기 편리하게 하기 위함
            for (data, label, use_mask) in tqdm(data_loader):
                img, mask = data[0].type(torch.FloatTensor).cuda(args.gpu), data[1].type(torch.FloatTensor).cuda(args.gpu)
                label = label.cuda(args.gpu)

                img_idx = np.arange(label.shape[0]) + img_cul_num
                
                # output과 relevance map을 얻음
                # output, _, R_ori = model.forward(img, label, mask, use_mask, args) # R, R_ori ####
                output = model(img)
                logit = torch.eye(2, device=img.device, dtype=img.dtype)[label] * output
                interp = model.lrp(logit, args, lrp_mode="composite")
                interp = torch.sum(interp, dim=1, keepdim=True)
                R_ori = interpretation.process_R(interp, args)
                
                # output으로 pred를 얻어내고 true or false 여부를 알아냄
                # (interpretation은 label이 가리키는 score에 대해서 구함)
                pred = torch.argmax(output, dim=1)
                true_false = (pred == label).type(torch.cuda.LongTensor)
                
                # R의 양수만 남기고 normalize한 것으로 IoU구함
                R_4_IoU = thr_max_norm(R_ori) # IoU 계산할 때 사용
                R_4_train = max_norm(R_ori) # Visualize할 때 사용

                # true_pos인 relevance map만 골라냄
                true_pos = (true_false * pred).type(torch.BoolTensor) # true_false는 pred가 맞았나 틀렸나를 의미
                R_true_pos = R_4_IoU[true_pos].cpu().numpy()
                img_idx_true_pos = img_idx[true_pos]
                if R_true_pos.shape[0] != 0:
                    
                    # relevance map 저장
                    if save_or_not(iou_path, 'true_pos/{}/R_4_train'.format(inter_from), args):
                        general.save_hdf5(iou_path,
                                          'true_pos/{}/R_4_train'.format(inter_from),
                                          R_4_train[true_pos].cpu().numpy()) ####
                        general.save_hdf5(iou_path,
                                          'true_pos/{}/R_4_IoU'.format(inter_from),
                                          R_true_pos) ####
                    
                    # relevance map 224x224로 resize
#                     resize = albu.Resize(R_true_pos.shape[0], R_true_pos.shape[1], args.mask_data_size, args.mask_data_size)
#                     R_true_pos = resize(image=R_true_pos)['image']
                    R_true_pos = skimage.transform.resize(R_true_pos, \
                                         (R_true_pos.shape[0], R_true_pos.shape[1], args.mask_data_size, args.mask_data_size))
                    
                    # true_pos인 seg label만 골라내기
                    mask_true_pos = mask.cpu().numpy()[true_pos]
                    
                    # seg label을 binary로 바꾸기
                    mask_true_pos = (mask_true_pos > 0).astype(np.int)
                    
                    # seg label을 저장
                    if save_or_not(iou_path, 'true_pos/{}/mask'.format(inter_from), args):
                        general.save_hdf5(iou_path, 'true_pos/{}/mask'.format(inter_from), mask_true_pos)

                    ##############################################################################
                    IoU_list = []
                    for threshold in threshold_list:
                        # relevance map을 binary로 바꾸기
                        R_true_pos_pos = (R_true_pos > float(threshold)).astype(np.int) ####
                            
                        # relevance map과 seg label로 IoU 계산하기
                        IS = R_true_pos_pos * mask_true_pos
                        union = (mask_true_pos + R_true_pos_pos) - IS
                        IoU = (np.sum(IS.reshape(IS.shape[0], -1), axis=1) / (np.sum(union.reshape(union.shape[0], -1), axis=1) + 1e-8)) * 100
                        IoU_list.append(IoU)
                    IoU_arr = np.stack(IoU_list, axis=1) # ex) (8, 10) threshold가 10가지라고 했을 때

                    if inter_from == 'ori':
                        total_IoU.append(IoU_arr)
                    ##############################################################################

                    img_true_pos = img.cpu().numpy()[true_pos]
                    output_true_pos = output.cpu().numpy()[true_pos]
                    label_true_pos = label.cpu().numpy()[true_pos]
                    if save_or_not(iou_path, 'true_pos/{}/img'.format(inter_from), args):
                        general.save_hdf5(iou_path, 'true_pos/{}/img'.format(inter_from), img_true_pos)
                        # general.save_hdf5(iou_path, 'true_pos/{}/union'.format(inter_from), union)
                        # general.save_hdf5(iou_path, 'true_pos/{}/IS'.format(inter_from), IS)
                        general.save_hdf5(iou_path, 'true_pos/{}/img_idx'.format(inter_from), img_idx_true_pos)
                        general.save_hdf5(iou_path, 'true_pos/{}/output'.format(inter_from), output_true_pos)
                        general.save_hdf5(iou_path, 'true_pos/{}/label'.format(inter_from), label_true_pos)
                    general.save_hdf5(iou_path, 'true_pos/{}/IoU'.format(inter_from), IoU_arr) ####
                
                
                # label은 tumor인데 pred는 liver로 한 경우의 interpretation을 골라내기 위함
                # (interpretation은 label이 가리키는 score에 대해서 구함) 
                false_neg = ((1 - true_false) * (1 - pred)).type(torch.BoolTensor) # true_false는 pred가 맞았나 틀렸나를 의미
                R_false_neg = R_4_IoU[false_neg].cpu().numpy()
                img_idx_false_neg = img_idx[false_neg]
                if R_false_neg.shape[0] != 0:
                    if save_or_not(iou_path, 'false_neg/{}/R_4_train'.format(inter_from), args):
                        general.save_hdf5(iou_path, 
                                          'false_neg/{}/R_4_train'.format(inter_from), 
                                          R_4_train[false_neg].cpu().numpy()) ####
                        general.save_hdf5(iou_path,
                                          'false_neg/{}/R_4_IoU'.format(inter_from),
                                          R_false_neg) ####
                    
                    
                    
                    
                    
                    
                    
                    # relevance map 224x224로 resize
#                     resize = albu.Resize(R_false_neg.shape[0], R_false_neg.shape[1], args.mask_data_size, args.mask_data_size)
#                     R_false_neg = resize(image=R_false_neg)['image']
                    R_false_neg = skimage.transform.resize(R_false_neg, \
                                         (R_false_neg.shape[0], R_false_neg.shape[1], args.mask_data_size, args.mask_data_size))
                    
                    # false_neg인 seg label만 골라내기
                    mask_false_neg = mask.cpu().numpy()[false_neg]
                    
                    # seg label을 binary로 바꾸기
                    mask_false_neg = (mask_false_neg > 0).astype(np.int)
                    
                    # seg label을 저장
                    if save_or_not(iou_path, 'false_neg/{}/mask'.format(inter_from), args):
                        general.save_hdf5(iou_path, 'false_neg/{}/mask'.format(inter_from), mask_false_neg)

                    #####################################################################################
                    IoU_list = []
                    for threshold in threshold_list:
                        # relevance map을 binary로 바꾸기
                        R_false_neg_pos = (R_false_neg > float(threshold)).astype(np.int)
                    
                        # relevance map과 seg label로 IoU 계산하기
                        IS = R_false_neg_pos * mask_false_neg
                        union = (mask_false_neg + R_false_neg_pos) - IS
                        IoU = (np.sum(IS.reshape(IS.shape[0], -1), axis=1) / (np.sum(union.reshape(union.shape[0], -1), axis=1) + 1e-8)) * 100
                        IoU_list.append(IoU)
                    IoU_arr = np.stack(IoU_list, axis=1) # ex) (8, 10) threshold가 10가지라고 했을 때

                    if inter_from == 'ori':
                        total_IoU.append(IoU_arr)
                    #####################################################################################
                    
                    img_false_neg = img.cpu().numpy()[false_neg]
                    output_false_neg = output.cpu().numpy()[false_neg]
                    label_false_neg = label.cpu().numpy()[false_neg]
                    if save_or_not(iou_path, 'false_neg/{}/img'.format(inter_from), args):
                        general.save_hdf5(iou_path, 'false_neg/{}/img'.format(inter_from), img_false_neg)
                        # general.save_hdf5(iou_path, 'false_neg/{}/union'.format(inter_from), union)
                        # general.save_hdf5(iou_path, 'false_neg/{}/IS'.format(inter_from), IS)
                        general.save_hdf5(iou_path, 'false_neg/{}/img_idx'.format(inter_from), img_idx_false_neg)
                        general.save_hdf5(iou_path, 'false_neg/{}/output'.format(inter_from), output_false_neg)
                        general.save_hdf5(iou_path, 'false_neg/{}/label'.format(inter_from), label_false_neg)
                    general.save_hdf5(iou_path, 'true_pos/{}/IoU'.format(inter_from), IoU_arr) ####
                    
                # label은 liver인데 pred는 tumor로 한 경우의 interpretation을 골라내기 위함
                # (interpretation은 label이 가리키는 score에 대해서 구함)
                false_pos = ((1 - true_false) * pred).type(torch.BoolTensor) # true_false는 pred가 맞았나 틀렸나를 의미
                R_false_pos = R_4_IoU[false_pos].cpu().numpy()
                img_idx_false_pos = img_idx[false_pos]
                if R_false_pos.shape[0] != 0:
                    if save_or_not(iou_path, 'false_pos/{}/R_4_train'.format(inter_from), args):
                        general.save_hdf5(iou_path, 
                                          'false_pos/{}/R_4_train'.format(inter_from), 
                                          R_4_train[false_pos].cpu().numpy()) ####
                        general.save_hdf5(iou_path,
                                          'false_pos/{}/R_4_IoU'.format(inter_from),
                                          R_false_pos) ####

                    img_false_pos = img.cpu().numpy()[false_pos]
                    mask_false_pos = mask.cpu().numpy()[false_pos]
                    output_false_pos = output.cpu().numpy()[false_pos]
                    label_false_pos = label.cpu().numpy()[false_pos]
                    if save_or_not(iou_path, 'false_pos/{}/img'.format(inter_from), args):
                        general.save_hdf5(iou_path, 'false_pos/{}/img'.format(inter_from), img_false_pos)
                        general.save_hdf5(iou_path, 'false_pos/{}/mask'.format(inter_from), mask_false_pos)
                        general.save_hdf5(iou_path, 'false_pos/{}/img_idx'.format(inter_from), img_idx_false_pos)
                        general.save_hdf5(iou_path, 'false_pos/{}/output'.format(inter_from), output_false_pos)
                        general.save_hdf5(iou_path, 'false_pos/{}/label'.format(inter_from), label_false_pos)

                # label은 liver인데 pred도 liver로 한 경우의 interpretation을 골라내기 위함
                # (interpretation은 label이 가리키는 score에 대해서 구함)
                true_neg = (true_false * (1 - pred)).type(torch.BoolTensor) # true_false는 pred가 맞았나 틀렸나를 의미
                R_true_neg = R_4_IoU[true_neg].cpu().numpy()
                img_idx_true_neg = img_idx[true_neg]
                if R_true_neg.shape[0] != 0:
                    if save_or_not(iou_path, 'true_neg/{}/R_4_train'.format(inter_from), args):
                        general.save_hdf5(iou_path, 
                                          'true_neg/{}/R_4_train'.format(inter_from), 
                                          R_4_train[true_neg].cpu().numpy()) ####
                        general.save_hdf5(iou_path,
                                          'true_neg/{}/R_4_IoU'.format(inter_from),
                                          R_true_neg) ####

                    img_true_neg = img.cpu().numpy()[true_neg]
                    mask_true_neg = mask.cpu().numpy()[true_neg]
                    output_true_neg = output.cpu().numpy()[true_neg]
                    label_true_neg = label.cpu().numpy()[true_neg]
                    if save_or_not(iou_path, 'true_neg/{}/img'.format(inter_from), args):
                        general.save_hdf5(iou_path, 'true_neg/{}/img'.format(inter_from), img_true_neg)
                        general.save_hdf5(iou_path, 'true_neg/{}/mask'.format(inter_from), mask_true_neg)
                        general.save_hdf5(iou_path, 'true_neg/{}/img_idx'.format(inter_from), img_idx_true_neg)
                        general.save_hdf5(iou_path, 'true_neg/{}/output'.format(inter_from), output_true_neg)
                        general.save_hdf5(iou_path, 'true_neg/{}/label'.format(inter_from), label_true_neg)

                



                img_cul_num += label.shape[0]
                ############################################################################
            
        print('IoU.hdf5 saved')
        total_IoU = np.concatenate(total_IoU, axis=0)
        print('total_IoU :', total_IoU.shape)
        IoU_mean = total_IoU.mean(axis=0) ####
        print('IoU_mean :', IoU_mean.shape)

        setting_dict = general.get_setting(args.result_path)
        ############################################################
        for num, threshold in enumerate(threshold_list):
            setting_dict['IoU_{}'.format(threshold)] = ['{:.2f}'.format(IoU_mean[num])]
        setting_dict['best_threshold'] = [threshold_list[IoU_mean.argmax()]]
        setting_dict['best_IoU'] = ['{:.2f}'.format(IoU_mean.max())]
        ############################################################
        general.write_setting(setting_dict, args.result_path)
    
    data_loader.dataset.mask_size = ori_mask_size




