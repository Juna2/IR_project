import os
import torch
import torch.nn.functional as F
import numpy as np
from utils import general



def process_R(R, args):
    if args.R_process is None:
        pass
    elif args.R_process == 'max_norm':
        R_abs = torch.abs(R)
        R_max = R_abs.max(2, keepdim=True)[0]
        R_max = R_max.max(3, keepdim=True)[0]
        R = R / (R_max + 1e-8)
    elif args.R_process == 'thr_max_norm':
        R = torch.nn.functional.threshold(R, threshold=0, value=0)
        R_max = R.max(2, keepdim=True)[0]
        R_max = R_max.max(3, keepdim=True)[0]
        R = R / (R_max + 1e-8)
    else:
        raise Exception('Wrong args.R_process : {}'.format(args.R_process))
    return R


def simple_grad(label, output, target_layer_output, args, **kwargs):
    if 'skip_channel_sum' not in list(kwargs.keys()): kwargs['skip_channel_sum'] = False
    if 'skip_R_process' not in list(kwargs.keys()): kwargs['skip_R_process'] = False
    if args.label_oppo_status: label = 1 - label
    selected_scores = output[np.arange(label.shape[0]), label]
    if args.label_oppo_status: label = 1 - label

    target_layer_grad = torch.autograd.grad(outputs=list(selected_scores), 
                                            inputs=target_layer_output, 
                                            retain_graph=True, create_graph=True)[0]

    ####################################################################
    # print('target_layer_grad :', target_layer_grad.shape)
    # check_filename = os.path.join(args.result_path, 'check.hdf5')
    # general.save_hdf5(check_filename, 'target_layer_grad', target_layer_grad.cpu().detach().numpy())
    ####################################################################

    if not kwargs['skip_channel_sum']:
        R = torch.sum(target_layer_grad, dim=1, keepdim=True)
    else:
        R = target_layer_grad

    if args.R_process is None or kwargs['skip_R_process']:
        pass
    elif args.R_process == 'max_norm':
        R_abs = torch.abs(R)
        R_max = R_abs.max(2, keepdim=True)[0]
        R_max = R_max.max(3, keepdim=True)[0]
        R = R / (R_max + 1e-8)
    elif args.R_process == 'thr_max_norm':
        R = torch.nn.functional.threshold(R, threshold=0, value=0)
        R_max = R.max(2, keepdim=True)[0]
        R_max = R_max.max(3, keepdim=True)[0]
        R = R / (R_max + 1e-8)
    else:
        raise Exception('Wrong args.R_process : {}'.format(args.R_process))

    return R


def grad_cam(label, output, target_layer_output, args, **kwargs):
    if args.label_oppo_status: label = 1 - label
    selected_scores = output[np.arange(label.shape[0]), label]
    if args.label_oppo_status: label = 1 - label

    target_layer_grad = torch.autograd.grad(outputs=list(selected_scores), 
                                            inputs=target_layer_output, 
                                            retain_graph=True, 
                                            create_graph=True)[0]

    ####################################################################
    # print('target_layer_grad :', target_layer_grad.shape)
    # check_filename = os.path.join(args.result_path, 'check.hdf5')
    # general.save_hdf5(check_filename, 'target_layer_grad', target_layer_grad.cpu().detach().numpy())
    ####################################################################

    # weight = torch.mean(target_layer_grad, (2, 3), keepdim=True)
    # R = torch.sum(target_layer_output*weight, dim=1, keepdim=True)
    weight = F.adaptive_avg_pool2d(target_layer_grad, 1)
    R = torch.mul(target_layer_output, weight).sum(dim=1, keepdim=True)

    return process_R(R, args)


# def score_cam(label, output, target_layer_output, args, **kwargs):
#     if args.label_oppo_status: label = 1 - label
#     selected_scores = output[np.arange(label.shape[0]), label]
#     if args.label_oppo_status: label = 1 - label

#     print('label :', label.shape)
#     print('output :', output.shape)
#     print('target_layer_output :', target_layer_output.shape)
#     # kwargs['model']
#     raise Exception('finished')

#     return process_R(R, args)



def integrated_grad(label, output, target_layer_output, args, **kwargs): # output, target_layer_output 사용 안함
    iterations = args.smooth_num
    input_zero = torch.zeros_like(kwargs['input_img']).cuda()
    for i in range(iterations):
        alpha = float(i) / iterations
        input_interpolation = (1-alpha) * input_zero + alpha * kwargs['input_img']
        output = kwargs['model'](input_interpolation)
        if i == 0:
            R = simple_grad(label, output, kwargs['activation'][args.target_layer], args, skip_channel_sum=True, skip_R_process=True)
        else:
            R += simple_grad(label, output, kwargs['activation'][args.target_layer], args, skip_channel_sum=True, skip_R_process=True)
    R = (kwargs['activation'][args.target_layer] - torch.zeros_like(kwargs['activation'][args.target_layer])) * (R / iterations)
    R = torch.sum(R, dim=1, keepdim=True)
    
    return process_R(R, args)


def smooth_grad(label, output, target_layer_output, args, **kwargs): # output, target_layer_output 사용 안함
    iterations = args.smooth_num
    # alpha = args.smooth_std
    if 'MSD' in args.dataset_class:
        alpha = args.smooth_std
    elif 'HAM10000' in args.dataset_class:
        alpha = 0.1
    else:
        raise Exception('Unexpected dataset')

    for i in range(iterations):
        inputs_noise = kwargs['input_img'] + alpha*torch.randn(kwargs['input_img'].shape).cuda()
        # activation_output = self.prediction(inputs_noise)
        output = kwargs['model'](inputs_noise)
        if i == 0:
            R = simple_grad(label, output, kwargs['activation'][args.target_layer], args, skip_R_process=True)
        else:
            # If you want to train the model by using LR with smooth grad, then you need to remove detach() function.
            R += simple_grad(label, output, kwargs['activation'][args.target_layer], args, skip_R_process=True)
    R = R / iterations

    return process_R(R, args)


def rrr(label, output, target_layer_output, args, **kwargs):
    if 'skip_channel_sum' not in list(kwargs.keys()): kwargs['skip_channel_sum'] = False
    if 'skip_R_process' not in list(kwargs.keys()): kwargs['skip_R_process'] = False

    # output_softmax = F.softmax(output, dim=1)
    # log_sum = torch.log(output_softmax).sum(1)
    log_sum = F.log_softmax(output, dim=1).sum(1)

    target_layer_grad = torch.autograd.grad(outputs=list(log_sum), 
                                            inputs=target_layer_output, 
                                            retain_graph=True, create_graph=True)[0]

    if not kwargs['skip_channel_sum']:
        R = torch.sum(target_layer_grad, dim=1, keepdim=True)
    else:
        R = target_layer_grad

    if args.R_process is None or kwargs['skip_R_process']:
        pass
    elif args.R_process == 'max_norm':
        R_abs = torch.abs(R)
        R_max = R_abs.max(2, keepdim=True)[0]
        R_max = R_max.max(3, keepdim=True)[0]
        R = R / (R_max + 1e-8)
    elif args.R_process == 'thr_max_norm':
        R = torch.nn.functional.threshold(R, threshold=0, value=0)
        R_max = R.max(2, keepdim=True)[0]
        R_max = R_max.max(3, keepdim=True)[0]
        R = R / (R_max + 1e-8)
    else:
        raise Exception('Wrong args.R_process : {}'.format(args.R_process))

    return R