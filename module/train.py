import os
import gc
import time
import torch
import numpy as np
from tqdm import tqdm
from utils import general
from utils import seg_interp
from utils import interpretation



def train(train_loader, model, epoch, criterion, optimizer, args):
    
    model.train()
    
    running_cls_loss = 0.0
    running_ir_loss = 0.0
    running_corrects = 0.0
    
    for (img, mask), label, use_mask in tqdm(train_loader, desc='train'):
        
        img = img.cuda(args.gpu)
        mask = mask.cuda(args.gpu)
        label = label.cuda(args.gpu)
        use_mask = use_mask.cuda(args.gpu)
        
        # output, lrp_loss, R = model.forward(img, label, mask, use_mask, args)
        
        output = model(img)
        logit = torch.eye(2, device=img.device, dtype=img.dtype)[label] * output
        interp = model.lrp(logit, args, lrp_mode="composite")
        interp = torch.sum(interp, dim=1, keepdim=True)
        interp = interpretation.process_R(interp, args)


        loss_filter = general.get_loss_filter(output, label, use_mask, args)
        ir_loss = seg_interp.__dict__[args.loss_type](interp, mask, loss_filter, args)

        _, pred = torch.max(output, dim=1)
        
        class_loss = criterion(output, label)
        train_loss = class_loss + args.lambda_for_final*ir_loss
        
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        running_cls_loss += class_loss.item() * img.size(0)
        running_ir_loss += ir_loss.item() * img.size(0)
        running_corrects += torch.sum(pred == label.data)

        if np.isnan(train_loss.detach().cpu().numpy()):
            print('loss is nan!')
            raise Exception('End this experiment')
        
        
    # epoch_cls_loss = running_cls_loss / len(train_loader.dataset)
    # epoch_ir_loss = running_ir_loss / len(train_loader.dataset)
    # epoch_acc = running_corrects.float() / len(train_loader.dataset)

#     t.set_description((
#     ' train (clf={:4.4f} lrp={:4.4f} tot={:4.4f})'.format(
#         epoch_cls_loss,
#         epoch_ir_loss,
#         epoch_acc)))
    



def train_autograd(train_loader, model, epoch, criterion, optimizer, args):
    assert args.interpreter in ['simple_grad', 'grad_cam', 'integrated_grad', 'smooth_grad', 'rrr']

    model.train()
    
    running_cls_loss = 0.0
    running_ir_loss = 0.0
    running_corrects = 0.0
    
    for (img, mask), label, use_mask in tqdm(train_loader, desc='train'):
        
        img = img.cuda(args.gpu)
        mask = mask.cuda(args.gpu)
        label = label.cuda(args.gpu)
        use_mask = use_mask.cuda(args.gpu)
        
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
        
        output = model(img)

        class_loss = criterion(output, label)
        target_layer_output = activation[args.target_layer]
        interp = interpretation.__dict__[args.interpreter](label, output, target_layer_output, args, \
            model=model, input_img=img, activation=activation)
        loss_filter = general.get_loss_filter(output, label, use_mask, args)
        ir_loss = seg_interp.__dict__[args.loss_type](interp, mask, loss_filter, args)

        # train_loss = class_loss + args.lambda_for_final*ir_loss
        train_loss = class_loss + args.lambda_for_final * ir_loss
        
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        running_cls_loss += class_loss.item() * img.size(0)
        running_ir_loss += ir_loss.item() * img.size(0)
        running_corrects += torch.sum(torch.max(output, dim=1)[1] == label.data)

        if np.isnan(train_loss.detach().cpu().numpy()):
            print('loss is nan!')
            raise Exception('End this experiment')
        
        hook_handler.remove()
        del activation
        
    # epoch_cls_loss = running_cls_loss / len(train_loader.dataset)
    # epoch_ir_loss = running_ir_loss / len(train_loader.dataset)
    # epoch_acc = running_corrects.float() / len(train_loader.dataset)

#     t.set_description((
#     ' train (clf={:4.4f} lrp={:4.4f} tot={:4.4f})'.format(
#         epoch_cls_loss,
#         epoch_ir_loss,
#         epoch_acc)))
    
    
        






































        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        