import os
import gc
from utils import general
from utils import seg_interp
from utils import interpretation

import time
import torch
import numpy as np
from tqdm import tqdm

import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score



def validation(name, val_loader, model, epoch, criterion, args): 
    
    model.eval()
    
    running_val_loss = 0.0
    running_cls_loss = 0.0
    running_ir_loss = 0.0
    running_corrects = 0
    use_mask_num = 0
    
    label_list, pred_list, pred_bin = [], [], [] # pred_bin is for keeping track of binary predictions.
    with torch.no_grad():
        for (img, mask), label, use_mask in tqdm(val_loader, desc='validation with {}set'.format(name)):

            img = img.cuda(args.gpu)
            mask = mask.cuda(args.gpu)
            label = label.cuda(args.gpu) # torch.Size([8])
            use_mask = use_mask.cuda(args.gpu) # torch.Size([8])
            
            # output, ir_loss, R = model.forward(img, label, mask, use_mask, args)
            output = model(img)
            logit = torch.eye(2, device=img.device, dtype=img.dtype)[label] * output
            interp = model.lrp(logit, args, lrp_mode="composite")
            interp = torch.sum(interp, dim=1, keepdim=True)
            interp = interpretation.process_R(interp, args)

            loss_filter = general.get_loss_filter(output, label, use_mask, args)
            ir_loss = seg_interp.__dict__[args.loss_type](interp, mask, loss_filter, args)
    
            _, pred = torch.max(output, dim=1)
            class_loss = criterion(output, label)
            val_loss = class_loss + args.lambda_for_final * ir_loss
    
            running_val_loss += val_loss.item() * img.size(0)
            running_cls_loss += class_loss.item() * img.size(0)
            running_ir_loss += ir_loss.item() * use_mask.sum() # 얘는 img.size(0)를 안썼다는 것에 유의!
            running_corrects += torch.sum(pred == label.data)
            use_mask_num += use_mask.sum()
        
            label_list.append(label.cpu().data.numpy())
            pred_list.append(torch.nn.functional.softmax(output, dim=1).cpu().data.numpy())
            pred_bin.append(pred.cpu().numpy())
            
    epoch_val_loss = running_val_loss / len(val_loader.dataset)
    epoch_cls_loss = running_cls_loss / len(val_loader.dataset)
    epoch_ir_loss = running_ir_loss / (use_mask_num + 1e-8) # 얘는 len(val_loader.dataset)를 안썼다는 것에 유의!
    epoch_acc = running_corrects.float() / len(val_loader.dataset)
    
    total_label = np.concatenate(label_list)
    total_score = np.concatenate(pred_list)
    accuracy = accuracy_score(total_label, total_score.argmax(axis=1))    
    auc = metrics.roc_auc_score(np.squeeze(total_label), total_score[:,1])
    pred_bin = np.hstack(pred_bin)
     
    print(epoch, 'Average {} cls_loss: {:.4f}, ir_loss: {:.4f}, acc: {:.4f}, pred: {}/{}'.format(
        name, epoch_cls_loss, epoch_ir_loss, epoch_acc, np.sum(pred_bin==0), np.sum(pred_bin==1)))
    print()
    
#     print('use_mask_num :', use_mask_num)
#     print('epoch_ir_loss :', epoch_ir_loss)
#     print('epoch_ir_loss :', type(epoch_ir_loss))
    result_filename = os.path.join(args.result_path, 'result.hdf5')
    general.save_hdf5(result_filename, '{}_loss'.format(name), np.array([epoch_val_loss]))
    general.save_hdf5(result_filename, '{}_class_loss'.format(name), np.array([epoch_cls_loss]))
    general.save_hdf5(result_filename, '{}_ir_loss'.format(name), np.array([epoch_ir_loss.item()]))
    general.save_hdf5(result_filename, '{}_accuracy'.format(name), np.array([accuracy.item()]))
    general.save_hdf5(result_filename, '{}_auc'.format(name), np.array([auc.item()]))
    
    return accuracy, auc



def validation_autograd(name, val_loader, model, epoch, criterion, args): 
    assert args.interpreter in ['simple_grad', 'grad_cam', 'score_cam', 'integrated_grad', 'smooth_grad', 'rrr']

    model.eval()
    
    running_val_loss = 0.0
    running_cls_loss = 0.0
    running_ir_loss = 0.0
    running_corrects = 0
    use_mask_num = 0
    
    label_list, pred_list, pred_bin = [], [], [] # pred_bin is for keeping track of binary predictions.

    for (img, mask), label, use_mask in tqdm(val_loader, desc='validation with {}set'.format(name)):

        img = img.cuda(args.gpu)
        mask = mask.cuda(args.gpu)
        label = label.cuda(args.gpu) # torch.Size([8])
        use_mask = use_mask.cuda(args.gpu) # torch.Size([8])
        
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
        _, pred = torch.max(output, dim=1)
        class_loss = criterion(output, label)
        target_layer_output = activation[args.target_layer]
        interp = interpretation.__dict__[args.interpreter](label, output, target_layer_output, args, \
            model=model, input_img=img, activation=activation)
        loss_filter = general.get_loss_filter(output, label, use_mask, args)
        ir_loss = seg_interp.__dict__[args.loss_type](interp, mask, loss_filter, args)
        val_loss = class_loss + args.lambda_for_final*ir_loss

        running_val_loss += val_loss.item() * img.size(0)
        running_cls_loss += class_loss.item() * img.size(0)
        running_ir_loss += ir_loss.item() * use_mask.sum() # 얘는 img.size(0)를 안썼다는 것에 유의!
        running_corrects += torch.sum(pred == label.data)
        use_mask_num += use_mask.sum()
    
        label_list.append(label.cpu().data.numpy())
        pred_list.append(torch.nn.functional.softmax(output, dim=1).cpu().data.numpy())
        pred_bin.append(pred.cpu().numpy())

        hook_handler.remove()
        del activation
            
    epoch_val_loss = running_val_loss / len(val_loader.dataset)
    epoch_cls_loss = running_cls_loss / len(val_loader.dataset)
    epoch_ir_loss = running_ir_loss / (use_mask_num + 1e-8) # 얘는 len(val_loader.dataset)를 안썼다는 것에 유의!
    epoch_acc = running_corrects.float() / len(val_loader.dataset)
    
    total_label = np.concatenate(label_list)
    total_score = np.concatenate(pred_list)
    accuracy = accuracy_score(total_label, total_score.argmax(axis=1))    
    auc = metrics.roc_auc_score(np.squeeze(total_label), total_score[:,1])
    pred_bin = np.hstack(pred_bin)
    
    print(epoch, 'Average {} cls_loss: {:.4f}, ir_loss: {:.4f}, acc: {:.4f}, pred: {}/{}'.format(
        name, epoch_cls_loss, epoch_ir_loss, epoch_acc, np.sum(pred_bin==0), np.sum(pred_bin==1)))
    print()
    
#     print('use_mask_num :', use_mask_num)
#     print('epoch_ir_loss :', epoch_ir_loss)
#     print('epoch_ir_loss :', type(epoch_ir_loss))
    result_filename = os.path.join(args.result_path, 'result.hdf5')
    general.save_hdf5(result_filename, '{}_loss'.format(name), np.array([epoch_val_loss]))
    general.save_hdf5(result_filename, '{}_class_loss'.format(name), np.array([epoch_cls_loss]))
    general.save_hdf5(result_filename, '{}_ir_loss'.format(name), np.array([epoch_ir_loss.item()]))
    general.save_hdf5(result_filename, '{}_accuracy'.format(name), np.array([accuracy.item()]))
    general.save_hdf5(result_filename, '{}_auc'.format(name), np.array([auc.item()]))
    
    return accuracy, auc











    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    