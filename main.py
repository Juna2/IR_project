import os
import time
import h5py
import random
import datetime
import numpy as np

# import sys
# sys.path.append('utils')
# sys.path.append('module')
# sys.path.append('datasets')

import torch
import torch.nn as nn
import torchvision

from module.arguments import get_args
args = get_args()
torch.cuda.set_device(args.gpu)
 
if args.seed is not None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

from utils import general
from utils import help_main
from utils import configuration
from utils import interpretation
from tqdm import tqdm

import module.train as train
import module.val as val


# arguments 저장
curr_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]
args.result_path = 'result/{}_{}'.format(args.result_path_tag, curr_time)
general.save_args(args.result_path, args)

assert args.loader_IoU in ['valid', 'test'], '{} is not on the list and val->(valid or test).'.format(args.loader_IoU)

# model
model = configuration.setup_model(args).cuda(args.gpu)
print(model)

# loss function
criterion = nn.CrossEntropyLoss().cuda(args.gpu)

# update할 param 지정 
params_to_update = None
if args.only_upd_fc:
    params_to_update = model.classifier.parameters()
else:
    params_to_update = model.parameters()

# optimizer
optimizer = torch.optim.Adam(params_to_update, lr=args.lr,
                            weight_decay=args.weight_decay)


# dataset
print('#################################### Dataset ####################################')
train_dataset = configuration.setup_dataset('train', args)
val_dataset = configuration.setup_dataset('valid', args)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size, shuffle=True,
    num_workers=args.workers, pin_memory=True, sampler=None
)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True
)
print('#################################################################################')

# start training
if not args.skip_train:
    val.__dict__[args.val_func]('train', train_loader, model, 0, criterion, args)
    val.__dict__[args.val_func]('val', val_loader, model, 0, criterion, args)
    start_epoch = 1
    best_acc = 0

    for epoch in range(start_epoch, args.epochs+1):
        
        # train
        train.__dict__[args.train_func](train_loader, model, epoch, criterion, optimizer, args)
        
        # val
        _   = val.__dict__[args.val_func]('train', train_loader, model, epoch, criterion, args)
        acc, auc = val.__dict__[args.val_func]('val', val_loader, model, epoch, criterion, args)

        is_best = acc >= best_acc
        best_acc = max(acc, best_acc)

        if is_best:
            setting_dict = general.get_setting(args.result_path)
            setting_dict['best_val_acc'] = ['{:.3f}'.format(best_acc)]
            setting_dict['best_val_auc'] = ['{:.3f}'.format(auc)]
            general.write_setting(setting_dict, args.result_path)
            
            multi_optim_list = ['sgd_no_mom_adam', 'sgd_no_mom_sgd']
            optim_state_dict = optimizer.state_dict() if not args.optimizer in multi_optim_list else optimizer2.state_dict()
            torch.save({
                'epoch': epoch, 'model': args.model, 'state_dict': model.state_dict(),
                'best_acc': best_acc, 'optimizer' : optim_state_dict,
            }, os.path.join(args.result_path, 'model_best.pth.tar'))
            
    # metric이 지정이 안됐으면 eval을 할 필요가 없으므로 모델도 로드하지 않음
    if args.metrics is not None:
        args.model_path = os.path.join(args.result_path, 'model_best.pth.tar')
        args.convert_model_from = 'same_kind' # cdep은 일반 torchvision model을 사용해서 lrp모듈 모델과의 호환을 위해 필요함
        model = configuration.setup_model(args).cuda(args.gpu)
        print('======= start testing =======')
else:
    print('############################')
    print('######  SKIP TRAINING  #####')
    print('############################')
    

# validate()의 epoch 인자로 0을 줄 것이기 때문에 
# 반드시 args.skip_train == True일 때만 해야 함
if args.skip_train:
    print()
    
    if args.val_func == 'interpretation':
        print('getting interpretation ...')
        val.__dict__[args.val_func]('val', val_loader, model, 0, criterion, args)
    elif args.val_func == 'for_tumor_size_recall_hist':
        print('getting for_tumor_size_recall_hist ...')
        val.__dict__[args.val_func]('val', val_loader, model, 0, criterion, args)
    elif args.val_func == 'skip':
        print('skip validation function')
        pass
    else:
        print('getting accuracy & AUC ...')
        acc, auc = val.__dict__[args.val_func]('val', val_loader, model, 0, criterion, args)
        setting_dict = general.get_setting(args.result_path)
        setting_dict['best_val_acc'] = ['{:.3f}'.format(acc)]
        setting_dict['best_val_auc'] = ['{:.3f}'.format(auc)]
        general.write_setting(setting_dict, args.result_path)

if args.metrics is not None and 'IoU' in args.metrics:
    print('getting IoU...')
    if args.loader_IoU == 'train':
        help_main.__dict__[args.IoU_func](train_loader, model, criterion, args)
    elif args.loader_IoU == 'valid':
        help_main.__dict__[args.IoU_func](val_loader, model, criterion, args)
    else:
        raise Exception('Wrong loader : {}'.format(args.loader_IoU))

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    