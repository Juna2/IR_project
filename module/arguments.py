import argparse

import sys
sys.path.append('../')

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str_None(v):
    if v.lower() in ['None', 'none', 'no', 'Non']:
        return None
    else:
        return v

def get_args():

    parser = argparse.ArgumentParser(description='LRP')

    parser.add_argument('--data_path', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('-a', '--model', metavar='ARCH')
    parser.add_argument('-j', '--workers', default=5, type=int, metavar='N',
                        help='number of data loading workers')
    parser.add_argument('--epochs', type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch_size', default=8, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight_decay', default=0, type=float,
                        metavar='W', help='weight decay (default: 0)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--lrp-test-model', type=int, metavar='PATH',
                        help='choose one of models you want have a test with')
    parser.add_argument("--evaluate", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="evaluate model on validation set")
    parser.add_argument('--what_to_patch_on_train', type=str_None, metavar='PATH', default='benign',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--what_to_patch_on_val', type=str_None, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument("--pretrained", type=str2bool, nargs='?',
                        const=True, default=True,
                        help="use pretrained weight initialization")
    parser.add_argument("--skip_train", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="skip the train and just evaluate")
    parser.add_argument("--use_autograd", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="use torch.autograd when getting an interpretation")
    parser.add_argument("--no_patch", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="no_patch for ISICDataset")
    parser.add_argument('--patch_transparency', type=int, metavar='N', default=255,
                        help="Patch's transparency 0-255, 0 is completely transparent")
    parser.add_argument("--no_healthy_pat", type=str2bool, nargs='?',
                        help="Do not use patient without tumor")
    parser.add_argument('--only_upd_fc', type=str2bool, metavar='PATH',  ####### 조심!!!
                        help='only update fully connected layer')
    parser.add_argument('--threshold', nargs='+', type=str, 
                        default=['0.05', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9'],
                        help='list of threshold for IoU')
    parser.add_argument('--model_path', type=str_None, metavar='PATH',
                        help='use pre-trained model & the path of the model')
    parser.add_argument('--convert_model_from', metavar='PATH',
                        help='Convert model from cdep or rsr-lrp or same_kind')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--dataset_np_seed', default=0, type=int,
                        help='Seed for dataset it decides overall numpy seed as well.')
    parser.add_argument('--ip', type=str,
                        help='which server is used')
    parser.add_argument('--seg_label', type=str, metavar='PATH',
                        help='seg_label ex) segmentation_resized or segmentation_resized_old')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('-rp', '--result-path', type=str,
                        help='path to save the result.')
    parser.add_argument('-t', '--trial', type=int,
                        help='number of different weight init trial.')
    parser.add_argument('--mask_data_size', type=int, default=56,
                        help='set the size of mask coming out from dataloader')  # help를 잘 읽어봐라
    parser.add_argument("--check_activ_output_shape", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="check_activ_output_shape")
    parser.add_argument('--screen_pid', type=str, metavar='PATH', default="-",
                        help='choose a model_type')
    # parser.add_argument('--threshold', type=float, help='IoU threshold.')
    parser.add_argument('--pixel_range_max', type=float, default=0,
                        help='pixel_range_max')
    parser.add_argument('--pixel_range_min', type=float, default=0,
                        help='pixel_range_min')
    parser.add_argument('--metrics', type=str_None,
                        help='all measurements to do after training.')  #
    parser.add_argument("--loss_alter", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Update by gradient of losses alternately")
    parser.add_argument('--dataset_class', type=str, metavar='PATH',
                        help='Determine dataset class')
    parser.add_argument('--input_depth', type=int, default=3,
                        help='set input depth')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='num_classes')
    parser.add_argument('--interp_save_num', type=int, default=50,
                        help='num of interpretation to save')
    parser.add_argument('--min_tumor_size', type=int, default=0,
                        help='minimum number of pixels of tumor for data')
    parser.add_argument("--small_dataset", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Use small dataset for test")
    parser.add_argument('--train_func', type=str, metavar='PATH',
                        help='Determine train function')  # gamma
    parser.add_argument('--val_func', type=str, metavar='PATH',
                        help='Determine validation function')  # gamma
    parser.add_argument('--interpreter', type=str, default='lrp',
                        help='if it is lrp, lrp will be used as a interpreter. If not, gradcam will be used')
    parser.add_argument('--R_process', type=str_None, default='max_norm',
                        help='How to process R after getting R at target layer')
    parser.add_argument('--loss_filter', type=str_None,
                        help='From which data you want to calculate loss(lrp_loss). ex) true_pos, pos, neg, all')
    parser.add_argument("--label_oppo", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Get interpretation from opposite label")
    parser.add_argument("--label_oppo_status", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="This is set by args.label_oppo_status so do not touch it")
    parser.add_argument('--train_ratio', type=float, default=1.0,
                        help='Ratio of how much train data want to use')
    parser.add_argument('--mask_ratio', type=float, default=1.0,
                        help='Ratio of how much train data want to use')
    parser.add_argument('--optimizer', type=str,
                        help='optimizer')
    parser.add_argument('--loader_IoU', type=str, default='valid',
                        help='data loader used for IoU')
    parser.add_argument('--IoU_func', type=str,
                        help='data loader used for IoU')
    parser.add_argument('--IoU_compact_save', type=str2bool, default=True,
                        help='save only IoU when calculation IoU')
    parser.add_argument('--result_path_tag', type=str,
                        help='result path tag')
    parser.add_argument('--lrp_target_layer', type=str_None,
                        help='label of mixed data 2')
    parser.add_argument('--target_layer', type=str_None,
                        help='A target layer for newly implemented autograd interpretation system')
    
    parser.add_argument('--loss_type', type=str, default='mask_LRP_seg',
                        help='absR or uniformR or corner or frame or sparseR')
    parser.add_argument('--smooth_std', type=float, default=0.1,
                        help='gamma for fucntion heatmap() from render.py ')
    parser.add_argument('--smooth_num', type=int, default=3,
                        help='T.T')

    parser.add_argument('--lambda_for_final', type=float,  # 0.001, #0.5,
                        help='if args. no_lambda_for_each_layer is True, regulizer rate for total loss. this is multiplied with lrp_loss')
    

    args = parser.parse_args()
    assert isinstance(args.loss_filter, str) or args.loss_filter is None
    assert args.convert_model_from in ['same_kind', 'cdep', 'rsr-lrp', 'custom']
    assert args.no_healthy_pat is not None
    assert args.only_upd_fc is not None
    assert args.optimizer is not None
    assert args.lrp_target_layer is not None or args.target_layer is not None

    return args
