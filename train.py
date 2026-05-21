
from socketserver import DatagramRequestHandler
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
import torch.utils.data
import os
import shutil
from PIL import Image
import numpy as np
import logging
from tensorboardX import SummaryWriter
from skimage import measure, io
import skimage.morphology as ski_morph
import matplotlib.pyplot as plt

from model import PointAnnoModel as create_model
import utils
from utils import cropping_center
from dataset import DataFolder
from my_transforms import get_transforms
from loss import calculate_loss, mse_loss
import torch.nn.functional as F


def main(opt):
    global best_score, num_iter, tb_writer, logger, logger_results
    best_score = 0
    opt.isTrain = True

    if not os.path.exists(opt.train['save_dir']):
        os.makedirs(opt.train['save_dir'])
    tb_writer = SummaryWriter('{:s}/tb_logs'.format(opt.train['save_dir']))

    use_cuda = torch.cuda.is_available() and len(opt.train['gpus']) > 0
    if use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in opt.train['gpus'])
    device = torch.device('cuda' if use_cuda else 'cpu')

    opt.define_transforms()
    opt.save_options()

    # set up logger
    logger, logger_results = setup_logging(opt)

    # ----- create model ----- #
    model_name = opt.model['name']
    model = create_model(model_name, opt.model['out_c'], opt.model['pretrained'])
    # if not opt.train['checkpoint']:
    #     logger.info(model)
    if use_cuda and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    # ----- define optimizer ----- #
    optimizer = torch.optim.Adam(model.parameters(), opt.train['lr'], betas=(0.9, 0.99),
                                 weight_decay=opt.train['weight_decay'])

    # ----- define criterion ----- #
    criterion = torch.nn.MSELoss(reduction='none').to(device)
    # criterion = mse_loss

    # ----- load data ----- #
    img_dir = '{:s}/train'.format(opt.train['img_dir'])
    target_dir = '{:s}/train'.format(opt.train['label_dir'])
    init_mask_dir = '{:s}/train'.format(opt.train['init_mask_dir'])
    true_mask_dir = '{:s}/train'.format(opt.train['true_mask_dir'])
    if opt.round == 0:
        dir_list = [img_dir, target_dir, init_mask_dir, true_mask_dir]
        post_fix = ['label_detect.png', 'init_mask.png', 'true_mask.png']
        num_channels = [3, 1, 1, 1]
        train_transform = get_transforms(opt.transform['train_stage1'])
    else:
        bg_dir = '{:s}/train'.format(opt.train['bg_dir'])
        dir_list = [img_dir, target_dir, bg_dir, init_mask_dir, true_mask_dir]
        post_fix = ['label_detect.png', 'label_bg.png', 'init_mask.png', 'true_mask.png']
        num_channels = [3, 1, 1, 1, 1]
        train_transform = get_transforms(opt.transform['train_stage2'])
    dataset = opt.dataset
    train_set = DataFolder(dir_list, post_fix, num_channels, train_transform, dataset)
    # print("train_set")
    # print(train_set)
    train_loader = DataLoader(train_set, batch_size=opt.train['batch_size'], shuffle=True,
                              num_workers=opt.train['workers'])
    val_transform = get_transforms(opt.transform['val'])

    # ----- training and validation ----- #
    num_epoch = opt.train['train_epochs']
    num_iter = num_epoch * len(train_loader)
    # print training parameters
    logger.info("=> Initial learning rate: {:g}".format(opt.train['lr']))
    logger.info("=> Batch size: {:d}".format(opt.train['batch_size']))
    logger.info("=> Number of training iterations: {:d}".format(num_iter))
    logger.info("=> Training epochs: {:d}".format(opt.train['train_epochs']))

    for epoch in range(num_epoch):
        # train for one epoch or len(train_loader) iterations
        logger.info('Epoch: [{:d}/{:d}]'.format(epoch+1, num_epoch))
        train_loss = train(opt, train_loader, model, optimizer, criterion, device)

        # evaluate on val set
        with torch.no_grad():
            val_recall, val_prec, val_F1 = validate(opt, model, val_transform, device)

        # check if it is the best accuracy
        is_best = val_F1 > best_score
        best_score = max(val_F1, best_score)

        cp_flag = True if (epoch + 1) % opt.train['checkpoint_freq'] == 0 else False
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, epoch, is_best, opt.train['save_dir'], cp_flag)

        # save the training results to txt files
        logger_results.info('{:d}\t{:.4f} || {:.4f}\t{:.4f}\t{:.4f}'
                            .format(epoch + 1, train_loss, val_recall, val_prec, val_F1))
        # tensorboard logs
        tb_writer.add_scalars('epoch_loss', {'train_loss': train_loss}, epoch)
        tb_writer.add_scalars('epoch_acc', {'val_recall': val_recall, 'val_prec': val_prec, 'val_F1': val_F1}, epoch)

    tb_writer.close()
    for i in list(logger.handlers):
        logger.removeHandler(i)
        i.flush()
        i.close()
    for i in list(logger_results.handlers):
        logger_results.removeHandler(i)
        i.flush()
        i.close()


def train(opt, train_loader, model, optimizer, criterion, device):
    # list to store the average loss for this epoch
    det_results = utils.AverageMeter(1)
    mutual_results = utils.AverageMeter(1)
    all_results = utils.AverageMeter(1)
    # switch to train mode
    model.train()
    seg_prob_list = []
    print('train loader', train_loader)

    for i, sample in enumerate(train_loader):
        if opt.round == 0:
            input, target, init_mask, true_mask = sample[0], sample[1], sample[2], sample[3]
            target = target.squeeze(1)
            input, target, init_mask, true_mask = input.to(device), target.to(device), init_mask.to(device), true_mask.to(device)
        else:
            input, target, bg, init_mask, true_mask = sample[0], sample[1], sample[2], sample[3], sample[4]
            target = target.squeeze(1)
            bg = bg.squeeze(1)
            input, target, bg, init_mask, true_mask = input.to(device), target.to(device), bg.to(device), init_mask.to(device), true_mask.to(device)
            # print("bg", bg.max())
            # print("init_mask", init_mask.max())
            # print("true_mask", true_mask.max())


        output = model(input)

        true_dict = {}
        pred_dict = {}

        pred_seg_hv = output["hv"]
        pred_seg_np = F.softmax(output["np"],dim=1)#[:,1:,:,:]
        pred_seg_encoding = output["seg_encoding"]

        pred_dict["seg_hv"] = pred_seg_hv
        pred_dict["seg_np"] = pred_seg_np
        pred_dict["seg_encoding"] = pred_seg_encoding

        # np.save("pred_hv.npy", pred_seg_hv.cpu().detach().numpy())



        det_prob = output["det_prob"].squeeze(1)
        det_prob_ = torch.sigmoid(det_prob)
        det_prob = det_prob_.unsqueeze(1)
        det_prob = utils.crop_to_shape(det_prob, pred_seg_np, data_format="NCHW").permute(0,2,3,1).contiguous()
        det_prob_1 = torch.logical_and(det_prob > opt.test['threshold'], det_prob < 0.9)
        
        # print(det_prob.sum())

        init_mask = utils.crop_to_shape(init_mask, pred_seg_np, data_format="NCHW")
        init_mask = init_mask.permute(0,2,3,1).contiguous()
        # print("det prob", det_prob.shape)
        # print("init mask", init_mask.shape)
        # fused_prob = torch.logical_or(det_prob_1, init_mask)
        fused_prob = init_mask
        # plt.imshow(fused_prob.cpu().numpy()[0,...,0])
        # plt.show()
        # plt.savefig("fuse.png")
        # 先注释掉
        dist_map = utils.generate_training_np(det_prob)
        retain = (dist_map>0)#.astype(np.float32)
        retain = torch.tensor(retain, device=device)
        # plt.imshow(retain.cpu().numpy()[0,...,0])
        # plt.show()
        # plt.savefig("retain.png")
        # print("fuse prob", fused_prob.shape)
        # print("retain", retain.shape)

        # fused_prob = (fused_prob * retain.cuda())#.type(torch.float32)
        # plt.imshow(fused_prob.cpu().numpy()[0,...,0])
        # plt.show()
        # plt.savefig("fil.png")
        # print(fused_prob)

        pred_det_encoding = output["det_encoding"]


        true_dict["fused_prob"] = fused_prob
        true_dict["det_encoding"] = pred_det_encoding
        true_dict["true_np"] = cropping_center(true_mask[:,2:,...]/255, (164,164), batch=True)


        # true_dict["true_hv"] = cropping_center(true_mask[:,:2,...]/127-1, (164,164), batch=True)
        # true_dict["true_hv"] = cropping_center(true_mask[:,:2,...]-1, (164,164), batch=True)

        true_hv = cropping_center(true_mask[:,:2,...]/127-1, (164,164), batch=True)
        true_hv[true_hv<0] = -1
        true_hv[true_hv>0] = 1
        true_dict["true_hv"] = true_hv


        # np.save("true_np.npy", true_dict["true_np"].cpu().numpy())
        # np.save("true_hv.npy", true_dict["true_hv"].cpu().numpy())

        # print("="*80)
        # print(true_dict["true_np"].shape)
        # print(true_dict["true_hv"].min(), true_dict["true_hv"].max())
        # hv_map = cropping_center(hv_map, crop_shape)
        # np_map = cropping_center(np_map, crop_shape)



        mask = torch.zeros_like(target).float()
        for k in range(target.size(0)):
            mask_k = ski_morph.dilation(target[k].cpu().numpy()==1, footprint=ski_morph.disk(opt.r2))
            mask[k] = torch.Tensor(mask_k.astype(np.float64))

        # update background
        if opt.round > 0:
            mask = (mask + bg) > 0
            mask = mask.float()
        weight_map = mask.float().clone()
        weight_map[target > 0] = 10

        loss_map = criterion(det_prob_,target)
        det_loss = torch.sum(loss_map * weight_map) / mask.sum()
        mutual_loss = calculate_loss(true_dict, pred_dict, opt.model["weight"]).squeeze(dim=-1)



        loss = det_loss + mutual_loss
        det_result = [det_loss.item(),]
        det_results.update(det_result, input.size(0))

        mutual_result = [mutual_loss.item(),]
        mutual_results.update(mutual_result, input.size(0))

        all_result = [loss.item(),]
        all_results.update(all_result, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        del input, output, loss, det_loss, mutual_loss

        if i % opt.train['log_interval'] == 0:
            logger.info('\tIteration: [{:d}/{:d}]\tDet_Loss {r[0]:.4f}'.format(i, len(train_loader), r=det_results.avg))
            logger.info('\tIteration: [{:d}/{:d}]\tMutual_Loss {r[0]:.4f}'.format(i, len(train_loader), r=mutual_results.avg))
            logger.info('\tIteration: [{:d}/{:d}]\tAll_Loss {r[0]:.4f}'.format(i, len(train_loader), r=all_results.avg))
    # seg_prob_arr = np.array(seg_prob_list)
    # np.save(f"{opt.round}_seg_prob.npy", seg_prob_arr)
    logger.info('\t=> Train Avg: Loss {r[0]:.4f}'.format(r=all_results.avg))

    return all_results.avg[0]


def validate(opt, model, data_transform, device):
    total_TP = 0.0
    total_FP = 0.0
    total_FN = 0.0

    # switch to evaluate mode
    model.eval()

    img_dir = '{:s}/images/val'.format(opt.train['data_dir'])
    label_dir = opt.test['label_dir']

    img_names = os.listdir(img_dir)
    prob_map_list = []
    for img_name in img_names:
        # load test image
        img_path = '{:s}/{:s}'.format(img_dir, img_name)
        img = Image.open(img_path)
        name = os.path.splitext(img_name)[0]

        label_path = '{:s}/{:s}_label_point.png'.format(label_dir, name)
        gt = io.imread(label_path)

        input, label = data_transform((img, Image.fromarray(gt)))
        input = input.unsqueeze(0)

        prob_map = get_probmaps(input, model, opt, device)
        prob_map = prob_map.cpu().numpy()
        # print(prob_map)
        # print("=" * 80)

        prob_map_list.append(prob_map)
        pred = prob_map > opt.test['threshold']  # prediction
        pred_labeled, N = measure.label(pred, return_num=True)
        if N > 1:
            bg_area = ski_morph.remove_small_objects(pred_labeled, opt.post['max_area']) > 0
            large_area = ski_morph.remove_small_objects(pred_labeled, opt.post['min_area']) > 0
            pred = pred * (bg_area==0) * (large_area>0)

        TP, FP, FN = utils.compute_accuracy(pred, gt, radius=opt.r1)
        total_TP += TP
        total_FP += FP
        total_FN += FN

    recall = float(total_TP) / (total_TP + total_FN + 1e-8)
    precision = float(total_TP) / (total_TP + total_FP + 1e-8)
    F1 = 2 * precision * recall / (precision + recall + 1e-8)
    logger.info('\t=> Val Avg:\tRecall {:.4f}\tPrec {:.4f}\tF1 {:.4f}'.format(recall, precision, F1))
    prob_map_arr = np.array(prob_map_list)
    # np.save("/data115_1/zhangye/my_code_TNBC/prob_map.npy", prob_map_arr)
    return recall, precision, F1


def get_probmaps(input, model, opt, device):
    size = opt.test['patch_size']
    overlap = opt.test['overlap']

    if size == 0:
        with torch.no_grad():
            output = model(input.to(device))
    else:
        output, seg_prob = utils.split_forward(model, input.to(device), size, overlap)
    output = output.squeeze(0)
    prob_maps = torch.sigmoid(output[0,:,:])

    return prob_maps


def save_checkpoint(state, epoch, is_best, save_dir, cp_flag):
    cp_dir = '{:s}/checkpoints'.format(save_dir)
    if not os.path.exists(cp_dir):
        os.mkdir(cp_dir)
    filename = '{:s}/checkpoint.pth.tar'.format(cp_dir)
    torch.save(state, filename)
    if cp_flag:
        shutil.copyfile(filename, '{:s}/checkpoint_{:d}.pth.tar'.format(cp_dir, epoch+1))
    if is_best:
        shutil.copyfile(filename, '{:s}/checkpoint_best.pth.tar'.format(cp_dir))


def setup_logging(opt):
    mode = 'w'

    # create logger for training information
    logger = logging.getLogger('train_logger')
    logger.setLevel(logging.DEBUG)
    # create console handler and file handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler('{:s}/train.log'.format(opt.train['save_dir']), mode=mode)
    file_handler.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter('%(asctime)s\t%(message)s', datefmt='%m-%d %I:%M')
    # add formatter to handlers
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    # add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # create logger for epoch results
    logger_results = logging.getLogger('results')
    logger_results.setLevel(logging.DEBUG)
    file_handler2 = logging.FileHandler('{:s}/epoch_results.txt'.format(opt.train['save_dir']), mode=mode)
    file_handler2.setFormatter(logging.Formatter('%(message)s'))
    logger_results.addHandler(file_handler2)

    logger.info('***** Training starts *****')
    logger.info('save directory: {:s}'.format(opt.train['save_dir']))
    if mode == 'w':
        logger_results.info('epoch\ttrain_loss\tval_recall\tval_prec\tval_F1')

    return logger, logger_results


if __name__ == '__main__':
    main()
