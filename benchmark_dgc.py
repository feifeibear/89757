import pdb
import sys
import argparse
import os
import time
import logging
from random import uniform
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import models
from torch.autograd import Variable
from data import get_dataset
from preprocess import get_transform
from utils import *
from ast import literal_eval
from torch.nn.utils import clip_grad_norm
from math import ceil
from math import sqrt
import numpy as np
from prune_utils.pruning import select_top_k, select_top_k_appr, check_sparsity
import horovod.torch as hvd
from horovod.torch.mpi_ops import poll, synchronize

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ConvNet Training')

parser.add_argument('--results_dir', metavar='RESULTS_DIR',
                    default='./Results/benchmark', help='results dir')
parser.add_argument('--save', metavar='SAVE', default='',
                    help='saved folder')
parser.add_argument('--dataset', metavar='DATASET', default='cifar10',
                    help='dataset name or folder')
parser.add_argument('--model', '-a', metavar='MODEL', default='resnet',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: alexnet)')
parser.add_argument('--input_size', type=int, default=None,
                    help='image input size')
parser.add_argument('--resnet_depth', type=int, default=18,
                    help='depth of resnet')
parser.add_argument('--model_config', default='',
                    help='additional architecture configuration')
parser.add_argument('--type', default='torch.cuda.FloatTensor',
                    help='type of tensor - e.g torch.cuda.HalfTensor')
parser.add_argument('--gpus', default='0',
                    help='gpus used for training - e.g 0,1,3')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=2048, type=int,
                    metavar='N', help='mini-batch size (default: 2048)')
parser.add_argument('--lr_bb_fix', dest='lr_bb_fix', action='store_true',
                    help='learning rate fix for big batch lr =  lr0*(batch_size/128)**0.5')
parser.add_argument('--no-lr_bb_fix', dest='lr_bb_fix', action='store_false',
                    help='learning rate fix for big batch lr =  lr0*(batch_size/128)**0.5')
parser.set_defaults(lr_bb_fix=True)
parser.add_argument('--regime_bb_fix', dest='regime_bb_fix', action='store_true',
                    help='regime fix for big batch e = e0*(batch_size/128)')
parser.add_argument('--no-regime_bb_fix', dest='regime_bb_fix', action='store_false',
                    help='regime fix for big batch e = e0*(batch_size/128)')
parser.set_defaults(regime_bb_fix=False)
parser.add_argument('--optimizer', default='SGD', type=str, metavar='OPT',
                    help='optimizer function used')
parser.add_argument('--lr', '--learning_rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', type=str, metavar='FILE',
                    help='evaluate model FILE on validation set')

parser.add_argument('-mb', '--mini-batch-size', default=64, type=int,
                    help='mini-mini-batch size (default: 64)')
parser.add_argument('--ghost_batch_size', type=int, default=0,
                    help='used for ghost batch size')
# parser.add_argument('--use_pruning', type=bool, default=False,
#                    help='whether use pruning')
parser.add_argument('--pruning_perc', default=0.1, type=float,
                    help='the percent of pruning gradient')
# parser.add_argument('--use_residue_acc', type=bool, default=False,
#                     help='whether use pruning')

parser.add_argument('--use_residue_acc', dest='use_residue_acc', action='store_true',
                    help='use residue accumulating')
parser.add_argument('--no_use_residue_acc', dest='use_residue_acc', action='store_false',
                    help='do not use residue accumulating')
parser.set_defaults(use_residue_acc=False)

parser.add_argument('--use_pruning', dest='use_pruning', action='store_true',
                    help='use gradient pruning')
parser.add_argument('--no_use_pruning', dest='use_pruning', action='store_false',
                    help='do not use gradient pruning')
parser.set_defaults(use_pruning=False)

parser.add_argument('--use_warmup', dest='use_warmup', action='store_true',
                    help='use warm up')
parser.add_argument('--no_use_warmup', dest='use_warmup', action='store_false',
                    help='do not use warm up')
parser.set_defaults(use_warmup=False)

parser.add_argument('--use_sync', dest='use_sync', action='store_true',
                    help='synchronize all parameters every sync_interval steps')
parser.add_argument('--no_use_sync', dest='use_sync', action='store_false',
                    help='synchronize all parameters every sync_interval steps')
parser.set_defaults(use_sync=False)

parser.add_argument('--sync_interval', default=100, type=int,
                    help='sync interval (default: 100)')

parser.add_argument('--use_nesterov', dest='use_nesterov', action='store_true',
                    help='to debug')
parser.add_argument('--no_use_nesterov', dest='use_nesterov', action='store_false',
                    help='no debug')
parser.set_defaults(use_nesterov=False)

parser.add_argument('--use_cluster', dest='use_cluster', action='store_true',
                    help='synchronize all parameters every sync_interval steps')
parser.add_argument('--no_use_cluster', dest='use_cluster', action='store_false',
                    help='synchronize all parameters every sync_interval steps')
parser.set_defaults(use_cluster=False)

parser.add_argument('--use_hvddist', dest='use_hvddist', action='store_true',
                    help='to use orignal hvddist')
parser.add_argument('--no_use_hvddist', dest='use_hvddist', action='store_false',
                    help='no use orignal hvddist')
parser.set_defaults(use_hvddist=False)




parser.add_argument('--use_debug', dest='use_debug', action='store_true',
                    help='to debug')
parser.add_argument('--no_use_debug', dest='use_debug', action='store_false',
                    help='no debug')
parser.set_defaults(use_debug=False)

parser.add_argument('--pruning_mode', '-pm', default=0, type=int,
                    help='prune mode')
def main():
    hvd.init()
    size = hvd.size()
    local_rank = hvd.local_rank()

    torch.manual_seed(123 + hvd.rank())
    global args, best_prec1
    best_prec1 = 0
    args = parser.parse_args()

    if args.pruning_mode == 1:
        print("thd mode")
        from hvd_utils.DGCoptimizer_thd import DGCDistributedOptimizer
    #elif args.pruning_mode == 2:
    #    print("chunck mode")
    #    from hvd_utils.DGCoptimizer_chunck import DGCDistributedOptimizer
    #elif args.pruning_mode == 3:
    #    print("topk mode")
    #    from hvd_utils.DGCoptimizer import DGCDistributedOptimizer
    #elif args.pruning_mode == 6:
    #    print("seperate mode")
    #    from hvd_utils.DGCoptimizer_thd_sep import DGCDistributedOptimizer
    #elif args.pruning_mode == 7:
    #    print("topk quant mode")
    #    from hvd_utils.DGCoptimizer_quant import DGCDistributedOptimizer
    #elif args.pruning_mode == 8:
    #    print("topk quant mode")
    #    from hvd_utils.DGCoptimizer_thd_quant import DGCDistributedOptimizer
    elif args.pruning_mode == 10:
        print("hybrid mode")
        from hvd_utils.DGCoptimizer_hybrid import DGCDistributedOptimizer
    elif args.pruning_mode == 11:
        print("hybrid quant mode")
        from hvd_utils.DGCoptimizer_hybrid_quant import DGCDistributedOptimizer
    elif args.pruning_mode == 12:
        print("hybrid v2 quant mode")
        from hvd_utils.DGCoptimizer_hybrid_quantv2 import DGCDistributedOptimizer
    elif args.pruning_mode == 13:
        print("hybrid v2 mode")
        from hvd_utils.DGCoptimizer_hybridv2 import DGCDistributedOptimizer
    else:
        print("pruning_mode should be set correctly")
        exit(0)
    from hvd_utils.DGCoptimizer_commoverlap import myhvdOptimizer



    if args.evaluate:
        args.results_dir = '/tmp'
    if args.save is '':
        args.save = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = os.path.join(args.results_dir, args.save)
    if not os.path.exists(save_path):
        if hvd.rank() == 0:
            os.makedirs(save_path)
        else:
            time.sleep(1)

    if hvd.rank() == 0:
        setup_logging(os.path.join(save_path, 'log.txt'))
        results_file = os.path.join(save_path, 'results.%s')
        results = ResultsLog(results_file % 'csv', results_file % 'html')

    if hvd.rank() == 0:
        logging.info("saving to %s", save_path)
        logging.debug("run arguments: %s", args)

    if 'cuda' in args.type:
        torch.cuda.manual_seed(123 + hvd.rank())
        args.gpus = [int(i) for i in args.gpus.split(',')]

        if args.use_cluster:
            torch.cuda.set_device(hvd.local_rank())
        else:
            if(hvd.local_rank() < len(args.gpus)):
                print("rank, ", hvd.local_rank(), " is runing on ", args.gpus[hvd.local_rank()])
                torch.cuda.set_device(args.gpus[hvd.local_rank()])
            else:
                print("rank, ", hvd.local_rank(), " is runing on ", args.gpus[0])
                torch.cuda.set_device(args.gpus[0])
        cudnn.benchmark = True
    else:
        args.gpus = None

    # create model
    logging.info("creating model %s", args.model)
    model = models.__dict__[args.model]
    model_config = {'input_size': args.input_size, 'dataset': args.dataset, 'depth': args.resnet_depth}

    if args.model_config is not '':
        model_config = dict(model_config, **literal_eval(args.model_config))

    model = model(**model_config)
    logging.info("created model with configuration: %s", model_config)

    # optionally resume from a checkpoint
    if args.evaluate:
        if not os.path.isfile(args.evaluate):
            parser.error('invalid checkpoint: {}'.format(args.evaluate))
        checkpoint = torch.load(args.evaluate)
        model.load_state_dict(checkpoint['state_dict'])
        logging.info("loaded checkpoint '%s' (epoch %s)",
                     args.evaluate, checkpoint['epoch'])
    elif args.resume:
        checkpoint_file = args.resume
        if os.path.isdir(checkpoint_file):
            results.load(os.path.join(checkpoint_file, 'results.csv'))
            checkpoint_file = os.path.join(
                checkpoint_file, 'model_best.pth.tar')
        if os.path.isfile(checkpoint_file):
            logging.info("loading checkpoint '%s'", args.resume)
            checkpoint = torch.load(checkpoint_file)
            args.start_epoch = checkpoint['epoch'] + 1
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            logging.info("loaded checkpoint '%s' (epoch %s)",
                         checkpoint_file, checkpoint['epoch'])
        else:
            logging.error("no checkpoint found at '%s'", args.resume)

    num_parameters = sum([l.nelement() for l in model.parameters()])
    logging.info("number of parameters: %d", num_parameters)

    # Data loading code
    default_transform = {
        'train': get_transform(args.dataset,
                               input_size=args.input_size, augment=True),
        'eval': get_transform(args.dataset,
                              input_size=args.input_size, augment=False)
    }
    transform = getattr(model, 'input_transform', default_transform)
    regime = getattr(model, 'regime', {0: {'optimizer': args.optimizer,
                                           'lr': args.lr
                                           #'momentum': args.momentum,
                                           #'weight_decay': args.weight_decay
                                           }})
    adapted_regime = {}
    logging.info('self-defined momentum : %f, weight_decay : %f', args.momentum, args.weight_decay)
    for e, v in regime.items():
        if args.lr_bb_fix and 'lr' in v:
            # v['lr'] *= (args.batch_size / args.mini_batch_size) ** 0.5
            v['lr'] *= (args.batch_size * hvd.size() / 128) ** 0.5
        adapted_regime[e] = v
    regime = adapted_regime

    # define loss function (criterion) and optimizer
    criterion = getattr(model, 'criterion', nn.CrossEntropyLoss)()
    criterion.type(args.type)
    model.type(args.type)

    #val_data = get_dataset(args.dataset, 'val', transform['eval'])
    #val_loader = torch.utils.data.DataLoader(
    #    val_data,
    #    batch_size=args.batch_size, shuffle=False,
    #    num_workers=args.workers, pin_memory=True)
    val_loader = None

    if args.evaluate:
        validate(val_loader, model, criterion, 0)
        return

    #train_data = get_dataset(args.dataset, 'train', transform['train'])
    #train_loader = torch.utils.data.DataLoader(
    #    train_data,
    #    batch_size=args.batch_size, shuffle=True,
    #    num_workers=args.workers, pin_memory=True)
    train_loader = None

    if hvd.rank() == 0:
        logging.info('training regime: %s', regime)
        print({i: list(w.size())
               for (i, w) in enumerate(list(model.parameters()))})
    init_weights = [w.data.cpu().clone() for w in list(model.parameters())]

    U = []
    V = []
    print("current rank ", hvd.rank(), "local_rank ", hvd.local_rank(), \
            " USE_PRUNING ", args.use_pruning)
    print("model ", args.model, " use_nesterov ", args.use_nesterov)

    #TODO u, v will be cleared at the begining of each epoch
    if args.use_pruning:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        if args.gpus is not None:
            optimizer = DGCDistributedOptimizer(optimizer, named_parameters=model.named_parameters(), use_gpu=True, momentum=0.9, weight_decay=1e-4)
        else:
            optimizer = DGCDistributedOptimizer(optimizer, named_parameters=model.named_parameters(), use_gpu=False, momentum=0.9, weight_decay=1e-4)
    else:
        if args.use_hvddist:
            print("use orignal hvd DistributedOptimizer")
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4,
                    nesterov=True)
            #optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
            optimizer = myhvdOptimizer(optimizer, named_parameters=model.named_parameters())
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
            if args.gpus is not None:
                optimizer = DGCDistributedOptimizer(optimizer, named_parameters=model.named_parameters(), use_gpu=True, momentum=0.9, weight_decay=1e-4, use_allgather=False)
            else:
                optimizer = DGCDistributedOptimizer(optimizer, named_parameters=model.named_parameters(), use_gpu=False, momentum=0.9, weight_decay=1e-4, use_allgather=False)

    hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    global_begin_time = time.time()
    for epoch in range(args.start_epoch, args.epochs // hvd.size()):
        #optimizer = adjust_optimizer(optimizer, epoch, regime)
        for e, v in regime.items():
            if epoch == e // hvd.size():
                for param_group in optimizer.param_groups:
                    param_group['lr'] = v['lr']
                break

        # train for one epoch
        train_result = train(train_loader, model, criterion, epoch, optimizer, U, V)
        sys.exit()

        train_loss, train_prec1, train_prec5, U, V = [
            train_result[r] for r in ['loss', 'prec1', 'prec5', 'U', 'V']]

        # evaluate on validation set
        val_result = validate(val_loader, model, criterion, epoch)
        val_loss, val_prec1, val_prec5 = [val_result[r]
                                          for r in ['loss', 'prec1', 'prec5']]

        # remember best prec@1 and save checkpoint
        is_best = val_prec1 > best_prec1
        best_prec1 = max(val_prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'model': args.model,
            'config': args.model_config,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'regime': regime
        }, is_best, path=save_path)
        if hvd.rank() == 0:
            if torch.__version__ == "0.4.0":
                logging.info('\n Epoch: {0}\t'
                     'Training Loss {train_loss:.4f} \t'
                     'Training Prec@1 {train_prec1:.3f} \t'
                     'Training Prec@5 {train_prec5:.3f} \t'
                     'Validation Loss {val_loss:.4f} \t'
                     'Validation Prec@1 {val_prec1:.3f} \t'
                     'Validation Prec@5 {val_prec5:.3f} \n'
                     .format(epoch + 1, train_loss=train_loss.cpu().numpy(), val_loss=val_loss.cpu().numpy(),
                             train_prec1=train_prec1.cpu().numpy(), val_prec1=val_prec1.cpu().numpy(),
                             train_prec5=train_prec5.cpu().numpy(), val_prec5=val_prec5.cpu().numpy()))
            else:
                logging.info('\n Epoch: {0}\t'
                     'Training Loss {train_loss:.4f} \t'
                     'Training Prec@1 {train_prec1:.3f} \t'
                     'Training Prec@5 {train_prec5:.3f} \t'
                     'Validation Loss {val_loss:.4f} \t'
                     'Validation Prec@1 {val_prec1:.3f} \t'
                     'Validation Prec@5 {val_prec5:.3f} \n'
                     .format(epoch + 1, train_loss=train_loss, val_loss=val_loss,
                             train_prec1=train_prec1, val_prec1=val_prec1,
                             train_prec5=train_prec5, val_prec5=val_prec5))

        #Enable to measure more layers
        idxs = [0]#,2,4,6,7,8,9,10]#[0, 12, 45, 63]

        step_dist_epoch = {'step_dist_n%s' % k: (w.data.cpu() - init_weights[k]).norm()
                           for (k, w) in enumerate(list(model.parameters())) if k in idxs}


        if(hvd.rank() == 0):
            current_time = time.time()
            if hvd.rank() == 0:
                results.add(epoch=epoch + 1, train_loss=train_loss.cpu().numpy(), val_loss=val_loss.cpu().numpy(),
                        train_error1=100 - train_prec1.cpu().numpy(), val_error1=100 - val_prec1.cpu().numpy(),
                        train_error5=100 - train_prec5.cpu().numpy(), val_error5=100 - val_prec5.cpu().numpy(),
                        eslapse = current_time - global_begin_time)
            else:
                results.add(epoch=epoch + 1, train_loss=train_loss, val_loss=val_loss,
                        train_error1=100 - train_prec1, val_error1=100 - val_prec1,
                        train_error5=100 - train_prec5, val_error5=100 - val_prec5,
                        eslapse = current_time - global_begin_time)

            #results.plot(x='epoch', y=['train_loss', 'val_loss'],
            #             title='Loss', ylabel='loss')
            #results.plot(x='epoch', y=['train_error1', 'val_error1'],
            #             title='Error@1', ylabel='error %')
            #results.plot(x='epoch', y=['train_error5', 'val_error5'],
            #             title='Error@5', ylabel='error %')

            #for k in idxs:
            #    results.plot(x='epoch', y=['step_dist_n%s' % k],
            #                 title='step distance per epoch %s' % k,
            #                 ylabel='val')

            results.save()


def forward(data_loader, model, criterion, epoch=0, training=True, optimizer=None, U=None, V=None):
    # hvd
    # if args.gpus and len(args.gpus) > 1:
    #    model = torch.nn.DataParallel(model, args.gpus)

    batch_time = AverageMeter()
    pruning_time = AverageMeter()
    select_time = AverageMeter()
    mask_time= AverageMeter()
    pack_time = AverageMeter()
    unpack_time = AverageMeter()
    mom_time = AverageMeter()
    allreduce_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()


    input_var = None
    target_var = None
    # for i, (inputs, target) in enumerate(data_loader):
    #     if args.gpus is not None:
    #         target = target.cuda(async=True)
    #     input_var = Variable(inputs.type(args.type), volatile=not training)
    #     target_var = Variable(target)
    #     if hvd.rank() == 0:
    #         print(input_var.size())
    #         print(target_var.size())
    #     break
    print("before iter")
    if "imagenet" in args.dataset:
        input_var = Variable(torch.randn(args.batch_size, 3, 244, 244).cuda(), volatile=not training)
        target_var = Variable(torch.LongTensor(args.batch_size).random_(0, 1000).cuda())
    elif "cifar10" in args.dataset:
        input_var = Variable(torch.randn(args.batch_size, 3, 32, 32).cuda(), volatile=not training)
        target_var = Variable(torch.LongTensor(args.batch_size).random_(0, 10).cuda())
    elif "cifar100" in args.dataset:
        input_var = Variable(torch.randn(args.batch_size, 3, 32, 32).cuda(), volatile=not training)
        target_var = Variable(torch.LongTensor(args.batch_size).random_(0, 100).cuda())

    for i in range(200):
        # measure data loading time

        # compute output
        if not training:
            output = model(input_var)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target_var.data, topk=(1, 5))
            losses.update(loss.data[0], input_var.size(0))
            top1.update(prec1[0], input_var.size(0))
            top5.update(prec5[0], input_var.size(0))

        else:
            if i > 30:
                torch.cuda.synchronize()
                end = time.time()

            output = model(input_var)
            loss = criterion(output, target_var)

            prec1, prec5 = accuracy(output.data, target_var.data, topk=(1, 5))
            losses.update(loss.data[0], input_var.size(0))
            top1.update(prec1[0], input_var.size(0))
            top5.update(prec5[0], input_var.size(0))

            #if args.use_pruning:
            #    clip_grad_norm(model.parameters(), 5. * (hvd.size() ** -0.5))
            if args.use_pruning:
                torch.cuda.synchronize()
                optimizer.pruning_time = 0.0
                optimizer.select_time = 0.0
                optimizer.pack_time = 0.0
                optimizer.unpack_time = 0.0
                optimizer.mask_time= 0.0
                optimizer.mom_time= 0.0
                optimizer.allreduce_time= 0.0


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            # Master
            if args.use_pruning:
                if i > 30:
                    torch.cuda.synchronize()
                    pruning_time.update(optimizer.pruning_time)
                    select_time.update(optimizer.select_time)
                    pack_time.update(optimizer.pack_time)
                    unpack_time.update(optimizer.unpack_time)
                    mask_time.update(optimizer.mask_time)
                    mom_time.update(optimizer.mom_time)
                    allreduce_time.update(optimizer.allreduce_time)
                            # if args.use_pruning:
            # else:
            #     # idx = 0
            #     # for p in list(model.parameters()):
            #     #     # print("accumulated sparsity is", check_sparsity(g))
            #     #     # TODO 1. use pytorch sgd optimizer to calculate mom and weight_decay, set mom and wd
            #     #     # used with pruning
            #     #     # TODO 2. implement weight_decay and momentum by myself, set mom=0 and wd = 0
            #     #     # used with baseline
            #     #     g = p.grad.data
            #     #     g.add_(torch.mul(p.data, args.weight_decay))
            #     #     V[idx] = torch.add(torch.mul(V[idx], args.momentum), g)
            #     #     p.grad.data = V[idx]
            #     #     idx = idx+1
            #     optimizer.synchronize()
            #     clip_grad_norm(model.parameters(), 5.)


            # measure elapsed time
            if i > 30:
                torch.cuda.synchronize()
                batch_time.update(time.time() - end)

        if i % args.print_freq == 0:
            if hvd.rank() == 0:
                logging.info('{phase} - Epoch: [{0}][{1}/{2}]\t'
                             'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                             'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                             'Prune {pruning_time.val:.9f} ({pruning_time.avg:.3f})\t'
                             'Select {select_time.val:.9f} ({select_time.avg:.3f})\t'
                             'mask {mask_time.val:.9f} ({mask_time.avg:.3f})\t'
                             'pack {pack_time.val:.9f} ({pack_time.avg:.3f})\t'
                             'unpack {unpack_time.val:.9f} ({unpack_time.avg:.3f})\t'
                             'mom {mom_time.val:.9f} ({mom_time.avg:.3f})\t'
                             'allreduce_time {allreduce_time.val:.9f} ({allreduce_time.avg:.3f})\t'
                             'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                             'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                             'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                                 epoch, i, 1000,
                                 phase='TRAINING' if training else 'EVALUATING',
                                 batch_time=batch_time,
                                 data_time=data_time,
                                 pruning_time = pruning_time,
                                 select_time = select_time,
                                 mask_time = mask_time,
                                 pack_time = pack_time,
                                 unpack_time = unpack_time,
                                 allreduce_time = allreduce_time,
                                 mom_time = mom_time,
                             loss=losses, top1=top1, top5=top5))

    return {'loss': losses.avg,
            'prec1': top1.avg,
            'prec5': top5.avg,
            'U' : U,
            'V' : V}


def train(data_loader, model, criterion, epoch, optimizer, U, V):
    # switch to train mode
    model.train()
    return forward(data_loader, model, criterion, epoch,
                   training=True, optimizer=optimizer, U=U, V=V)


def validate(data_loader, model, criterion, epoch):
    # switch to evaluate mode
    model.eval()
    return forward(data_loader, model, criterion, epoch,
                   training=False, optimizer=None, U=None, V=None)


if __name__ == '__main__':
    main()
