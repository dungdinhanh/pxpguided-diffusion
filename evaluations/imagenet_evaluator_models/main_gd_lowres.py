import argparse
import os
import time
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import math
from evaluations.imagenet_evaluator_models.models_imp import *
from evaluations.imagenet_evaluator_models.data_loader import data_loader_gd
from evaluations.imagenet_evaluator_models.helper import AverageMeter, save_checkpoint2, accuracy, adjust_learning_rate

from guided_diffusion import dist_util, logger
from hfai.nn.parallel import DistributedDataParallel as DDP
import hfai
import hfai.multiprocessing
import hfai.distributed as dist
import hfai.client

model_names = [
    'alexnet', 'squeezenet1_0', 'squeezenet1_1', 'densenet121',
    'densenet169', 'densenet201', 'densenet201', 'densenet161',
    'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19', 'vgg19_bn', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152'
]

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-a', '--arch', metavar='ARCH', default='alexnet', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: alexnet)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='numer of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful to restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR',
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='Weight decay (default: 1e-4)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-m', '--pin-memory', dest='pin_memory', action='store_true',
                    help='use pin memory')
parser.add_argument('-p', '--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--print-freq', '-f', default=10, type=int, metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoitn, (default: None)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--log_dir', help="saving log and model directory", default="eval_models/test")
parser.add_argument('--img_size', help="Image size", default=64, type=int)
parser.add_argument('--local', dest='local', action='store_true')

best_prec1 = 0.0


def main(local_rank):
    global args, best_prec1
    args = parser.parse_args()
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)
    model_name = f"{args.arch}_im{args.img_size}"
    last_checkpoint_name = f"{model_name}_last.pt"
    last_checkpoint = os.path.join(log_dir, last_checkpoint_name)
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
    else:
        print("=> creating model '{}'".format(args.arch))

    if args.arch == 'alexnet':
        model = alexnet(pretrained=args.pretrained)
    elif args.arch == 'squeezenet1_0':
        model = squeezenet1_0(pretrained=args.pretrained)
    elif args.arch == 'squeezenet1_1':
        model = squeezenet1_1(pretrained=args.pretrained)
    elif args.arch == 'densenet121':
        if args.img_size == 64 or args.img_size == 128:
            model = densenet121lr(pretrained=args.pretrained)
            conv0 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
            n = conv0.kernel_size[0] * conv0.kernel_size[1] * conv0.out_channels
            conv0.weight.data.normal_(0, math.sqrt(2. / n))
            model.features.conv0 = conv0
            model.features.norm0 = nn.BatchNorm2d(64)
        else:
            model = densenet121(pretrained=args.pretrained)
    elif args.arch == 'densenet169':
        if args.img_size == 64 or args.img_size == 128:
            model = densenet169lr()
            conv0 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
            n = conv0.kernel_size[0] * conv0.kernel_size[1] * conv0.out_channels
            conv0.weight.data.normal_(0, math.sqrt(2. / n))
            model.features.conv0 = conv0
            model.features.norm0 = nn.BatchNorm2d(64)
        else:
            model = densenet169(pretrained=args.pretrained)
    elif args.arch == 'densenet201':
        if args.img_size == 64 or args.img_size == 128:
            model = densenet201lr(pretrained=args.pretrained)
            conv0 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
            n = conv0.kernel_size[0] * conv0.kernel_size[1] * conv0.out_channels
            conv0.weight.data.normal_(0, math.sqrt(2. / n))
            model.features.conv0 = conv0
            model.features.norm0 = nn.BatchNorm2d(64)
        else:
            model = densenet201(pretrained=args.pretrained)
    elif args.arch == 'densenet161':
        if args.img_size == 64 or args.img_size == 128:
            model = densenet161lr(pretrained=args.pretrained)
            conv0 = nn.Conv2d(3, 96, kernel_size=7, stride=1, padding=3, bias=False)
            n = conv0.kernel_size[0] * conv0.kernel_size[1] * conv0.out_channels
            conv0.weight.data.normal_(0, math.sqrt(2. / n))
            model.features.conv0 = conv0
            model.features.norm0 = nn.BatchNorm2d(96)
        else:
            model = densenet161(pretrained=args.pretrained)
    elif args.arch == 'resnet18':
        model = resnet18(pretrained=args.pretrained)
    elif args.arch == 'resnet34':
        model = resnet34(pretrained=args.pretrained)
        if args.img_size == 64 or args.img_size == 128:
            # Change first layer
            conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
            n = conv1.kernel_size[0] * conv1.kernel_size[1] * conv1.out_channels
            conv1.weight.data.normal_(0, math.sqrt(2. / n))
            model.conv1 = conv1
    elif args.arch == 'resnet50':
        model = resnet50(pretrained=args.pretrained)
        if args.img_size == 64 or args.img_size == 128:
            # Change first layer
            conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
            n = conv1.kernel_size[0] * conv1.kernel_size[1] * conv1.out_channels
            conv1.weight.data.normal_(0, math.sqrt(2. / n))
            model.conv1 = conv1
    elif args.arch == 'resnet101':
        model = resnet101(pretrained=args.pretrained)
        if args.img_size == 64 or args.img_size == 128:
            # Change first layer
            conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
            n = conv1.kernel_size[0] * conv1.kernel_size[1] * conv1.out_channels
            conv1.weight.data.normal_(0, math.sqrt(2. / n))
            model.conv1 = conv1
    elif args.arch == 'resnet152':
        model = resnet152(pretrained=args.pretrained)
        if args.img_size == 64 or args.img_size == 128:
            # Change first layer
            conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
            n = conv1.kernel_size[0] * conv1.kernel_size[1] * conv1.out_channels
            conv1.weight.data.normal_(0, math.sqrt(2. / n))
            model.conv1 = conv1
    else:
        raise NotImplementedError

    dist_util.setup_dist(local_rank)


    checkpoint = None

    # use cuda
    model.to(dist_util.dev())

    if os.path.isfile(args.resume):
        if dist.get_rank() == 0:
            print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = dist_util.load_state_dict(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        if dist.get_rank() == 0:
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    else:
        print_rank("=> no checkpoint found at '{}'".format(args.resume))
        if os.path.isfile(last_checkpoint):
            print_rank("=> found last checkpoint")
            checkpoint = dist_util.load_state_dict(last_checkpoint)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print_rank("=> loaded checkpoint '{}' (epoch {})".format(last_checkpoint, checkpoint['epoch']))
        else:
            if dist.get_rank() == 0:
                print("=> no last checkpoint found")



    dist_util.sync_params(model.parameters())
    # model = torch.nn.parallel.DistributedDataParallel(model)
    model = DDP(
        model,
        device_ids=[dist_util.dev()],
        # output_device=dist_util.dev(),
        broadcast_buffers=False,
        # bucket_cap_mb=128,
        find_unused_parameters=False,
    )

    # define loss and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])

    # cudnn.benchmark = True

    # Data loading
    mini = args.local
    train_loader, val_loader = data_loader_gd(args.batch_size, args.workers, args.pin_memory, mini, args.img_size)

    if args.evaluate:
        validate(val_loader, model, criterion, args.print_freq, -1)
        return
    print_rank("Eval before training")
    best_prec1, _ =  validate(val_loader, model, criterion, args.print_freq, -1)
    print_rank("Start training")
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args.print_freq)

        # evaluate on validation set
        prec1, prec5 = validate(val_loader, model, criterion, args.print_freq, epoch)


        if dist.get_rank() == 0:
            # remember the best prec@1 and save checkpoint only for rank 0, because other ranks do not have many meaning
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)

            if ((epoch != 0) and (epoch%5 == 0)) or is_best:
                if is_best:
                    print_rank(f"saving new best accuracy {best_prec1}")
                else:
                    print_rank("saving freq checkpoint")
                save_checkpoint2({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.module.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer': optimizer.state_dict()
                }, is_best, model_name, log_dir)
                if hfai.client.receive_suspend_command():
                    hfai.client.go_suspend()
            elif hfai.client.receive_suspend_command():
                save_checkpoint2({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.module.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer': optimizer.state_dict()
                }, is_best, model_name, log_dir)
                print_rank("Complete saving checkpoint - client suspend. Good luck next run")
                hfai.client.go_suspend()


def train(train_loader, model, criterion, optimizer, epoch, print_freq):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    loss_acc, correct1, correct5, total = torch.zeros(4).cuda()
    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print_rank('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

    batch_num = i + 1
    loss_acc += losses.sum
    correct1 += (top1.sum / 100.0)
    correct5 += (top5.sum / 100.0)
    total += top1.count
    for x in [loss_acc, correct1, correct5, total]:
        dist.reduce(x, 0)
    if dist.get_rank() == 0:
        loss_train = loss_acc.item() / dist.get_world_size() / batch_num
        acc1 = 100 * correct1.item() / total.item()
        acc5 = 100 * correct5.item() / total.item()
        print(f'Epoch: {epoch}, Loss train: {loss_train}, Acc1: {acc1:.2f}%, Acc5: {acc5:.2f}%', flush=True)


def validate(val_loader, model, criterion, print_freq, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    loss_acc, correct1, correct5, total = torch.zeros(4).cuda()
    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)
        with torch.no_grad():
            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % print_freq == 0:
                print_rank('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
    batch_num = i + 1
    loss_acc += losses.sum
    correct1 += (top1.sum / 100.0)
    correct5 += (top5.sum / 100.0)
    total += top1.count
    for x in [loss_acc, correct1, correct5, total]:
        dist.reduce(x, 0)
    if dist.get_rank() == 0:
        loss_val = loss_acc.item() / dist.get_world_size() / batch_num
        acc1 = 100.0 * correct1.item() / total.item()
        acc5 = 100.0 * correct5.item() / total.item()
        print(f'Epoch: {epoch}, Loss: {loss_val}, Acc1: {acc1:.2f}%, Acc5: {acc5:.2f}%', flush=True)

    # print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    return correct1.item()/total.item(), correct5.item()/total.item()


def print_rank(message: str, main_rank=0):
    if dist.get_rank() == main_rank:
        print(message)



if __name__ == "__main__":
    ngpus = torch.cuda.device_count()
    hfai.multiprocessing.spawn(main, args=(), nprocs=ngpus, bind_numa=False)
