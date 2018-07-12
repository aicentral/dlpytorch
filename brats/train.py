#!/usr/bin/env python3

from local import *
import time
import argparse
import torch

import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.transforms as transforms

from torch.utils.data import DataLoader

import torchbiomed.transforms as biotransforms
import torchbiomed.loss as bioloss
import torchbiomed.utils as utils

import os
import sys
import math

import shutil

import setproctitle

import vnet
import make_graph
from functools import reduce
import operator

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv3d') != -1:
        nn.init.kaiming_normal(m.weight)
        m.bias.data.zero_()

def datestr():
    now = time.gmtime()
    return '{}{:02}{:02}_{:02}{:02}'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)


def save_checkpoint(state, is_best, path, prefix, filename='checkpoint.pth.tar'):
    prefix_save = os.path.join(path, prefix)
    name = prefix_save + '_' + filename
    torch.save(state, name)
    if is_best:
        shutil.copyfile(name, prefix_save + '_model_best.pth.tar')


def inference(args, loader, model, transforms):
    src = args.inference
    dst = args.save
    model.eval()
    for i, data in enumerate(loader):
        data=np.array(data)[0]
        data=torch.from_numpy(data)
        data=data.type(torch.FloatTensor)
        shape = data.size()
        if args.cuda:
            data.pin_memory()
            data = data.cuda()
        data = Variable(data, volatile=True)
        output = model(data)
        _, output = output.max(1)
        output = output.view(shape)
        output = output.cpu()
        results = torch.squeeze(output.data).numpy().astype(np.int16)
        print("saving result {}".format(i))
        np.savez_compressed('/home/TRAINING/datasets/brats/{}/out{}'.format(dst,i), results)

# performing post-train inference:
# train.py --resume <model checkpoint> --i <input dataset> --save <output directory>

def noop(x):
    return x

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=4)
    parser.add_argument('--dice', action='store_true')
    parser.add_argument('--ngpu', type=int, default=1)
    parser.add_argument('--nEpochs', type=int, default=1)
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('-i', '--inference', default='', type=str, metavar='PATH',
                        help='run inference on data set and save results')

    # 1e-8 works well for lung masks but seems to prevent
    # rapid learning for nodule masks
    parser.add_argument('--weight-decay', '--wd', default=1e-8, type=float,
                        metavar='W', help='weight decay (default: 1e-8)')
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--save')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--opt', type=str, default='adam',
                        choices=('sgd', 'adam', 'rmsprop'))
    args = parser.parse_args()
    best_prec1 = 100.
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.save = args.save or 'work120/vnet.base.{}'.format(datestr())
    nll = True
    if args.dice:
        nll = False
    weight_decay = args.weight_decay
    setproctitle.setproctitle(args.save)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    print("build vnet")
    model = vnet.VNet(elu=False, nll=nll)
    batch_size = args.ngpu*args.batchSz
    gpu_ids = range(args.ngpu)
    if args.cuda:
        model = nn.parallel.DataParallel(model, device_ids=gpu_ids)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        model.apply(weights_init)

    if nll:
        train = train_nll
        test = test_nll
        class_balance = True
    else:
        train = train_dice
        test = test_dice
        class_balance = False

    print('  + Number of params: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    print('  + Cuda enabled=',args.cuda)
    if args.cuda:
        model = model.cuda()

    if os.path.exists(args.save):
        shutil.rmtree(args.save)
    os.makedirs(args.save, exist_ok=True)
    
    datadir='/home/TRAINING/datasets/brats/'
    if args.inference != '':
        if not args.resume:
            print("args.resume must be set to do inference")
            exit(1)
        kwargs = {'num_workers': 1} if args.cuda else {}
        src = args.inference; dst = args.save
        inference_batch_size = args.ngpu
        dataz=np.load(datadir+src) #'hgg_data120.npz')
        alldata= dataz.f.arr_0
        testdata=alldata[170:]
        dataset=[]
        for x in zip(testdata):
            x=np.expand_dims(x, axis=0)
            dataset.append(x)
        
        loader = DataLoader(dataset, batch_size=inference_batch_size, shuffle=False, collate_fn=noop, **kwargs)
        inference(args, loader, model, trainTransform)
        return

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    print("loading training set")
    dataz=np.load(datadir+'hgg_data120.npz')
    alldata= dataz.f.arr_0
    lblz=np.load(datadir+'hgg_labels120.npz')
    alllabels= lblz.f.arr_0
    alllabels[alllabels>0]=1  # for now , work only on 1 class 
    traindata=alldata[:170]
    testdata=alldata[170:]
    print('number of classes=',np.max(alllabels))
    trainSet=[];testSet=[]
    for x,y in zip(traindata,alllabels[:170]):
        x=np.expand_dims(x, axis=0)
        y=np.expand_dims(y, axis=0)
        trainSet.append((x,y))
    for x,y in zip(testdata,alllabels[170:]):
        x=np.expand_dims(x, axis=0)
        y=np.expand_dims(y, axis=0)
        testSet.append((x,y))

    trainLoader = DataLoader(trainSet, batch_size=batch_size, shuffle=True, **kwargs)
    print("loading test set")

    testLoader = DataLoader(testSet, batch_size=batch_size, shuffle=False, **kwargs)

    bg_weight = 0.011 # I calculated that somewhere else # target_mean / (1. + target_mean)
    fg_weight = 1. - bg_weight
    class_weights = torch.FloatTensor([bg_weight, fg_weight]) 
    if args.cuda:
        class_weights = class_weights.cuda()

    if args.opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=1e-1,
                              momentum=0.99, weight_decay=weight_decay)
    elif args.opt == 'adam':
        optimizer = optim.Adam(model.parameters(), weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), weight_decay=weight_decay)

    trainF = open(os.path.join(args.save, 'train.csv'), 'w')
    testF = open(os.path.join(args.save, 'test.csv'), 'w')
    err_best = 100.
    for epoch in range(1, args.nEpochs + 1):
        adjust_opt(args.opt, optimizer, epoch)
        train(args, epoch, model, trainLoader, optimizer, trainF, class_weights)
        err = test(args, epoch, model, testLoader, optimizer, testF, class_weights)
        is_best = False
        if err < best_prec1:
            is_best = True
            best_prec1 = err
        save_checkpoint({'epoch': epoch,
                         'state_dict': model.state_dict(),
                         'best_prec1': best_prec1},
                        is_best, args.save, "vnet")
        os.system('./plot.py {} {} &'.format(len(trainLoader), args.save))

    trainF.close()
    testF.close()


def train_nll(args, epoch, model, trainLoader, optimizer, trainF, weights):
    model.train()
    nProcessed = 0
    nTrain = len(trainLoader.dataset)
    for batch_idx, (data, target) in enumerate(trainLoader):
        data=data.type(torch.FloatTensor)
        target=target.type(torch.LongTensor)
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        target = target.view(target.numel())
        loss = F.nll_loss(output, target, weight=weights)
        dice_loss = bioloss.dice_error(output, target)
        # make_graph.save('/tmp/t.dot', loss.creator); assert(False)
        loss.backward()
        optimizer.step()
        nProcessed += len(data)
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        incorrect = pred.ne(target.data).cpu().sum()
        err = 100.*incorrect/target.numel()
        partialEpoch = epoch + batch_idx / len(trainLoader) - 1
        print('Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.4f}\tError: {:.3f}\t Dice: {:.6f}'.format(
            partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader),
            loss.data[0], err, dice_loss))

        trainF.write('{},{},{}\n'.format(partialEpoch, loss.data[0], err))
        trainF.flush()


def test_nll(args, epoch, model, testLoader, optimizer, testF, weights):
    model.eval()
    test_loss = 0
    dice_loss = 0
    incorrect = 0
    numel = 0
    for data, target in testLoader:
        data=data.type(torch.FloatTensor)
        target=target.type(torch.LongTensor)
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        target = target.view(target.numel())
        numel += target.numel()
        output = model(data)
        test_loss += F.nll_loss(output, target, weight=weights).data[0]
        dice_loss += bioloss.dice_error(output, target)
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        incorrect += pred.ne(target.data).cpu().sum()

    test_loss /= len(testLoader)  # loss function already averages over batch size
    dice_loss /= len(testLoader)
    err = 100.*incorrect/numel
    print('\nTest set: Average loss: {:.4f}, Error: {}/{} ({:.3f}%) Dice: {:.6f}\n'.format(
        test_loss, incorrect, numel, err, dice_loss))

    testF.write('{},{},{}\n'.format(epoch, test_loss, err))
    testF.flush()
    return err


def train_dice(args, epoch, model, trainLoader, optimizer, trainF, weights):
    model.train()
    nProcessed = 0
    nTrain = len(trainLoader.dataset)
    for batch_idx, (data, target) in enumerate(trainLoader):
        data=data.type(torch.FloatTensor)
        target=target.type(torch.LongTensor)
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = bioloss.dice_loss(output, target)
        # make_graph.save('/tmp/t.dot', loss.creator); assert(False)
        loss.backward()
        optimizer.step()
        nProcessed += len(data)
        err = 100.*(1. - loss.data[0])
        partialEpoch = epoch + batch_idx / len(trainLoader) - 1
        print('Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.8f}\tError: {:.8f}'.format(
            partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader),
            loss.data[0], err))

        trainF.write('{},{},{}\n'.format(partialEpoch, loss.data[0], err))
        trainF.flush()


def test_dice(args, epoch, model, testLoader, optimizer, testF, weights):
    model.eval()
    test_loss = 0
    incorrect = 0
    for data, target in testLoader:
        data=data.type(torch.FloatTensor)
        target=target.type(torch.LongTensor)
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        loss = bioloss.dice_loss(output, target).data[0]
        test_loss += loss
        incorrect += (1. - loss)

    test_loss /= len(testLoader)  # loss function already averages over batch size
    nTotal = len(testLoader)
    err = 100.*incorrect/nTotal
    print('\nTest set: Average Dice Coeff: {:.4f}, Error: {}/{} ({:.0f}%)\n'.format(
        test_loss, incorrect, nTotal, err))

    testF.write('{},{},{}\n'.format(epoch, test_loss, err))
    testF.flush()
    return err


def adjust_opt(optAlg, optimizer, epoch):
    if optAlg == 'sgd':
        if epoch < 150:
            lr = 1e-1
        elif epoch == 150:
            lr = 1e-2
        elif epoch == 225:
            lr = 1e-3
        else:
            return

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

if __name__ == '__main__':
    main()
