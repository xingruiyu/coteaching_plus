# -*- coding:utf-8 -*-
from __future__ import print_function 
import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from data.cifar import CIFAR10, CIFAR100
from data.mnist import MNIST
from data.newsgroups import NewsGroups
from data.torchlist import ImageFilelist
from model import MLPNet, CNN_small, CNN, NewsNet
from preact_resnet import PreActResNet18
import argparse, sys
import numpy as np
import datetime
import shutil

from loss import loss_coteaching, loss_coteaching_plus

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type = float, default = 0.001)
parser.add_argument('--result_dir', type = str, help = 'dir to save result txt files', default = 'results/')
parser.add_argument('--noise_rate', type = float, help = 'corruption rate, should be less than 1', default = 0.2)
parser.add_argument('--forget_rate', type = float, help = 'forget rate', default = None)
parser.add_argument('--noise_type', type = str, help='[pairflip, symmetric]', default='symmetric')
parser.add_argument('--num_gradual', type = int, default = 10, help='how many epochs for linear drop rate. This parameter is equal to Ek for lambda(E) in the paper.')
parser.add_argument('--dataset', type = str, help = 'mnist, cifar10, cifar100, or imagenet_tiny', default = 'mnist')
parser.add_argument('--n_epoch', type=int, default=200)
parser.add_argument('--optimizer', type = str, help='[adam, sgd_imagenet_tiny]', default='adam')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--print_freq', type=int, default=50)
parser.add_argument('--num_workers', type=int, default=2, help='how many subprocesses to use for data loading')
parser.add_argument('--epoch_decay_start', type=int, default=80)
parser.add_argument('--model_type', type = str, help='[coteaching, coteaching_plus]', default='coteaching_plus')
parser.add_argument('--fr_type', type = str, help='forget rate type, selected from [linear_first, dual_linear]', default='linear_first')

args = parser.parse_args()

# Seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Hyper Parameters
batch_size = 128
learning_rate = args.lr 

# load dataset
if args.dataset=='mnist':
    input_channel = 1
    init_epoch = 0
    num_classes = 10
    args.n_epoch = 200
    train_dataset = MNIST(root='./data/',
                                download=True,  
                                train=True, 
                                transform=transforms.ToTensor(),
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate
                                )
    
    test_dataset = MNIST(root='./data/',
                               download=True,  
                               train=False, 
                               transform=transforms.ToTensor(),
                               noise_type=args.noise_type,
                               noise_rate=args.noise_rate
                                )
    
if args.dataset=='cifar10':
    input_channel=3
    init_epoch = 20
    num_classes = 10
    args.n_epoch = 200
    train_dataset = CIFAR10(root='./data/',
                                download=True,  
                                train=True, 
                                transform=transforms.ToTensor(),
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate
                                )
    
    test_dataset = CIFAR10(root='./data/',
                                download=True,  
                                train=False, 
                                transform=transforms.ToTensor(),
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate
                                )

if args.dataset=='cifar100':
    input_channel=3
    init_epoch = 5
    num_classes = 100
    args.n_epoch = 200
    train_dataset = CIFAR100(root='./data/',
                                download=True,  
                                train=True, 
                                transform=transforms.ToTensor(),
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate
                                )
    
    test_dataset = CIFAR100(root='./data/',
                                download=True,  
                                train=False, 
                                transform=transforms.ToTensor(),
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate
                                )


if args.dataset=='news':
    init_epoch=0
    train_dataset = NewsGroups(root='./data/',
                                dataset_type='multiple',  
                                train=True, 
                                transform=transforms.ToTensor(),
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate
                                )
    
    test_dataset = NewsGroups(root='./data/',
                               dataset_type='multiple',  
                               train=False, 
                               transform=transforms.ToTensor(),
                               noise_type=args.noise_type,
                               noise_rate=args.noise_rate
                                )
    num_classes=train_dataset.num_classes
 
if args.dataset == 'imagenet_tiny':
    init_epoch = 100
    #data_root = '/home/xingyu/Data/phd/data/imagenet-tiny/tiny-imagenet-200'
    data_root = 'data/imagenet-tiny/tiny-imagenet-200'
    train_kv = "train_noisy_%s_%s_kv_list.txt" % (args.noise_type, args.noise_rate) 
    test_kv = "val_kv_list.txt"

    normalize = transforms.Normalize(mean=[0.4802, 0.4481, 0.3975], 
                                     std =[0.2302, 0.2265, 0.2262])

    train_dataset = ImageFilelist(root=data_root, flist=os.path.join(data_root, train_kv),
               transform=transforms.Compose([transforms.RandomResizedCrop(56),
               transforms.RandomHorizontalFlip(),
               transforms.ToTensor(), 
               normalize,
       ]))

    test_dataset = ImageFilelist(root=data_root, flist=os.path.join(data_root, test_kv),
               transform=transforms.Compose([transforms.Resize(64),
               transforms.CenterCrop(56),
               transforms.ToTensor(),
               normalize,
       ]))

if args.forget_rate is None:
    forget_rate=args.noise_rate
else:
    forget_rate=args.forget_rate

if args.dataset == 'imagenet_tiny':
    noise_or_not = np.load(os.path.join(data_root, 'noise_or_not_%s_%s.npy' %(args.noise_type, args.noise_rate)))
else:
    noise_or_not = train_dataset.noise_or_not

# Adjust learning rate and betas for Adam Optimizer
mom1 = 0.9
mom2 = 0.1
alpha_plan = [learning_rate] * args.n_epoch
beta1_plan = [mom1] * args.n_epoch
for i in range(args.epoch_decay_start, args.n_epoch):
    alpha_plan[i] = float(args.n_epoch - i) / (args.n_epoch - args.epoch_decay_start) * learning_rate
    beta1_plan[i] = mom2

def adjust_learning_rate(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr']=alpha_plan[epoch]
        param_group['betas']=(beta1_plan[epoch], 0.999) 
       
def adjust_learning_rate_imagenet_tiny(optimizer, epoch):
    """decrease the learning rate at 100 and 150 epoch"""
    lr = args.lr
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# define drop rate schedule
def gen_forget_rate(fr_type='linear_first'):
    # define drop rate schedule
    if fr_type=='linear_first':
        # 1. linear in the first 20 epoch, then constant (noise_rate).
        #              _______
        #           /
        rate_schedule = np.ones(args.n_epoch)*forget_rate
        rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate, args.num_gradual)

    if fr_type=='dual_linear':
        # 2. dual linear
        #       /(smaller slop)
        #      /(larger slop)
        rate_schedule = np.ones(args.n_epoch)*forget_rate
        rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate, args.num_gradual) 
        rate_schedule[args.num_gradual:] = np.linspace(forget_rate, 2*forget_rate, args.n_epoch-args.num_gradual)
        
    return rate_schedule

rate_schedule = gen_forget_rate(args.fr_type)
  
save_dir = args.result_dir +'/' +args.dataset+'/%s/' % args.model_type

if not os.path.exists(save_dir):
    os.system('mkdir -p %s' % save_dir)

model_str = args.dataset + '_%s_' % args.model_type + args.noise_type + '_' + str(args.noise_rate)

txtfile = save_dir + "/" + model_str + ".txt"
nowTime=datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
if os.path.exists(txtfile):
    os.system('mv %s %s' % (txtfile, txtfile+".bak-%s" % nowTime))


def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

# Train the Model
def train(train_loader,epoch, model1, optimizer1, model2, optimizer2):
    print('Training %s...' % model_str)
    
    train_total=0
    train_correct=0 
    train_total2=0
    train_correct2=0 

    for i, (data, labels, indexes) in enumerate(train_loader):
        ind=indexes.cpu().numpy().transpose()
      
        labels = Variable(labels).cuda()
        
        if args.dataset=='news':
            data = Variable(data.long()).cuda()
        else:
            data = Variable(data).cuda()
        # Forward + Backward + Optimize
        logits1=model1(data)
        prec1,  = accuracy(logits1, labels, topk=(1, ))
        train_total+=1
        train_correct+=prec1

        logits2 = model2(data)
        prec2,  = accuracy(logits2, labels, topk=(1, ))
        train_total2+=1
        train_correct2+=prec2
        if epoch < init_epoch:
            loss_1, loss_2, _, _ = loss_coteaching(logits1, logits2, labels, rate_schedule[epoch], ind, noise_or_not)
        else:
            if args.model_type=='coteaching_plus':
                loss_1, loss_2, _, _ = loss_coteaching_plus(logits1, logits2, labels, rate_schedule[epoch], ind, noise_or_not, epoch*i)

        optimizer1.zero_grad()
        loss_1.backward()
        optimizer1.step()
        optimizer2.zero_grad()
        loss_2.backward()
        optimizer2.step()
        if (i+1) % args.print_freq == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Training Accuracy1: %.4F, Training Accuracy2: %.4f, Loss1: %.4f, Loss2: %.4f' 
                  %(epoch+1, args.n_epoch, i+1, len(train_dataset)//batch_size, prec1, prec2, loss_1.item(), loss_2.item()))

    train_acc1=float(train_correct)/float(train_total)
    train_acc2=float(train_correct2)/float(train_total2)
    return train_acc1, train_acc2

# Evaluate the Model
def evaluate(test_loader, model1, model2):
    print('Evaluating %s...' % model_str)
    model1.eval()    # Change model to 'eval' mode.
    correct1 = 0
    total1 = 0
    for data, labels, _ in test_loader:
        if args.dataset=='news':
            data = Variable(data.long()).cuda()
        else:
            data = Variable(data).cuda()
        logits1 = model1(data)
        outputs1 = F.softmax(logits1, dim=1)
        _, pred1 = torch.max(outputs1.data, 1)
        total1 += labels.size(0)
        correct1 += (pred1.cpu() == labels.long()).sum()

    model2.eval()    # Change model to 'eval' mode 
    correct2 = 0
    total2 = 0
    for data, labels, _ in test_loader:
        if args.dataset=='news':
            data = Variable(data.long()).cuda()
        else:
            data = Variable(data).cuda()
        logits2 = model2(data)
        outputs2 = F.softmax(logits2, dim=1)
        _, pred2 = torch.max(outputs2.data, 1)
        total2 += labels.size(0)
        correct2 += (pred2.cpu() == labels.long()).sum()
 
    acc1 = 100*float(correct1)/float(total1)
    acc2 = 100*float(correct2)/float(total2)
    return acc1, acc2

def main():
    # Data Loader (Input Pipeline)
    print('loading dataset...')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size, 
                                               num_workers=args.num_workers,
                                               drop_last=True,
                                               shuffle=True)
    
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size, 
                                              num_workers=args.num_workers,
                                              drop_last=True,
                                              shuffle=False)
    # Define models
    print('building model...')
    if args.dataset == 'mnist':
        clf1 = MLPNet()
    if args.dataset == 'cifar10':
        clf1 = CNN_small(num_classes)
    if args.dataset == 'cifar100':
        clf1 = CNN(n_outputs=num_classes)
    if args.dataset=='news':
        clf1 = NewsNet(weights_matrix=train_dataset.weights_matrix, context_size=1000, hidden_size=300, num_classes=num_classes)
    if args.dataset=='imagenet_tiny':
        clf1 = PreActResNet18(num_classes=200)

    clf1.cuda()
    print(clf1.parameters)
    if args.optimizer == 'adam':
        optimizer1 = torch.optim.Adam(clf1.parameters(), lr=learning_rate)
    if args.optimizer == 'sgd_imagenet_tiny':
        optimizer1 = torch.optim.SGD(clf1.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    
    if args.dataset == 'mnist':
        clf2 = MLPNet()
    if args.dataset == 'cifar10':
        clf2 = CNN_small(num_classes)
    if args.dataset == 'cifar100':
        clf2 = CNN(n_outputs=num_classes)
    if args.dataset=='news':
        clf2 = NewsNet(weights_matrix=train_dataset.weights_matrix, context_size=1000, hidden_size=300, num_classes=num_classes)
    if args.dataset=='imagenet_tiny':
        clf2 = PreActResNet18(num_classes=200)

    clf2.cuda()
    print(clf2.parameters)
    if args.optimizer == 'adam':
        optimizer2 = torch.optim.Adam(clf2.parameters(), lr=learning_rate)
    if args.optimizer == 'sgd_imagenet_tiny':
        optimizer2 = torch.optim.SGD(clf2.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)

    with open(txtfile, "a") as myfile:
        myfile.write('epoch train_acc1 train_acc2 test_acc1 test_acc2\n')

    epoch=0
    train_acc1=0
    train_acc2=0
    # evaluate models with random weights
    test_acc1, test_acc2=evaluate(test_loader, clf1, clf2)
    print('Epoch [%d/%d] Test Accuracy on the %s test data: Model1 %.4f %% Model2 %.4f %%' % (epoch+1, args.n_epoch, len(test_dataset), test_acc1, test_acc2))
    # save results
    with open(txtfile, "a") as myfile:
        myfile.write(str(int(epoch)) + ' '  + str(train_acc1) +' '  + str(train_acc2) +' '  + str(test_acc1) + " " + str(test_acc2)  + "\n")

    # training
    for epoch in range(1, args.n_epoch):
        # train models
        clf1.train()
        clf2.train()
        if args.optimizer == 'adam':
            adjust_learning_rate(optimizer1, epoch)
            adjust_learning_rate(optimizer2, epoch)
        if args.optimizer == 'sgd_imagenet_tiny':
            adjust_learning_rate_imagenet_tiny(optimizer1, epoch)
            adjust_learning_rate_imagenet_tiny(optimizer2, epoch)
        train_acc1, train_acc2 = train(train_loader, epoch, clf1, optimizer1, clf2, optimizer2)
        # evaluate models
        test_acc1, test_acc2 = evaluate(test_loader, clf1, clf2)
        # save results
        print('Epoch [%d/%d] Test Accuracy on the %s test data: Model1 %.4f %% Model2 %.4f %%' % (epoch+1, args.n_epoch, len(test_dataset), test_acc1, test_acc2))
        with open(txtfile, "a") as myfile:
            myfile.write(str(int(epoch)) + ' '  + str(train_acc1) +' '  + str(train_acc2) +' '  + str(test_acc1) + " " + str(test_acc2) + "\n")

if __name__=='__main__':
    main()
