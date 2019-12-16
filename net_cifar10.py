from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time

from torch.optim import lr_scheduler
from torch.autograd import Variable


def squash(x):
    lengths2 = x.pow(2).sum(dim=2)
    lengths = lengths2.sqrt()
    x = x * (lengths2 / (1 + lengths2) / lengths).view(x.size(0), x.size(1), 1)
    return x


class AgreementRouting(nn.Module):
    def __init__(self, input_caps, output_caps, n_iterations):
        super(AgreementRouting, self).__init__()
        self.n_iterations = n_iterations
        self.b = nn.Parameter(torch.zeros((1, output_caps)))

    def forward(self, u_predict):
        batch_size, input_caps, output_caps, output_dim = u_predict.size()

        c = F.softmax(self.b)
        c = c.expand(input_caps,output_caps)
        #print(type(c))
        #print(c)
        s = (c.unsqueeze(2) * u_predict).sum(dim=1)
        v = squash(s)

        if self.n_iterations > 0:
            b_batch = self.b.expand((batch_size, 1, output_caps))
            for r in range(self.n_iterations):
                v = v.unsqueeze(1)
                b_batch = b_batch + (u_predict * v).sum(-1)

                c = F.softmax(b_batch.view(-1, output_caps)).view(-1, input_caps, output_caps, 1)
                s = (c * u_predict).sum(dim=1)
                v = squash(s)
        return v

###############################
class FastCapsLayer(nn.Module):
    def __init__(self, input_caps, input_dim, output_caps, output_dim, routing_module):
        super(FastCapsLayer, self).__init__()
        self.input_dim = input_dim
        self.input_caps = input_caps
        self.output_dim = output_dim
        self.output_caps = output_caps
        self.weights = nn.Parameter(torch.Tensor(input_caps, input_dim, output_caps * output_dim))  #W_ij（1152，8，10*16）
        self.routing_module = routing_module
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.input_caps)  #这个sqrt括号里的这个数是怎么确定的？
        self.weights.data.uniform_(-stdv, stdv) #参数初始化

    def forward(self, caps_output):     #这里的caps_output就是PrimaryCaps中得到的out,Size(N,32*28*28,8)=(128,25088,8)
        #print(caps_output.size())                   torch.Size([128, 1152, 8])
        caps_output = caps_output.unsqueeze(2)      #在caps_output的第二维增加一个维度
        #print(caps_output.size())                   #torch.Size([128, 1152, 1, 8])
        #print(self.weights.size())
        u_predict = caps_output.matmul(self.weights)        #torch.Size([128, 1152, 1, 160])
        u_predict = u_predict.view(u_predict.size(0), self.input_caps, self.output_caps, self.output_dim)   #(u.size(0),1152,10,16) u.size(0)也就是batch_size  #torch.Size([128, 36, 10, 16])
        v = self.routing_module(u_predict)

        probs = v.pow(2).sum(dim=2).sqrt()  #求输出的L2范数，即预测的概率
        label = torch.argmax(probs,dim=1)
        label = label.view(label.size(0),1)
        #print(label)
        one_hot = torch.zeros(label.size(0),10).scatter_(1, label.cpu(), 1)
        reconst_targets1 = one_hot.view(one_hot.size(0),1,1,10)
        reconst_targets1 = reconst_targets1.expand(reconst_targets1.size(0),8,8,10)
        reconst_targets1 = reconst_targets1.permute(0, 3, 1, 2).contiguous()
        #print(reconst_targets1.size())      #torch.Size([128, 8, 8, 10])
        reconst_targets2 = u_predict.view(u_predict.size(0),8,8,-1)
        reconst_targets2 = reconst_targets2.permute(0, 3, 1, 2).contiguous()
        reconst_targets3 = torch.cat((reconst_targets1.cpu(),reconst_targets2.cpu()),1)
        reconst_targets3 = reconst_targets3.cuda()
        return v,reconst_targets3
##############################


class PrimaryCapsLayer(nn.Module):
    def __init__(self, input_channels, output_caps, output_dim, kernel_size, stride):
        super(PrimaryCapsLayer, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_caps * output_dim, kernel_size=kernel_size, stride=stride)
        self.input_channels = input_channels
        self.output_caps = output_caps
        self.output_dim = output_dim

    def forward(self, input):
        out = self.conv(input)
        N, C, H, W = out.size()             #N,C,H,W=batch,in_channels,in_height,in_width,最后两项分别对应的是图像的高和宽
        out = out.view(N, self.output_caps, self.output_dim, H, W)  #output_caps=32 output_dim=8

        # will output N x OUT_CAPS x OUT_DIM
        out = out.permute(0, 1, 3, 4, 2).contiguous()   #permute函数，将tensor的维度换位，因为conv要求输入的类型是NHWC
        out = out.view(out.size(0), -1, out.size(4))    #view函数类似于resize     (N,1152,8)*****这里的1152哪里来的？
        out = squash(out)                               #squashing操作不改变size
        return out


class CapsNet(nn.Module):
    def __init__(self, routing_iterations, n_classes=10):   #n_classes输出胶囊数量，10个，对应10个类别
        super(CapsNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 256, kernel_size=9, stride=1) #卷积核是9*9的
        self.primaryCaps = PrimaryCapsLayer(256, 1, 256, kernel_size=9, stride=2)  # 256个输入通道，32个输出胶囊，8是每个胶囊的维度，每个胶囊尺寸是8*8
        self.num_primaryCaps = 1 * 8 * 8   #卷积层胶囊数量
        routing_module = AgreementRouting(self.num_primaryCaps, n_classes, routing_iterations)
        self.digitCaps = FastCapsLayer(self.num_primaryCaps, 256, n_classes, 16, routing_module)  #分别是输入胶囊数量，输入胶囊的维度，输出胶囊的数量，输出胶囊的维度

    def forward(self, input):
        x = self.conv1(input)   #输入数据（28*28的图片）
        x = F.relu(x)           #relu激活函数
        x = self.primaryCaps(x) #经过主胶囊层
        x, y = self.digitCaps(x)   #经过数字胶囊层
        probs = x.pow(2).sum(dim=2).sqrt()  #求输出的L2范数，即预测的概率
        return y, probs

#######################################
class ConvReconstructionNet(nn.Module):
    def __init__(self, n_dim=16, n_classes=10):
        super(ConvReconstructionNet, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(170, 16, 4, stride=2,padding=1)
        self.deconv2 = nn.ConvTranspose2d(16, 3, 4, stride=2,padding=1)


    def forward(self, x, target):
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = x.view(-1,1024)
        #print(x.size())
        return x
#######################################

##############
class CapsNetWithConvReconstruction(nn.Module):
    def __init__(self, capsnet, convreconstruction_net):
        super(CapsNetWithConvReconstruction, self).__init__()
        self.capsnet = capsnet
        self.convreconstruction_net = convreconstruction_net

    def forward(self, x, target):
        y, probs = self.capsnet(x)
        reconstruction = self.convreconstruction_net(y, target)
        return reconstruction, probs
#############

class MarginLoss(nn.Module):        #损失函数
    def __init__(self, m_pos, m_neg, lambda_):
        super(MarginLoss, self).__init__()
        self.m_pos = m_pos      #m+
        self.m_neg = m_neg      #m-
        self.lambda_ = lambda_  #λ

    def forward(self, lengths, targets, size_average=True):
        t = torch.zeros(lengths.size()).long()
        if targets.is_cuda:
            t = t.cuda()
        t = t.scatter_(1, targets.data.view(-1, 1), 1)      #scatter是一个填充的操作
        targets = Variable(t)                               #targets就是T_k
        losses = targets.float() * F.relu(self.m_pos - lengths).pow(2) + \
                 self.lambda_ * (1. - targets.float()) * F.relu(lengths - self.m_neg).pow(2)    #使用relu确保>0，论文中是Max(0,x)
        return losses.mean() if size_average else losses.sum()   #啥时候用Mean啥时候用sum？


if __name__ == '__main__':      #主函数
    start = time.process_time()
    import argparse
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.autograd import Variable

    # Training settings
    parser = argparse.ArgumentParser(description='CapsNet with CIFAR10')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--routing_iterations', type=int, default=3)
    parser.add_argument('--with_reconstruction', action='store_true', default=True)
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
#训练数据载入
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Pad(2), transforms.RandomCrop(32),
                           transforms.ToTensor()
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
#测试数据载入
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor()
        ])),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)

    model = CapsNet(args.routing_iterations)
#开始计时
    start = time.process_time()

    if args.with_reconstruction:
        reconstruction_model = ConvReconstructionNet(16, 10)
        reconstruction_alpha = 0.0005
        model = CapsNetWithConvReconstruction(model, reconstruction_model)

    if args.cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=15, min_lr=1e-6)

    loss_fn = MarginLoss(0.9, 0.1, 0.5)

#训练过程
    def train(epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target, requires_grad=False)
            optimizer.zero_grad()
            if args.with_reconstruction:
                output, probs = model(data, target)
                #print(output.size())
                reconstruction_loss = F.mse_loss(output, data.view(-1, 1024))
                margin_loss = loss_fn(probs, target)
                loss = reconstruction_alpha * reconstruction_loss + margin_loss
            else:
                output, probs = model(data)
                loss = loss_fn(probs, target)
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))
#测试过程
    def test():
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            with torch.no_grad():
                data, target = Variable(data), Variable(target)

            if args.with_reconstruction:
                output, probs = model(data, target)
                reconstruction_loss = F.mse_loss(output, data.view(-1, 1024), size_average=False).item()
                test_loss += loss_fn(probs, target, size_average=False).item()
                test_loss += reconstruction_alpha * reconstruction_loss
            else:
                output, probs = model(data)
                test_loss += loss_fn(probs, target, size_average=False).item()

            pred = probs.data.max(1, keepdim=True)[1]  # get the index of the max probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        return test_loss

#具体地训练以及结果显示
    for epoch in range(1, args.epochs + 1):

        train(epoch)

        test_loss = test()
        scheduler.step(test_loss)
        torch.save(model.state_dict(),
                   '{:03d}_model_dict_{}routing_reconstruction{}.pth'.format(epoch, args.routing_iterations,
                                                                             args.with_reconstruction))

    elapsed = (time.process_time() - start)
    print("Time used:",elapsed)
