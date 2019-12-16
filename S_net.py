from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import numpy as np

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
        self.b = nn.Parameter(torch.zeros((input_caps, output_caps)))

    def forward(self, u_predict):
        batch_size, input_caps, output_caps, output_dim = u_predict.size()

        c = F.softmax(self.b)
        s = (c.unsqueeze(2) * u_predict).sum(dim=1)
        v = squash(s)

        if self.n_iterations > 0:
            b_batch = self.b.expand((batch_size, input_caps, output_caps))
            for r in range(self.n_iterations):
                v = v.unsqueeze(1)
                b_batch = b_batch + (u_predict * v).sum(-1)

                c = F.softmax(b_batch.view(-1, output_caps)).view(-1, input_caps, output_caps, 1)
                s = (c * u_predict).sum(dim=1)
                v = squash(s)

        return v


class CapsLayer(nn.Module):
    def __init__(self, input_caps, input_dim, output_caps, output_dim, routing_module):
        super(CapsLayer, self).__init__()
        self.input_dim = input_dim
        self.input_caps = input_caps
        self.output_dim = output_dim
        self.output_caps = output_caps
        self.weights = nn.Parameter(torch.Tensor(input_caps, input_dim, output_caps * output_dim))
        # l = 6
        # k = 4
        # self.pij = nn.Parameter(torch.Tensor(l,k)).cuda()
        # self.q_i = torch.Tensor(input_caps, input_dim, self.pij.size(0)).cuda()
        # self.w_j = torch.Tensor(self.pij.size(1), output_caps * output_dim).cuda()
        self.routing_module = routing_module
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.input_caps)
        #self.weights = self.q_i.matmul(self.pij).matmul(self.w_j)
        self.weights.data.uniform_(-stdv, stdv)

    def forward(self, caps_output):
        #print(caps_output.size())
        #print(self.weights.size())
        caps_output = caps_output.unsqueeze(2)
        u_predict = caps_output.matmul(self.weights)

        u_predict = u_predict.squeeze(2)
        A = u_predict
        B = A.permute(0, 2, 1)
        # print(A.size())
        # print(B.size())
        C = B.matmul(A).cpu().detach().numpy()
        e_vals, e_vecs = np.linalg.eig(C)
        # print('e_vals:',e_vals)
        sorted_indices = np.argsort(e_vals, axis=1)
        list = sorted_indices[:, -1]
        # print(sorted_indices)
        # print('list:',list)
        eval = np.zeros(list.shape)
        for i in range(list.shape[0]):
            eval[i] = e_vals[i][list[i]]
        eval = torch.from_numpy(eval).cuda()   #torch.Size([128])
        print(eval.size())




        v = v.view(v.size(0),self.output_caps, self.output_dim)

        #u_predict = u_predict.view(u_predict.size(0), self.input_caps, self.output_caps, self.output_dim)
        #print(u_predict.size())
        #v = self.routing_module(u_predict)
        #U,S,V = torch.svd(v)
        #print('v:',v,'S:',S)
        #print(v.size())
        return v


class PrimaryCapsLayer(nn.Module):
    def __init__(self, input_channels, output_caps, output_dim, kernel_size, stride):
        super(PrimaryCapsLayer, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_caps * output_dim, kernel_size=kernel_size, stride=stride)
        self.input_channels = input_channels
        self.output_caps = output_caps
        self.output_dim = output_dim

    def forward(self, input):
        out = self.conv(input)
        N, C, H, W = out.size()
        out = out.view(N, self.output_caps, self.output_dim, H, W)

        # will output N x OUT_CAPS x OUT_DIM
        out = out.permute(0, 1, 3, 4, 2).contiguous()

        # caps = out.squeeze(0)
        # caps = caps.permute(0, 3, 1, 2).contiguous()
        # caps = caps.view(36,caps.size(0),8)
        #
        # for i in range(caps.size(0)):
        #     A = caps[i]
        #     print(A.size())
        #     U,S,V = torch.svd(A)
        #     print(i,S)
        out = out.view(out.size(0), -1, out.size(4))
        out = squash(out)
        return out


class CapsNet(nn.Module):
    def __init__(self, routing_iterations, n_classes=10):
        super(CapsNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 256, kernel_size=9, stride=1)
        self.primaryCaps = PrimaryCapsLayer(256, 32, 8, kernel_size=9, stride=2)  # outputs 6*6
        self.num_primaryCaps = 32 * 6 * 6
        routing_module = AgreementRouting(self.num_primaryCaps, n_classes, routing_iterations)
        self.digitCaps = CapsLayer(self.num_primaryCaps, 8, n_classes, 16, routing_module)
        # routing_module_1 = AgreementRouting(self.num_primaryCaps, 32, routing_iterations)
        # routing_module_2 = AgreementRouting(32, n_classes, routing_iterations)
        # self.digitCaps_1 = CapsLayer(self.num_primaryCaps, 8, 32, 12, routing_module_1)
        # self.digitCaps_2 = CapsLayer(32, 12, n_classes, 16, routing_module_2)

    def forward(self, input):
        x = self.conv1(input)
        x = F.relu(x)
        #print(x.size())
        x = self.primaryCaps(x)
        #print(x.size())

        # caps = x.permute(1, 0, 2).contiguous()
        #
        # for i in range(caps.size(0)):
        #     A = caps[i]
        #     #print(A.size())
        #     U,S,V = torch.svd(A)
        #     print(i,S)
        x = self.digitCaps(x)
        #x = self.digitCaps_1(x)
        # caps = x.permute(1, 0, 2).contiguous()
        #
        # for i in range(caps.size(0)):
        #     A = caps[i]
        #     #print(A.size())
        #     U,S,V = torch.svd(A)
        #     print(i,S)
        #x = self.digitCaps_2(x)
        #print(x.size())

        # caps = x.permute(1, 0, 2).contiguous()
        #
        # for i in range(caps.size(0)):
        #     A = caps[i]
        #     #print(A.size())
        #     U,S,V = torch.svd(A)
        #     print(i,S)
        probs = x.pow(2).sum(dim=2).sqrt()
        #print('probs:',probs)
        return x, probs


class ReconstructionNet(nn.Module):
    def __init__(self, n_dim=16, n_classes=10):
        super(ReconstructionNet, self).__init__()
        self.fc1 = nn.Linear(n_dim * n_classes, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 784)
        self.n_dim = n_dim
        self.n_classes = n_classes

    def forward(self, x, target):
        mask = Variable(torch.zeros((x.size()[0], self.n_classes)), requires_grad=False)
        if next(self.parameters()).is_cuda:
            mask = mask.cuda()
        mask.scatter_(1, target.view(-1, 1), 1.)
        mask = mask.unsqueeze(2)
        x = x * mask
        x = x.view(-1, self.n_dim * self.n_classes)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x


class CapsNetWithReconstruction(nn.Module):
    def __init__(self, capsnet, reconstruction_net):
        super(CapsNetWithReconstruction, self).__init__()
        self.capsnet = capsnet
        self.reconstruction_net = reconstruction_net

    def forward(self, x, target):
        x, probs = self.capsnet(x)
        reconstruction = self.reconstruction_net(x, target)
        return reconstruction, probs


class MarginLoss(nn.Module):
    def __init__(self, m_pos, m_neg, lambda_):
        super(MarginLoss, self).__init__()
        self.m_pos = m_pos
        self.m_neg = m_neg
        self.lambda_ = lambda_

    def forward(self, lengths, targets, size_average=True):
        t = torch.zeros(lengths.size()).long()
        if targets.is_cuda:
            t = t.cuda()
        t = t.scatter_(1, targets.data.view(-1, 1), 1)
        targets = Variable(t)
        losses = targets.float() * F.relu(self.m_pos - lengths).pow(2) + \
                 self.lambda_ * (1. - targets.float()) * F.relu(lengths - self.m_neg).pow(2)
        return losses.mean() if size_average else losses.sum()


if __name__ == '__main__':

    import argparse
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.autograd import Variable

    # Training settings
    parser = argparse.ArgumentParser(description='CapsNet with MNIST')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
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

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Pad(2), transforms.RandomCrop(28),
                           transforms.ToTensor()
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor()
        ])),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)

    model = CapsNet(args.routing_iterations)

    if args.with_reconstruction:
        reconstruction_model = ReconstructionNet(16, 10)
        reconstruction_alpha = 0.0005
        model = CapsNetWithReconstruction(model, reconstruction_model)

    if args.cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=15, min_lr=1e-6)

    loss_fn = MarginLoss(0.9, 0.1, 0.5)


    def train(epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target, requires_grad=False)
            optimizer.zero_grad()
            if args.with_reconstruction:
                output, probs = model(data, target)
                reconstruction_loss = F.mse_loss(output, data.view(-1, 784))
                margin_loss = loss_fn(probs, target)
                loss = reconstruction_alpha * reconstruction_loss + margin_loss
            else:
                output, probs = model(data)
                loss = loss_fn(probs, target)
            loss.backward(retain_graph=True)
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))

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
                reconstruction_loss = F.mse_loss(output, data.view(-1, 784), size_average=False).item()
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


    for epoch in range(1, args.epochs + 1):
        start = time.process_time()
        train(epoch)
        elapsed = (time.process_time() - start)
        print("Time used:", elapsed)
        test_loss = test()
        scheduler.step(test_loss)
        torch.save(model.state_dict(),'{:03d}_model_dict_{}routing_reconstruction{}.pth'.format(epoch, args.routing_iterations,args.with_reconstruction))