"""PyTorch DDP training script."""
# pylint: disable=line-too-long
# adapted from https://github.com/yangkky/distributed_tutorial/blob/master/src/mnist-distributed.py

import argparse
import os
from datetime import datetime

import torch
import torchvision
from torch import distributed as dist
from torch import multiprocessing as mp
from torch import nn
from torchvision import transforms


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--nodes",
    type=int,
    default=1,
    help="The number of nodes to use for distributed " "training",
  )
  parser.add_argument(
    "--nr",
    type=int,
    default=0,
    help="The rank of the node for multi-node distributed " "training",
  )
  parser.add_argument(
    "--gpus",
    type=int,
    default=1,
    help="The number of processes to launch on each node, "
         "for GPU training, this is recommended to be set "
         "to the number of GPUs in your system so that "
         "each process can be bound to a single GPU.",
  )
  parser.add_argument(
    "--master_addr",
    type=str,
    help="Master node (rank 0)'s address, should be either "
         "the IP address or the hostname of node 0, for "
         "single node multi-proc training, the "
         "--master_addr can simply be 127.0.0.1",
  )
  parser.add_argument(
    "--master_port",
    default='1234',
    type=str,
    help="Master node (rank 0)'s free port that needs to "
         "be used for communciation during distributed "
         "training",
  )
  parser.add_argument(
    '--epochs',
    default=2,
    type=int,
    metavar='N',
    help='number of total epochs to run'
  )

  args = parser.parse_args()
  args.world_size = args.gpus * args.nodes
  os.environ['MASTER_ADDR'] = args.master_addr
  os.environ['MASTER_PORT'] = args.master_port
  mp.spawn(train, args=(args,), nprocs=args.gpus)


class ConvNet(nn.Module):
  """ConvNet class."""
  def __init__(self, num_classes=10):
    super().__init__()
    self.layer1 = nn.Sequential(
      nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
      nn.BatchNorm2d(16),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2))
    self.layer2 = nn.Sequential(
      nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2))
    self.fc = nn.Linear(7 * 7 * 32, num_classes)

  def forward(self, x):
    out = self.layer1(x)
    out = self.layer2(out)
    out = out.reshape(out.size(0), -1)
    out = self.fc(out)
    return out


def train(pid, args):
  """Train model."""
  rank = args.nr * args.gpus + pid
  dist.init_process_group(backend='nccl', init_method='env://',
                          world_size=args.world_size, rank=rank)
  torch.manual_seed(0)
  model = ConvNet()
  torch.cuda.set_device(pid)
  model.cuda(pid)
  batch_size = 100
  # define loss function (criterion) and optimizer
  criterion = nn.CrossEntropyLoss().cuda(pid)
  optimizer = torch.optim.SGD(model.parameters(), 1e-4)
  # Wrap the model
  model = nn.parallel.DistributedDataParallel(model, device_ids=[pid])
  # Data loading code
  train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    transform=transforms.ToTensor(),
    download=True
  )
  train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset,
    num_replicas=args.world_size,
    rank=rank
  )
  train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    sampler=train_sampler
  )

  start = datetime.now()
  total_step = len(train_loader)
  for epoch in range(args.epochs):
    for i, (images, labels) in enumerate(train_loader):
      images = images.cuda(non_blocking=True)
      labels = labels.cuda(non_blocking=True)
      # Forward pass
      outputs = model(images)
      loss = criterion(outputs, labels)

      # Backward and optimize
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      if (i + 1) % 100 == 0 and pid == 0:
        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1,
                                                                 args.epochs,
                                                                 i + 1,
                                                                 total_step,
                                                                 loss.item()))
  if pid == 0:
    print("Training complete in: " + str(datetime.now() - start))


if __name__ == '__main__':
  main()
