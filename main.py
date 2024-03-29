'''Train CIFAR10 with PyTorch.'''
import argparse
import os
import time
import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
import torchvision.transforms as transforms

from models import *
from utils import format_time
from utils import progress_bar

from tensorboardutils import matplotlib_imshow
from tensorboardutils import select_n_random
from tensorboardutils import plot_classes_preds
from tensorboardutils import add_pr_curve_tensorboard

from const_params import *

import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=DEFAULT_LR, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=2)



# Model
print('==> Building model..')
# net = VGG('VGG19')
net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = densenet_cifar()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])

    print("checkpoint:")
    best_acc = checkpoint['acc']
    print("last_best_acc:", best_acc)
    start_epoch = checkpoint['epoch']
    print("start_epoch:", start_epoch)

    if args.lr != DEFAULT_LR:
        print("new_lr:", args.lr)
    elif args.lr == DEFAULT_LR:  # because lr is getting smaller, we don't consider the situation that we set lr back to 0.1
        args.lr = checkpoint['lr']  # and if lr we get is 0.1 ,we see it as default value.
        print("last_lr:", args.lr)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10)

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# create grid of images
img_grid = torchvision.utils.make_grid(images)

# show images
# matplotlib_imshow(img_grid, one_channel=False)

img_grid_unnorm = img_grid /2 +0.5

# write to tensorboard
writer.add_image('four_fashion_mnist_images', img_grid_unnorm)

writer.add_graph(net, images)   #TODO:不知道为什么这个net在TensorBoard中是横着的。
writer.close()

# select random images and their target indices
images, labels = select_n_random(trainset.data, trainset.targets)

# get the class labels for each image
class_labels = [classes[lab] for lab in labels]

# log embeddings
features = images.reshape(-1, 32 * 32 * 3)
writer.add_embedding(features,
                    metadata=class_labels,
                    label_img=torch.transpose(images,1,3))
writer.close()

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        if batch_idx % 100 == 99:  # every 100 mini-batches...

            # ...log the running loss
            writer.add_scalar('training loss',
                              train_loss / 100,
                              epoch * len(trainloader) + batch_idx)

            # ...log a Matplotlib Figure showing the model's predictions on a
            # random mini-batch
            writer.add_figure('predictions vs. actuals',
                              plot_classes_preds(net, inputs, targets,classes),
                              global_step=epoch * len(trainloader) + batch_idx)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    # 1. gets the probability predictions in a test_size x num_classes Tensor
    # 2. gets the preds in a test_size Tensor
    # takes ~10 seconds to run
    class_probs = []
    class_preds = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()

            _, predicted = outputs.max(1)
            probability =  [F.softmax(el, dim=0) for el in outputs]
            class_probs.append(probability)
            class_preds.append(predicted)

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
    test_preds = torch.cat(class_preds)

    # plot all the pr curves
    for i in range(len(classes)):
        add_pr_curve_tensorboard(i, test_probs, test_preds,classes)

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'lr': optimizer.param_groups[0]["lr"],
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

print(net)
timer1 = time.perf_counter()
for epoch in range(start_epoch, 1 + start_epoch + MATCHING_EPOCHES):
    timer0 = time.process_time()
    train(epoch)
    test(epoch)
    print("best_acc:", best_acc, "%% &cost_time:%f s" % (time.process_time() - timer0), " lr:",
          optimizer.param_groups[0]["lr"])
    scheduler.step(best_acc)
print("total_cost_time:", format_time(time.perf_counter() - timer1), "s")

# 【】学起来每次改代码用git
# 【】保存一个自己修改过的完美的代码。
# 【】使用预训练模型
