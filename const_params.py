from torch.utils.tensorboard import SummaryWriter
# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/fashion_mnist_experiment_1')

# global parameters
DEFAULT_LR = 0.1
MATCHING_EPOCHES = 200
TRAIN_BATCH_SIZE = 128
TEST_BATCH_SIZE = 100

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')