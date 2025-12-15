import torch
torch.manual_seed(seed=42)
import torch.nn as nn

class CheckpointWrapper(torch.nn.Module):
    def __init__(self, module, signature=""):
        super().__init__()
        self.module = module
        self.signature = signature
        print(f"Checkpointing module {self.module.__class__.__name__}")

    def forward(self, *inputs):
        def forward_fn(*inputs):
            print(f"Executing {self.signature}")
            return self.module(*inputs)
        return torch.utils.checkpoint.checkpoint(forward_fn, *inputs, use_reentrant=True)

class LeNet(nn.Module):
    def __init__(self, num_classes=100):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=2)
            , nn.ReLU(inplace=False)
            , nn.MaxPool2d(kernel_size=2, stride=2)
            , nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
            , nn.ReLU(inplace=False)
            , nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # self.features = nn.Sequential(
        #     nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=2)
        #     , CheckpointWrapper(nn.ReLU(inplace=False), "nn.ReLU(inplace=False):0")
        #     , CheckpointWrapper(nn.MaxPool2d(kernel_size=2, stride=2), "nn.MaxPool2d(kernel_size=2, stride=2):1")
        #     , CheckpointWrapper(nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0), "nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0):2")
        #     , CheckpointWrapper(nn.ReLU(inplace=False), "nn.ReLU(inplace=False):3")
        #     , CheckpointWrapper(nn.MaxPool2d(kernel_size=2, stride=2), "nn.MaxPool2d(kernel_size=2, stride=2):4")
        # )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 6 * 6, 120)
            , CheckpointWrapper(nn.ReLU(inplace=False), "nn.ReLU(inplace=False):5")
            , CheckpointWrapper(nn.Linear(120, 84), "nn.Linear(120, 84):6")
            , CheckpointWrapper(nn.ReLU(inplace=False), "nn.ReLU(inplace=False):7")
            , CheckpointWrapper(nn.Linear(84, num_classes), "nn.Linear(84, num_classes):8")
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def lenet(num_classes=100):
    return LeNet(num_classes=num_classes)