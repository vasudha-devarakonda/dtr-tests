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
            # print(f"Executing {self.signature}")
            return self.module(*inputs)
        return torch.utils.checkpoint.checkpoint(forward_fn, *inputs, use_reentrant=True)
    
class AlexNet(nn.Module):
    def __init__(self, num_classes=100):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # was 11x11, now 3x3
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32 -> 16
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16 -> 8
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 8 -> 41
            
            
            # nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
            # , nn.ReLU(inplace=False)
            # , nn.MaxPool2d(kernel_size=2, stride=2)
            # , CheckpointWrapper(nn.Conv2d(64, 192, kernel_size=3, padding=1), signature="conv2")
            # , nn.ReLU(inplace=False)
            # , nn.MaxPool2d(kernel_size=2, stride=2)
            # , CheckpointWrapper(nn.Conv2d(192, 384, kernel_size=3, padding=1), signature="conv3")
            # , nn.ReLU(inplace=False)
            # , CheckpointWrapper(nn.Conv2d(384, 256, kernel_size=3, padding=1), signature="conv4")
            # , nn.ReLU(inplace=False)
            # , CheckpointWrapper(nn.Conv2d(256, 256, kernel_size=3, padding=1), signature="conv5")
            # , nn.ReLU(inplace=False)
            # , nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(inplace=False),
            # nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=False),
            nn.Linear(512, num_classes),
            
            # CheckpointWrapper(nn.Linear(256 * 4 * 4, 1024), signature="fc1"),
            # CheckpointWrapper(nn.ReLU(inplace=False), signature="relu1"),
            # CheckpointWrapper(nn.Linear(1024, 512), signature="fc2"),
            # CheckpointWrapper(nn.ReLU(inplace=False), signature="relu2"),
            # nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        # size = x.storage().nbytes()
        # signature = x.storage().signature()
        # print(f"Input tensor size : {size}")
        # print(f"Input tensor signature : {signature}")
        x = self.classifier(x)
        # x = torch.utils.checkpoint.checkpoint(self.classifier, x, use_reentrant=True)
        return x
    
    # def __init__(self, num_classes=100):
    #     super(AlexNet, self).__init__()
    #     self.features1 = nn.Sequential(
    #         nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
    #         nn.ReLU(inplace=False),
    #         nn.MaxPool2d(kernel_size=2, stride=2)  # 32 -> 16
    #     )
        
    #     self.features2 = nn.Sequential(
    #         nn.Conv2d(64, 192, kernel_size=3, padding=1),
    #         nn.ReLU(inplace=False),
    #         nn.MaxPool2d(kernel_size=2, stride=2)  # 16 -> 8
    #     )
        
    #     self.features3 = nn.Sequential(
    #         nn.Conv2d(192, 384, kernel_size=3, padding=1),
    #         nn.ReLU(inplace=False)
    #     )
        
    #     self.features4 = nn.Sequential(
    #         nn.Conv2d(384, 256, kernel_size=3, padding=1),
    #         nn.ReLU(inplace=False)
    #     )
        
    #     self.features5 = nn.Sequential(
    #         nn.Conv2d(256, 256, kernel_size=3, padding=1),
    #         nn.ReLU(inplace=False),
    #         nn.MaxPool2d(kernel_size=2, stride=2)  # 8 -> 4
    #     )
        
    #     self.feature12 = nn.Sequential(
    #         self.features1,
    #         self.features2
    #     )
        
    #     self.feature345 = nn.Sequential(
    #         self.features3,
    #         self.features4,
    #         self.features5
    #     )
        
    #     self.feature12345 = nn.Sequential(
    #         self.feature12,
    #         self.feature345
    #     )
        
        
    #     # self.classifier = nn.Sequential(
    #     #     # nn.Dropout(),
    #     #     nn.Linear(256 * 4 * 4, 1024),
    #     #     nn.ReLU(inplace=False),
    #     #     # nn.Dropout(),
    #     #     nn.Linear(1024, 512),
    #     #     nn.ReLU(inplace=False),
    #     #     nn.Linear(512, num_classes),
    #     # )
        
    #     self.classifier1 = nn.Sequential(
    #         # nn.Dropout(),
    #         nn.Linear(256 * 4 * 4, 1024),
    #         nn.ReLU(inplace=False),
    #     )
        
    #     self.classifier2 = nn.Sequential(
    #         # nn.Dropout(),
    #         nn.Linear(1024, 512),
    #         nn.ReLU(inplace=False),
    #     )
        
    #     self.classifier12 = nn.Sequential(
    #         self.classifier1,
    #         self.classifier2,
    #         nn.Linear(512, num_classes),
    #     )

    # def forward(self, x):
    #     x = self.feature12345(x)
    #     x = x.view(x.size(0), -1)
        
    #     x = self.classifier12(x)
    #     return x
    
def alexnet(num_classes=100):
    return AlexNet(num_classes=num_classes)