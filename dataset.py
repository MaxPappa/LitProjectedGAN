import torch
from torchvision import transforms, datasets


def load_data(data_path, batch_size):#, mean, std):
    train_transforms = transforms.Compose([transforms.Resize((256, 256)),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
    train_data = datasets.ImageFolder(data_path, transform=train_transforms)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    return trainloader
