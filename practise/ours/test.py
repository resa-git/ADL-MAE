import matplotlib.pyplot as plt
import numpy as np
import models_mae

x,y = next(iter(data_loader_train))

img = x[8].detach().cpu()
npimg = img.numpy()
plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest') 
plt.show()


model = models_mae.__dict__['mae_vit_base_patch16']()
checkpoint = torch.load('output_dir3/checkpoint-70.pth', map_location='cpu')
model.load_state_dict(checkpoint['model'])
model.train(True)
#y = model(x)
loss, y, _ = model(samples, mask_ratio=args.mask_ratio)




















y = checkpoint(x)   



transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)


trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)



from torchvision import datasets
data_path = './'

cifar10 =  datasets.CIFAR10(data_path, train=True, download=True)
cifar10_val = datasets.CIFAR10(data_path, train=False, download=True)