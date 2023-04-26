from torch.utils.tensorboard import SummaryWriter
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.manifold import TSNE
from torch.utils.tensorboard import SummaryWriter
import numpy as np




class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))]
)
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
classes = testset.classes

# st.write("latent space visualisation")


autoencoder = Autoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.01)

num_epochs = 50


def run_training():
    for epoch in range(num_epochs):
        running_loss = 0.0
        print("................................", epoch, ".................................")
        for i, data in enumerate(trainloader, 0):
            inputs, _ = data
            optimizer.zero_grad()
            outputs = autoencoder(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i  == len(trainloader)-1:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss/len(trainloader)))
                running_loss = 0.0

        if (epoch+1)%5==0:
            dataiter = iter(testloader)
            images, labels= dataiter.__next__()

            encoded = autoencoder.encoder(images)
            encoded = encoded.view(encoded.size(0), -1)
            class_names = testloader.dataset.classes
            label_names = [class_names[label] for label in labels]

            tsne = TSNE(n_components=3)
            features_tsne = tsne.fit_transform(encoded.detach().numpy())
            print(list(features_tsne[:10]),list(label_names[:10]))
            fig, ax = plt.subplots()
            ax.scatter(features_tsne[:, 0], features_tsne[:, 1], c=labels.numpy())
            ax.set_title(f"t-SNE visualization after epoch {epoch+1}")
            ax.set_xlabel("Dimension 1")
            ax.set_ylabel("Dimension 2")
            plt.savefig(f"plots/plot{(epoch)//5+1}.png")
    

if __name__ == '__main__':
    run_training()
    print("END")


