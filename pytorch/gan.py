from torch import nn
import torch

real_label = 1.
fake_label = 0.

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def get_real_data(data_dim, batch_size):
    for i in range(300):
        a = torch.distributions.uniform.Uniform(1,2).sample((batch_size,1))
        base = torch.linspace(-1,1,data_dim).view(1,-1)
        yield a*torch.pow(base,2) + (a-1)


class GAN(nn.Module):
    def __init__(self, latent_dim, data_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.data_dim = data_dim

        self.g = self._get_generator()
        self.d = self._get_discriminator()

        self.optimizerD = torch.optim.Adam(self.d.parameters(), lr=0.0001, betas=(0.5, 0.999))
        self.optimizerG = torch.optim.Adam(self.g.parameters(), lr=0.0001, betas=(0.5, 0.999))

        self.loss_func = nn.BCELoss()
    
    def forward(self, n):
        return self.g(torch.normal(0,1,size=(n,self.latent_dim),device=device))
    
    def _get_generator(self):
        generator = nn.Sequential(
            nn.Linear(self.latent_dim,32),
            nn.ReLU(),
            nn.Linear(32,self.data_dim)
        )
        return generator
    
    def _get_discriminator(self):
        discriminator = nn.Sequential(
            nn.Linear(self.data_dim,32),
            nn.ReLU(),
            nn.Linear(32,1),
            nn.Sigmoid()
        )
        return discriminator
    
    def train_g(self,data):
        self.g.zero_grad()
        label = torch.full((data.shape[0],),real_label,dtype=torch.float, device=device)
        output = self.d(data).view(-1)
        loss = self.loss_func(output,label)
        loss.backward()
        self.optimizerG.step()
        accuracy = ((output>0.5).float().view(-1) == label).sum()/label.shape[0]
        return loss.item(), accuracy.item()
    def train_d(self,data,label):
        self.d.zero_grad()
        pred = self.d(data).view(-1)
        loss = self.loss_func(pred, label)
        loss.backward()
        self.optimizerD.step()
        accuracy = ((pred>0.5).float().view(-1) == label).sum()/label.shape[0]
        return loss.item(), accuracy.item()
    def step(self,data):
        # train discriminator, maximize log(D(x)) + log(1 - D(Gz))
        fake_data = self(data.shape[0])
        label = torch.full((data.shape[0]*2,),real_label,dtype=torch.float,device=device)
        label[data.shape[0]:]=fake_label
        d_data = torch.cat((data,fake_data.detach()),0)
        d_loss, d_acc = self.train_d(d_data,label)
        g_loss, g_acc = self.train_g(fake_data)
        return d_loss,d_acc,g_loss,g_acc


def train(gan,epoch):
    for ep in range(epoch):
        for t, data in enumerate(get_real_data(16,32)):
            data = data.to(device)
            d_loss, d_acc, g_loss, g_acc = gan.step(data)
            if t %400 == 0:
                print(f"Epoch {ep}  |   t {t}   |   d_acc={round(d_acc,2)}  |   g_acc={round(g_acc,2)}  |   d_loss={round(d_loss,2)}    |   g_loss={round(g_loss,2)}")
if __name__ == "__main__":
    LATENT_DIM = 16
    DATA_DIM = 16
    BATCH_SIZE = 32
    EPOCH = 20

    model = GAN(LATENT_DIM,DATA_DIM)
    model = model.to(device)
    train(model,EPOCH)
