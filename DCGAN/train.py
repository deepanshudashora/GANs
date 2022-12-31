import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import Discriminator, Generator, initialize_weights
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available else "cpu")
learning_rate = 2e-4
batch_size = 128
image_size = 64
channels_img = 3 # 1 for mnist
z_dim = 100
num_epochs = 10
features_disc = 64
features_gen = 64
step=0
transforms = transforms.Compose(
    [
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(channels_img)],[0.5 for _ in range(channels_img)]
        ),
    ]
)

#dataset = datasets.MNIST(root = "dataset/", train=True, transform=transforms,download=True)
dataset = datasets.ImageFolder(root = "celeb_dataset", transform=transforms)
loader = DataLoader(dataset,batch_size=batch_size,shuffle=True)
gen = Generator(z_dim, channels_img, features_gen).to(device)
disc = Discriminator(channels_img, features_disc).to(device)
initialize_weights(gen)
initialize_weights(disc)

opt_gen = optim.Adam(gen.parameters(), lr=learning_rate, betas = (0.5,0.999))
opt_disc = optim.Adam(disc.parameters(), lr=learning_rate, betas = (0.5,0.999))
criterion = nn.BCELoss()

fixed_noise = torch.randn(32,z_dim,1,1).to(device)
writer_real = SummaryWriter(f"celab_logs/real")
writer_fake = SummaryWriter(f"celeab_logs/fake")

gen.train()
disc.train()

for epoch in range(num_epochs):
    # img and labels
    for batch_idx, (real,_) in enumerate(tqdm(loader)):
        real = real.to(device)
        noise = torch.randn((batch_size,z_dim,1,1)).to(device)
        
        # Train Disc
        fake = gen(noise)
        disc_real = disc(real).reshape (-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake).reshape(-1)
        lossD_fake = criterion(disc_fake,torch.zeros_like(disc_fake))
        lossD = (lossD_real + lossD_fake) / 2
        disc.zero_grad()
        lossD.backward(retain_graph=True)
        opt_disc.step()
        
        # Train Generator
        output = disc(fake).reshape(-1)
        lossG = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()
        
        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] \ "
                f"Loss D: {lossD: .4f}, LossG: {lossG: .4f}"
            )
            
            with torch.no_grad():
                fake = gen(fixed_noise)

                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                
                writer_fake.add_image(
                    "CELEB Fake Image", img_grid_fake, global_step=step
                )
                
                writer_real.add_image(
                    "CELEB real Image", img_grid_real, global_step=step
                )
                
            step += 1