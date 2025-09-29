import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# ======================
# 경로 & 하이퍼파라미터
# ======================
DATASET_DIR = "./augmented_BS"   # 증강된 데이터셋 폴더
OUTPUT_DIR = "./dcgan_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

image_size = 128
batch_size = 64
nz = 100          # latent vector 크기
ngf = 64          # generator feature map 수
ndf = 64          # discriminator feature map 수
num_epochs = 50
lr = 0.0002
beta1 = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# 데이터셋
# ======================
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # -1~1 정규화
])

dataset = ImageFolder(root=DATASET_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ======================
# 모델 정의 (DCGAN)
# ======================
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1)

# ======================
# 학습 준비
# ======================
netG = Generator().to(device)
netD = Discriminator().to(device)

criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# ======================
# 학습 루프
# ======================
for epoch in range(num_epochs):
    for i, (data, _) in enumerate(dataloader):
        # ----------------------
        # (1) Discriminator 학습
        # ----------------------
        netD.zero_grad()
        real = data.to(device)
        b_size = real.size(0)
        label = torch.full((b_size,), 1., device=device)

        output = netD(real)
        errD_real = criterion(output, label)
        errD_real.backward()

        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(0.)
        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        optimizerD.step()

        # ----------------------
        # (2) Generator 학습
        # ----------------------
        netG.zero_grad()
        label.fill_(1.)
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        optimizerG.step()

        # ----------------------
        # 로그
        # ----------------------
        if i % 50 == 0:
            print(f"[{epoch}/{num_epochs}] [{i}/{len(dataloader)}] "
                  f"Loss_D: {errD_real.item()+errD_fake.item():.4f} "
                  f"Loss_G: {errG.item():.4f}")

    # 샘플 이미지 저장
    with torch.no_grad():
        fake = netG(fixed_noise).detach().cpu()
    torchvision.utils.save_image(fake, f"{OUTPUT_DIR}/epoch_{epoch}.png", normalize=True)
