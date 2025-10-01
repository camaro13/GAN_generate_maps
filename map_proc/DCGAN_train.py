import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import torchvision.utils as vutils
import csv
from datetime import datetime

# =========================
# 1. 경로 설정
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, "..", "train")   # 방 이미지 폴더
JSON_PATH = os.path.join(IMG_DIR, "Maps_metadata.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "cgan_output")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# 2. 데이터셋 클래스 (room_type=0,8 제외)
# =========================
class ZeldaDataset(Dataset):
    def __init__(self, img_dir, json_path, transform=None):
        self.img_dir = img_dir
        with open(json_path, "r", encoding="utf-8") as f:
            all_meta = json.load(f)

        # room_type=0 (빈 공간), room_type=8 (데이터 부족) 제거
        self.metadata = [d for d in all_meta if d["room_type"] not in [0, 8]]
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        data = self.metadata[idx]
        img_path = os.path.join(self.img_dir, data["file"])
        image = Image.open(img_path).convert("RGB")

        # room_type=1~7 → 라벨 0~6
        label = data["room_type"] - 1

        if self.transform:
            image = self.transform(image)
        return image, label

# =========================
# 3. Transform (추가 증강 없음, Tensor + Normalize만)
# =========================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # [-1, 1]
])

dataset = ZeldaDataset(IMG_DIR, JSON_PATH, transform=transform)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2)

# =========================
# 4. cGAN 모델 정의 (room_type=1~7 → 7개 클래스)
# =========================
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(7, 32)  # 라벨 차원 32
        self.model = nn.Sequential(
            nn.ConvTranspose2d(100+32, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_input = self.label_emb(labels).unsqueeze(2).unsqueeze(3)
        input = torch.cat((noise, label_input), 1)  # (B,132,1,1)
        return self.model(input)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(7, 32)
        self.model = nn.Sequential(
            nn.Conv2d(3+32, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Linear(32768, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        label_input = self.label_emb(labels).unsqueeze(2).unsqueeze(3)
        label_input = label_input.expand(labels.size(0), 32, img.size(2), img.size(3))
        input = torch.cat((img, label_input), 1)
        return self.model(input)

# =========================
# 5. 학습 루프
# =========================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    netG = Generator().to(device)
    netD = Discriminator().to(device)

    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=0.0001, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=0.0001, betas=(0.5, 0.999))

    # 고정된 노이즈 & 라벨 (샘플 확인용)
    fixed_noise = torch.randn(7, 100, 1, 1, device=device)
    fixed_labels = torch.arange(0, 7, device=device)  # 0~6 (room_type=1~7)

    # CSV 로그 저장 준비
    log_path = os.path.join(OUTPUT_DIR, "training_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Loss_D", "Loss_G", "Time"])

    epochs = 500
    for epoch in range(epochs):
        for i, (imgs, labels) in enumerate(dataloader):
            real_imgs, labels = imgs.to(device), labels.to(device)
            b_size = real_imgs.size(0)

            real = torch.full((b_size,), 1.0, device=device)
            fake = torch.full((b_size,), 0.0, device=device)

            # ---- Discriminator ----
            netD.zero_grad()
            output = netD(real_imgs, labels).view(-1)
            lossD_real = criterion(output, real)
            lossD_real.backward()

            noise = torch.randn(b_size, 100, 1, 1, device=device)
            fake_imgs = netG(noise, labels)
            output = netD(fake_imgs.detach(), labels).view(-1)
            lossD_fake = criterion(output, fake)
            lossD_fake.backward()
            optimizerD.step()

            # ---- Generator ----
            netG.zero_grad()
            output = netD(fake_imgs, labels).view(-1)
            lossG = criterion(output, real)
            lossG.backward()
            optimizerG.step()

        # 로그 출력 및 CSV 기록
        now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[Epoch {epoch+1}/{epochs}] Loss_D: {(lossD_real+lossD_fake).item():.4f}, Loss_G: {lossG.item():.4f}, Time: {now_time}")
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, (lossD_real+lossD_fake).item(), lossG.item(), now_time])

        # 샘플 이미지 저장 (10 에폭마다)
        if (epoch+1) % 10 == 0:
            with torch.no_grad():
                fake = netG(fixed_noise, fixed_labels).detach().cpu()
            vutils.save_image(
                fake,
                os.path.join(OUTPUT_DIR, f"epoch_{epoch+1}.png"),
                normalize=True,
                nrow=7   # 한 줄에 7개 (room_type=1~7)
            )

    # 모델 저장
    torch.save(netG.state_dict(), os.path.join(OUTPUT_DIR, "netG_final.pth"))
    print("✅ Generator 저장 완료")

    # 타입별 10장씩 생성
    save_dir = os.path.join(OUTPUT_DIR, "generated_samples")
    os.makedirs(save_dir, exist_ok=True)

    netG.eval()
    with torch.no_grad():
        for label in range(7):  # 0~6 (room_type=1~7)
            noise = torch.randn(10, 100, 1, 1, device=device)
            labels = torch.full((10,), label, dtype=torch.long, device=device)

            fake_imgs = netG(noise, labels).detach().cpu()
            vutils.save_image(
                fake_imgs,
                os.path.join(save_dir, f"roomtype_{label+1}.png"),  # 다시 1~7로 저장
                normalize=True,
                nrow=5
            )

    print(f"✅ 각 타입별 10장씩 생성 완료 → {save_dir}")
