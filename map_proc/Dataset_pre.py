import os
import cv2
import numpy as np
from tqdm import tqdm

# ======================
# 경로 설정
# ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "..", "train_before")   # 원본 방 이미지 폴더
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "train")  # 증강된 방 저장 폴더
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================
# 증강 함수
# ======================
def augment_image(img):
    h, w = img.shape[:2]
    augmented = []

    # 1. 원본
    augmented.append(img)

    # 2. 좌우 반전
    augmented.append(cv2.flip(img, 1))

    # 3. 상하 반전
    augmented.append(cv2.flip(img, 0))

    # 4. 90도 회전
    augmented.append(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))
    augmented.append(cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE))

    # 5. 밝기/대비 조정
    for alpha in [0.8, 1.2]:  # contrast
        for beta in [-20, 20]:  # brightness
            new_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
            augmented.append(new_img)

    return augmented

# ======================
# 메인 증강 루프
# ======================
count = 0
for fname in tqdm(os.listdir(INPUT_DIR)):
    if fname.lower().endswith((".png", ".jpg", ".jpeg")):
        path = os.path.join(INPUT_DIR, fname)
        img = cv2.imread(path)

        if img is None:
            continue

        aug_list = augment_image(img)
        for i, aug in enumerate(aug_list):
            out_name = f"{os.path.splitext(fname)[0]}_aug{i}.png"
            cv2.imwrite(os.path.join(OUTPUT_DIR, out_name), aug)
            count += 1

print(f"\n✅ 증강 완료: 총 {count}장 저장됨")
print(f"✅ 저장 경로: {OUTPUT_DIR}")
