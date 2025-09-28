import cv2
import numpy as np
import os

# =========================
# 1. 경로 설정
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # map_proc
PROCESSED_DIR = os.path.join(BASE_DIR, "..", "processed_nes")

# =========================
# 2. 검정 방 필터링 함수
# =========================
def is_black_image(img_path, min_var=5):
    img = cv2.imread(img_path)
    if img is None:
        return True  # 불러오기 실패하면 삭제 대상으로
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.var(gray) < min_var  # 분산이 낮으면 "거의 검정"

# =========================
# 3. processed_nes 내 파일 처리
# =========================
removed = 0
for fname in os.listdir(PROCESSED_DIR):
    if fname.lower().endswith((".png", ".jpg", ".jpeg")):
        fpath = os.path.join(PROCESSED_DIR, fname)
        if is_black_image(fpath):
            os.remove(fpath)
            removed += 1

print(f"\n✅ 삭제된 검정 방 개수: {removed}")
