import cv2
import numpy as np
import os

# =========================
# 1. 경로 설정
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # map_proc 폴더
ZELDA_DIR = os.path.join(BASE_DIR, "..", "zelda", "A Link to the Past")   # 원본 맵 폴더
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "processed_LP")  # 잘라낸 방 저장 폴더

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# 2. Flood Fill + Contour 기반 방 분리 함수
# =========================
def extract_rooms(img_path, min_size=40):
    img = cv2.imread(img_path)
    if img is None:
        print(f"[무시됨] 이미지 로드 실패: {img_path}")
        return 0

    h, w, _ = img.shape

    # --- Flood Fill로 배경 제거 ---
    mask = np.zeros((h+2, w+2), np.uint8)
    floodfilled = img.copy()
    cv2.floodFill(floodfilled, mask, (0,0), (0,0,0))

    # --- Threshold 후 Contour 찾기 ---
    gray = cv2.cvtColor(floodfilled, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    count = 0
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        if w < min_size or h < min_size:  # 너무 작은 노이즈는 무시
            continue

        room = img[y:y+h, x:x+w]
        fname = f"{os.path.splitext(os.path.basename(img_path))[0]}_room{i:02d}.png"
        fpath = os.path.join(OUTPUT_DIR, fname)
        cv2.imwrite(fpath, room)
        count += 1

    return count

# =========================
# 3. 폴더 내 모든 이미지 처리
# =========================
total = 0
for fname in os.listdir(ZELDA_DIR):
    if fname.lower().endswith((".png", ".jpg", ".jpeg")):
        fpath = os.path.join(ZELDA_DIR, fname)
        count = extract_rooms(fpath)
        print(f"✅ {fname}: {count}개 방 추출 완료")
        total += count

print(f"\n🎉 전체 추출된 방 개수: {total}")
print(f"✅ 저장 경로: {OUTPUT_DIR}")
