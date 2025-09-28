import cv2
import os

# =========================
# 1. 경로 설정
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # map_proc 폴더
ZELDA_DIR = os.path.join(BASE_DIR, "..", "zelda", "Phantom Hourglass")   # 원본 맵 폴더
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "processed_PH")  # 잘라낸 방 저장 폴더

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# 2. 자르기 함수
# =========================
def slice_map(img_path, rows=8, cols=8):
    img = cv2.imread(img_path)
    if img is None:
        print(f"[무시됨] 이미지 로드 실패: {img_path}")
        return 0

    h, w, _ = img.shape
    room_h = h / rows
    room_w = w / cols

    count = 0
    for gy in range(rows):
        for gx in range(cols):
            y1 = int(round(gy * room_h))
            y2 = int(round((gy+1) * room_h))
            x1 = int(round(gx * room_w))
            x2 = int(round((gx+1) * room_w))

            room = img[y1:y2, x1:x2]

            fname = f"{os.path.splitext(os.path.basename(img_path))[0]}_x{gx:02d}_y{gy:02d}.png"
            fpath = os.path.join(OUTPUT_DIR, fname)
            cv2.imwrite(fpath, room)
            count += 1
    return count

# =========================
# 3. 단일 이미지 처리
# =========================
TARGET_FILE = "The Legend of Zelda_ Phantom Hourglass Maps4.png"  # 👉 여기 파일 이름만 바꿔주면 됨
img_path = os.path.join(ZELDA_DIR, TARGET_FILE)

count = slice_map(img_path, rows=4, cols=4)  # 👉 원하는 방 개수로 설정
print(f"\n✅ {TARGET_FILE}: {count}개 방 추출 완료")
print(f"✅ 저장 경로: {OUTPUT_DIR}")
