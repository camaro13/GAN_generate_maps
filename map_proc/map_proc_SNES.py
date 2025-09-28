import cv2
import numpy as np
import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ZELDA_DIR = os.path.join(BASE_DIR, "..", "zelda", "link's awakening")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "processed_LA")
JSON_PATH = os.path.join(OUTPUT_DIR, "LA_zelda_metadata.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def split_la_map(img_path, out_size=(128,128), min_area=2000):
    img = cv2.imread(img_path)
    if img is None:
        print(f"[무시됨] 이미지 로드 실패: {img_path}")
        return [], []

    # 배경색 추출 (좌상단 픽셀)
    bg_color = img[0,0].tolist()

    # 배경 마스크 (허용 범위 확대)
    mask = cv2.inRange(img, np.array(bg_color)-40, np.array(bg_color)+40)
    mask_inv = cv2.bitwise_not(mask)

    # 윤곽선 찾기 (방 단위)
    contours, _ = cv2.findContours(mask_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rooms_meta = []
    rooms_data = []

    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < min_area:  # 너무 작은 영역은 무시
            continue

        room = img[y:y+h, x:x+w]
        resized = cv2.resize(room, out_size, interpolation=cv2.INTER_NEAREST)

        fname = f"{os.path.splitext(os.path.basename(img_path))[0]}_room_{i:03d}.png"
        fpath = os.path.join(OUTPUT_DIR, fname)
        cv2.imwrite(fpath, resized)

        rooms_data.append(resized)
        rooms_meta.append({
            "file": fname,
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h),
            "area": int(area)
        })

    return rooms_data, rooms_meta

# =========================
# 3. 전체 폴더 처리
# =========================
all_rooms = []
all_meta = []

for fname in os.listdir(ZELDA_DIR):
    if fname.lower().endswith((".png", ".jpg", ".jpeg")):
        fpath = os.path.join(ZELDA_DIR, fname)
        rooms, meta = split_la_map(fpath)
        print(f"{fname}: {len(rooms)}개 방 추출")
        all_rooms.extend(rooms)
        all_meta.extend(meta)

# =========================
# 4. NumPy + JSON 저장
# =========================
np.save(os.path.join(OUTPUT_DIR, "LA_zelda_rooms.npy"), np.array(all_rooms))

with open(JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(all_meta, f, indent=2)

print(f"\n✅ LA Zelda 최종 방 개수: {len(all_rooms)}")
print(f"✅ NumPy 저장 완료: {os.path.join(OUTPUT_DIR, 'LA_zelda_rooms.npy')}")
print(f"✅ JSON 저장 완료: {JSON_PATH}")
