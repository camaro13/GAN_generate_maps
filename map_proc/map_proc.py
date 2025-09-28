import cv2
import numpy as np
import os
import json

# =========================
# 1. 경로 설정
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # map_proc 폴더
# ZELDA_DIR = os.path.join(BASE_DIR, "..", "zelda", "The Legend of Zelda")
# OUTPUT_DIR = os.path.join(BASE_DIR, "..", "processed_nes")
# JSON_PATH = os.path.join(OUTPUT_DIR, "nes_zelda_metadata.json")
ZELDA_DIR = os.path.join(BASE_DIR, "..", "zelda", "BS Zelda")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "processed_BS")
JSON_PATH = os.path.join(OUTPUT_DIR, "BS_zelda_metadata.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# 2. NES Zelda 맵 자르기 + 좌표 저장
# =========================
def split_nes_map(img_path, room_w=256, room_h=176, out_size=(128,128)):
    img = cv2.imread(img_path)
    if img is None:
        print(f"[무시됨] 이미지 로드 실패: {img_path}")
        return []

    h, w, _ = img.shape
    rooms_meta = []
    rooms_data = []

    for gy, y in enumerate(range(0, h, room_h)):
        for gx, x in enumerate(range(0, w, room_w)):
            room = img[y:y+room_h, x:x+room_w]

            # 정확한 방 크기만 처리
            if room.shape[0] == room_h and room.shape[1] == room_w:
                gray = cv2.cvtColor(room, cv2.COLOR_BGR2GRAY)
                # if np.var(gray) < 5:  # 거의 검정 방이면 스킵
                #     continue

                resized = cv2.resize(room, out_size, interpolation=cv2.INTER_NEAREST)

                # 파일 이름 만들기
                fname = f"{os.path.splitext(os.path.basename(img_path))[0]}_x{gx:02d}_y{gy:02d}.png"
                fpath = os.path.join(OUTPUT_DIR, fname)

                cv2.imwrite(fpath, resized)

                rooms_data.append(resized)
                rooms_meta.append({
                    "file": fname,
                    "grid_x": gx,
                    "grid_y": gy
                })

    return rooms_data, rooms_meta

# =========================
# 3. The Legend of Zelda 폴더 전체 처리
# =========================
all_rooms = []
all_meta = []

for fname in os.listdir(ZELDA_DIR):
    if fname.lower().endswith((".png", ".jpg", ".jpeg")):
        fpath = os.path.join(ZELDA_DIR, fname)
        rooms, meta = split_nes_map(fpath)
        print(f"{fname}: {len(rooms)}개 방 추출 (검정 방 제외)")

        all_rooms.extend(rooms)
        all_meta.extend(meta)

# =========================
# 4. NumPy + JSON 저장
# =========================
np.save(os.path.join(OUTPUT_DIR, "BS_zelda_rooms.npy"), np.array(all_rooms))

with open(JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(all_meta, f, indent=2)

print(f"\n✅ BS Zelda 최종 방 개수: {len(all_rooms)}")
print(f"✅ NumPy 저장 완료: {os.path.join(OUTPUT_DIR, 'BS_zelda_rooms.npy')}")
print(f"✅ JSON 저장 완료: {JSON_PATH}")
