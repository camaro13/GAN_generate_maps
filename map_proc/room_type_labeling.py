import os
import json

# =========================
# 경로 설정
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ZELDA_DIR = os.path.join(BASE_DIR, "..", "processed_TLZ")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "processed_TLZ")

MAP_JSON = os.path.join(BASE_DIR, "..", "processed_TLZ", "TLZ_zelda_array.json")  # 너가 만든 맵 구조 JSON
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "TLZ_zelda_metadata.json")

# =========================
# 방 타입 정의 (임의 예시)
# =========================
ROOM_TYPES = {
    0: "empty",
    1: "normal",
    2: "treasure",
    3: "boss",
    4: "start",
    5: "key",
    6: "event",
    7: "stair"
}

# =========================
# JSON 로딩 & 라벨링 변환
# =========================
with open(MAP_JSON, "r", encoding="utf-8") as f:
    map_data = json.load(f)

all_rooms_meta = []

for floor_data in map_data["floors"]:
    floor_num = floor_data["floor"]
    layout = floor_data["layout"]

    for y, row in enumerate(layout):
        for x, cell in enumerate(row):
            if cell != 0:  # 0은 빈칸이니까 스킵
                # 파일 이름 규칙 (이미 방 자를 때 쓴 규칙 맞추기)
                fname = f"floor{floor_num}_x{x:02d}_y{y:02d}.png"

                all_rooms_meta.append({
                    "file": fname,
                    "floor": floor_num,
                    "grid_x": x,
                    "grid_y": y,
                    "room_type": cell   # ← 숫자 그대로 저장
                })

# =========================
# JSON 저장
# =========================
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(all_rooms_meta, f, indent=2, ensure_ascii=False)

print(f"✅ 라벨링 JSON 저장 완료: {OUTPUT_JSON}")