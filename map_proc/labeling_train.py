import os
import json
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, "..", "train")   # 증강 이미지 포함 폴더
OUTPUT_JSON = os.path.join(IMG_DIR, "Maps_metadata.json")

# 원본 방 타입 JSON 목록
SOURCE_JSONS = [
    os.path.join(BASE_DIR, "..", "processed_BS", "BS_zelda_array.json"),
    os.path.join(BASE_DIR, "..", "processed_LA", "LA_zelda_array.json"),
    os.path.join(BASE_DIR, "..", "processed_TLZ", "TLZ_zelda_array.json"),
]

# =========================
# 1. 원본 JSON 합치기
# =========================
roomtype_map = {}  # (floor_name, gx, gy) -> room_type

def normalize_floor(name: str) -> str:
    """floor 이름 정규화"""
    if name.startswith("temple_"):
        name = name.replace("temple_", "", 1)
    if name.startswith("ZeldaLevel"):
        name = name.replace("Zelda", "", 1)  # ZeldaLevel6Q2 -> Level6Q2
    return name

for jpath in SOURCE_JSONS:
    with open(jpath, "r", encoding="utf-8") as f:
        data = json.load(f)

    for floor_data in data["floors"]:
        floor_name = normalize_floor(floor_data["floor"])
        layout = floor_data["layout"]

        for gy, row in enumerate(layout):
            for gx, rtype in enumerate(row):
                if rtype != 0:
                    roomtype_map[(floor_name, gx, gy)] = rtype

print(f"✅ 원본 JSON 총 {len(roomtype_map)}개 방 정보 로드")

# =========================
# 2. 기존 metadata 불러오기
# =========================
if os.path.exists(OUTPUT_JSON):
    with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
        try:
            metadata = json.load(f)
        except json.JSONDecodeError:
            metadata = []
else:
    metadata = []

existing_files = set(item["file"] for item in metadata)

# =========================
# 3. 이미지 파일 처리
# =========================
for fname in os.listdir(IMG_DIR):
    if not fname.lower().endswith(".png"):
        continue
    if fname in existing_files:
        continue

    # 확장자 제거 + 증강 접미사 제거
    name_no_ext = os.path.splitext(fname)[0]
    name_base = re.sub(r"_aug\d+$", "", name_no_ext)

    # 좌표 파싱
    match_xy = re.search(r"_x(\d+)_y(\d+)", name_base)
    if not match_xy:
        print(f"[무시됨] 좌표 파싱 실패: {fname}")
        continue
    gx, gy = int(match_xy.group(1)), int(match_xy.group(2))

    # floor_name = _x 앞부분
    floor_name_raw = name_base.split("_x")[0]
    floor_name_norm = normalize_floor(floor_name_raw)

    room_type = roomtype_map.get((floor_name_norm, gx, gy), -1)
    if room_type == -1:
        print(f"[경고] 매칭 실패: {fname} → floor:{floor_name_norm}, x:{gx}, y:{gy}")

    metadata.append({
        "file": fname,
        "room_type": room_type,
        "grid_x": gx,
        "grid_y": gy,
        "floor": floor_name_norm
    })

print(f"✅ 최종 JSON 총 {len(metadata)}개 항목")

# =========================
# 4. JSON 저장
# =========================
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"✅ JSON 업데이트 완료 → {OUTPUT_JSON}")