# import os
# import json

# # =========================
# # 1. 경로 설정
# # =========================
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # map_proc 폴더
# ZELDA_DIR = os.path.join(BASE_DIR, "..", "processed_BS")   # 원본 JSON 있는 폴더
# OUTPUT_DIR = os.path.join(BASE_DIR, "..", "processed_BS")      # 출력 JSON 저장할 폴더

# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # =========================
# # 2. JSON 불러오기
# # =========================
# INPUT_JSON = os.path.join(ZELDA_DIR, "BS_zelda_array.json")   # 네가 만든 원본 JSON
# OUTPUT_JSON = os.path.join(OUTPUT_DIR, "zelda_dungeon_labels.json")  # 변환된 JSON 저장 경로

# with open(INPUT_JSON, "r", encoding="utf-8") as f:
#     dungeon = json.load(f)

# # =========================
# # 3. 변환
# # =========================
# labels = []

# for floor_data in dungeon["floors"]:
#     floor = floor_data["floor"]
#     layout = floor_data["layout"]

#     for y, row in enumerate(layout):
#         for x, cell in enumerate(row):
#             cell_type = int(cell)

#             if cell_type != 0:  # 빈칸 제외
#                 locked = (cell_type == 6)  # 6=잠긴문 → locked=True

#                 labels.append({
#                     "file": f"Floor{floor}_x{x:02d}_y{y:02d}.png",
#                     "type": cell_type,
#                     "locked": locked
#                 })

# # =========================
# # 4. JSON 저장
# # =========================
# with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
#     json.dump(labels, f, indent=2, ensure_ascii=False)

# print(f"✅ 변환 완료: {OUTPUT_JSON}")


import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ZELDA_DIR = os.path.join(BASE_DIR, "..", "processed_BS")   # 원본 JSON 있는 폴더
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "processed_BS")      # 출력 JSON 저장할 폴더

os.makedirs(OUTPUT_DIR, exist_ok=True)

INPUT_JSON = os.path.join(ZELDA_DIR, "BS_zelda_array.json")   # 네가 만든 원본 JSON
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "zelda_dungeon_labels.json")  # 변환된 JSON 저장 경로

with open(INPUT_JSON, "r", encoding="utf-8") as f:
    dungeon = json.load(f)

# floors 데이터를 dict로 변환 (floor_num → layout)
dungeon_map = {f["floor"]: f["layout"] for f in dungeon["floors"]}

locked_rooms = set()
visited = {}  # {(floor, x, y): max_keys}

# =========================
# 2. 반복 DFS (스택 사용)
# =========================
def explore(start_floor, start_x, start_y):
    stack = [(start_floor, start_x, start_y, 0)]  # (floor, x, y, keys)
    max_keys = sum(row.count(5) for floor in dungeon_map.values() for row in floor)  # 전체 열쇠방 개수

    while stack:
        floor, x, y, keys = stack.pop()

        # 열쇠 수 상한 제한
        if keys > max_keys:
            keys = max_keys

        state = (floor, x, y)
        if state in visited and visited[state] >= keys:
            continue
        visited[state] = keys

        layout = dungeon_map[floor]
        H, W = len(layout), len(layout[0])
        room_type = layout[y][x]

        # 열쇠방
        if room_type == 5:
            keys = min(keys + 1, max_keys)

        # 보스방
        if room_type == 3:
            if keys > 0:
                keys -= 1
            else:
                locked_rooms.add((floor, x, y))
                continue

        # 잠긴방
        if room_type == 6:
            if keys > 0:
                keys -= 1
            else:
                locked_rooms.add((floor, x, y))
                continue

        # 4방향 탐색
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = x+dx, y+dy
            if 0 <= ny < H and 0 <= nx < W:
                if layout[ny][nx] != 0:
                    stack.append((floor, nx, ny, keys))


# =========================
# 3. 시작방에서 탐색
# =========================
for f, layout in dungeon_map.items():
    for y, row in enumerate(layout):
        for x, cell in enumerate(row):
            if cell == 4:  # 시작방
                explore(f, x, y)

# =========================
# 4. JSON 출력
# =========================
labels = []
for f, layout in dungeon_map.items():
    for y, row in enumerate(layout):
        for x, cell in enumerate(row):
            if cell != 0:
                labels.append({
                    "file": f"{f}_x{x:02d}_y{y:02d}.png",
                    "type": cell,
                    "locked": (f, x, y) in locked_rooms
                })

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(labels, f, indent=2, ensure_ascii=False)

print(f"✅ 변환 완료: {OUTPUT_JSON}")