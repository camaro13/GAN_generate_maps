import cv2
import os

# =========================
# 1. 경로 설정
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # map_proc 폴더
INPUT_DIR = os.path.join(BASE_DIR, "..", "processed_LA")   # 방 이미지 폴더
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "processed_LA_128")  # 리사이즈 저장 폴더

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# 2. 리사이즈 함수
# =========================
def resize_images(input_dir, output_dir, size=(128,128)):
    count = 0
    for fname in os.listdir(input_dir):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            fpath = os.path.join(input_dir, fname)
            img = cv2.imread(fpath)

            if img is None:
                print(f"[무시됨] 이미지 로드 실패: {fpath}")
                continue

            resized = cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)

            out_path = os.path.join(output_dir, fname)
            cv2.imwrite(out_path, resized)
            count += 1
    return count

# =========================
# 3. 실행
# =========================
count = resize_images(INPUT_DIR, OUTPUT_DIR)
print(f"\n✅ {count}개 이미지가 128x128로 변환 완료")
print(f"✅ 저장 경로: {OUTPUT_DIR}")
