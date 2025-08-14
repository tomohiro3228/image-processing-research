import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def color_hist(filename, save_dir):
    img = np.asarray(Image.open(filename).convert("L")).reshape(-1, 1)
    
    plt.figure()
    plt.hist(img, bins=256, range=(0, 256), color='#1f77b4')
    plt.title(f"Histogram of {os.path.basename(filename)}")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.grid(True)

    # 保存先のファイル名
    base_name = os.path.splitext(os.path.basename(filename))[0]
    save_path = os.path.join(save_dir, f"{base_name}_hist.png")
    plt.savefig(save_path)
    plt.close()  # メモリ節約のために閉じる

# カレントディレクトリの画像ファイルを対象
image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
current_dir = os.getcwd()
image_files = [f for f in os.listdir(current_dir) if f.lower().endswith(image_extensions)]

# 出力ディレクトリを用意（なければ作成）
output_dir = os.path.join(current_dir, "histograms")
os.makedirs(output_dir, exist_ok=True)

# 各画像についてヒストグラムを保存
for image_file in image_files:
    color_hist(image_file, output_dir)

print(f"ヒストグラム画像を '{output_dir}' フォルダに保存しました。")
