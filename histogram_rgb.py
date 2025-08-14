import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def rgb_histograms(filename, save_dir):
    """RGB各チャンネルのヒストグラムを作成・保存"""
    img = np.asarray(Image.open(filename).convert("RGB"))
    
    # RGB各チャンネルに分離
    r_channel = img[:, :, 0].flatten()
    g_channel = img[:, :, 1].flatten()
    b_channel = img[:, :, 2].flatten()
    
    base_name = os.path.splitext(os.path.basename(filename))[0]
    
    # 1. 個別のヒストグラム（赤チャンネル）
    plt.figure(figsize=(8, 6))
    plt.hist(r_channel, bins=256, range=(0, 256), color='red', alpha=0.7)
    plt.title(f"Red Channel Histogram - {os.path.basename(filename)}")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    save_path = os.path.join(save_dir, f"{base_name}_red_hist.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. 個別のヒストグラム（緑チャンネル）
    plt.figure(figsize=(8, 6))
    plt.hist(g_channel, bins=256, range=(0, 256), color='green', alpha=0.7)
    plt.title(f"Green Channel Histogram - {os.path.basename(filename)}")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    save_path = os.path.join(save_dir, f"{base_name}_green_hist.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. 個別のヒストグラム（青チャンネル）
    plt.figure(figsize=(8, 6))
    plt.hist(b_channel, bins=256, range=(0, 256), color='blue', alpha=0.7)
    plt.title(f"Blue Channel Histogram - {os.path.basename(filename)}")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    save_path = os.path.join(save_dir, f"{base_name}_blue_hist.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. 重ね合わせたRGBヒストグラム
    plt.figure(figsize=(10, 6))
    plt.hist(r_channel, bins=256, range=(0, 256), color='red', alpha=0.6, label='Red')
    plt.hist(g_channel, bins=256, range=(0, 256), color='green', alpha=0.6, label='Green')
    plt.hist(b_channel, bins=256, range=(0, 256), color='blue', alpha=0.6, label='Blue')
    plt.title(f"RGB Histogram - {os.path.basename(filename)}")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_path = os.path.join(save_dir, f"{base_name}_rgb_combined_hist.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()



# メイン処理
if __name__ == "__main__":
    # カレントディレクトリの画像ファイルを対象
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    current_dir = os.getcwd()
    image_files = [f for f in os.listdir(current_dir) if f.lower().endswith(image_extensions)]
    
    if not image_files:
        print("画像ファイルが見つかりませんでした。")
        exit()
    
    # 出力ディレクトリを用意（なければ作成）
    output_dir = os.path.join(current_dir, "rgb_histograms")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"処理対象の画像: {len(image_files)}個")
    
    # 各画像についてRGBヒストグラムを作成・保存
    for i, image_file in enumerate(image_files, 1):
        print(f"処理中... ({i}/{len(image_files)}) {image_file}")
        try:
            rgb_histograms(image_file, output_dir)
        except Exception as e:
            print(f"エラー: {image_file} の処理に失敗しました - {e}")
    
    print(f"\nRGBヒストグラム画像を '{output_dir}' フォルダに保存しました。")
    print("各画像につき以下のファイルが作成されます:")
    print("- [画像名]_red_hist.png (赤チャンネル)")
    print("- [画像名]_green_hist.png (緑チャンネル)")
    print("- [画像名]_blue_hist.png (青チャンネル)")
    print("- [画像名]_rgb_combined_hist.png (RGB重ね合わせ)")