import pyautogui
import cv2
import time
import os
from datetime import datetime


class I2CAutomation:
    def __init__(self, folder_name):
        # 安全機能：マウスを画面の左上角に移動すると緊急停止
        pyautogui.FAILSAFE = True
        # スクリーンショット機能を有効化
        pyautogui.useImageNotFoundException = False
        
        # 写真保存用のディレクトリを作成
        base_path = r"C:\Users\nitto\data"
        self.save_directory = os.path.join(base_path, folder_name)
        
        # ディレクトリが存在しない場合は作成
        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)
            print(f"フォルダを作成しました: {self.save_directory}")
        else:
            print(f"既存のフォルダを使用します: {self.save_directory}")
        
        # カメラの初期化
        self.camera = cv2.VideoCapture(0)  # 0は通常メインカメラ
        if not self.camera.isOpened():
            print("カメラが見つかりません。USBカメラまたは内蔵カメラを確認してください。")
            exit(1)
    
    def find_gui_elements(self):
        """GUI要素の位置を特定"""
        print("GUI要素の座標を設定中...")
        
        # 実際の環境で測定された正確な座標
        # 画面解像度: 1920x1200
        try:
            self.write_value_input = (938, 428)  # Write Value入力欄の実測座標
            self.write_button = (1107, 373)     # Writeボタンの実測座標
            
            print(f"Write Value入力欄: {self.write_value_input}")
            print(f"Writeボタン: {self.write_button}")
            print("実測座標で設定完了！")
            
        except Exception as e:
            print(f"座標設定エラー: {e}")
            return False
        
        return True
    
    def clear_input_field(self):
        """入力欄をクリアする"""
        # 入力欄をクリック
        pyautogui.click(self.write_value_input)
        time.sleep(0.2)
        
        # 全選択してクリア
        pyautogui.hotkey('ctrl', 'a')
        time.sleep(0.1)
        pyautogui.press('delete')
        time.sleep(0.1)
    
    def input_value(self, value):
        """指定された値を入力"""
        print(f"値 '{value}' を入力中...")
        
        # 入力欄をクリック
        pyautogui.click(self.write_value_input)
        time.sleep(0.2)
        
        # 既存の値をクリア
        self.clear_input_field()
        
        # 新しい値を入力（writeメソッドを使用）
        pyautogui.write(value)
        time.sleep(0.3)
    
    def click_write_button(self):
        """Writeボタンをクリック"""
        print("Writeボタンをクリック中...")
        pyautogui.click(self.write_button)
        print("レジスタ変更後3秒待機中...")
        time.sleep(3)  # レジスタ変更の反映を待つ
    
    def take_photo(self, filename):
        """写真を撮影して保存"""
        print(f"写真撮影中: {filename}")
        
        # カメラから画像を取得
        ret, frame = self.camera.read()
        
        if ret:
            # ファイルパスを作成
            filepath = os.path.join(self.save_directory, filename)
            
            # 画像を保存
            success = cv2.imwrite(filepath, frame)
            if success:
                print(f"写真を保存しました: {filepath}")
            else:
                print(f"写真の保存に失敗しました: {filepath}")
            
            # ファイルサイズを確認
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath)
                print(f"ファイルサイズ: {file_size} bytes")
            
        else:
            print("カメラからの画像取得に失敗しました")
            # カメラの再初期化を試行
            print("カメラの再初期化を試行中...")
            self.camera.release()
            time.sleep(0.5)
            self.camera = cv2.VideoCapture(0)
            if self.camera.isOpened():
                print("カメラの再初期化に成功しました")
                # 再度撮影を試行
                ret, frame = self.camera.read()
                if ret:
                    filepath = os.path.join(self.save_directory, filename)
                    cv2.imwrite(filepath, frame)
                    print(f"再試行で写真を保存しました: {filepath}")
            else:
                print("カメラの再初期化に失敗しました")
    
    def run_automation(self):
        """メインの自動化処理"""
        # テスト用の値リスト
        test_values = ['00', '01', '02', '04', '08', '16', '32', '64', 'C8', 'FF']
        
        print("I2C GUI自動化を開始します...")
        print("緊急停止したい場合は、マウスを画面の左上角に移動してください")
        
        # GUI要素の位置を特定
        if not self.find_gui_elements():
            return
        
        # 開始前の待機時間
        print("3秒後に開始します...")
        time.sleep(3)
        
        # 各値について処理を実行
        for i, value in enumerate(test_values, 1):
            try:
                print(f"\n=== ステップ {i}/10: 値 '{value}' の処理 ===")
                
                # 1. 値を入力
                self.input_value(value)
                
                # 2. Writeボタンをクリック
                self.click_write_button()
                
                # 3. 写真を撮影
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{value}.jpg"
                self.take_photo(filename)
                
                # 次の操作前の待機
                time.sleep(1)
                
            except pyautogui.FailSafeException:
                print("緊急停止が実行されました")
                break
            except Exception as e:
                print(f"エラーが発生しました: {e}")
                print("処理を続行します...")
                continue
        
        print("\n自動化処理が完了しました！")
        self.cleanup()
    
    def cleanup(self):
        """リソースのクリーンアップ"""
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
        print(f"写真は '{self.save_directory}' フォルダに保存されました")

def main():
    """メイン関数"""
    print("=== I2C GUI自動化ツール ===")
    print()
    
    # フォルダ名の入力
    while True:
        folder_name = input("保存用フォルダ名を入力してください: ").strip()
        if folder_name:
            # ファイル名として使用できない文字をチェック
            invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
            if any(char in folder_name for char in invalid_chars):
                print("フォルダ名に使用できない文字が含まれています。")
                print("使用できない文字: < > : \" / \\ | ? *")
                continue
            break
        else:
            print("フォルダ名を入力してください。")
    
    print(f"保存先: C:\\Users\\nitto\\iCloudDrive\\Desktop\\大学院\\研究\\data\\{folder_name}")
    
    # 自動化実行
    automation = I2CAutomation(folder_name)
    
    # カメラテスト
    print("\nカメラテストを実行中...")
    ret, frame = automation.camera.read()
    if ret:
        print("✅ カメラは正常に動作しています")
        print(f"画像サイズ: {frame.shape[1]}x{frame.shape[0]}")
    else:
        print("❌ カメラに問題があります")
        response = input("カメラに問題がありますが続行しますか？ (y/n): ")
        if response.lower() != 'y':
            automation.cleanup()
            return
    
    # 最終確認
    print("\n注意事項:")
    print("1. SVMCTLのI2C設定画面を開いてください")
    print("2. Slave Address: 2A、Sub Address: B47A が設定されていることを確認してください")
    print("3. Write Valueフィールドが見えている状態にしてください")
    print("4. 緊急停止したい場合は、マウスを画面左上角に移動してください")
    
    confirm = input("\n自動化を開始しますか？ (y/n): ")
    if confirm.lower() == 'y':
        automation.run_automation()
    else:
        print("自動化をキャンセルしました。")
        automation.cleanup()

if __name__ == "__main__":
    main()