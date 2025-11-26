# utils/get_base_path.py
import sys
import os

def get_base_path() -> str:
    """
    取得應用程式的根目錄。
    
    如果在 PyInstaller 打包的環境中執行，它會返回可執行檔所在的目錄。
    否則，返回專案的根目錄。
    """
    if getattr(sys, 'frozen', False):
        # 作為打包後的 .exe 執行
        return os.path.dirname(sys.executable)
    else:
        # 作為 .py 腳本執行
        # 這個檔案在 utils/，所以我們需要向上兩層來到專案根目錄
        return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
