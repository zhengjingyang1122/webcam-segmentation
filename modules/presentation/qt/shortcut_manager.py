# modules/presentation/qt/shortcut_manager.py
from pathlib import Path
from typing import Dict, List
import yaml
from utils.get_base_path import get_base_path


class ShortcutManager:
    """管理應用程式快捷鍵的類別"""
    
    def __init__(self):
        self.config_path = Path(get_base_path()) / "config" / "config.yaml"
        self.shortcuts: Dict[str, List[str]] = {}
        self.load_shortcuts()
    
    def load_shortcuts(self) -> None:
        """從 config.yaml 載入快捷鍵設定"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                viewer_shortcuts = config.get('shortcuts', {}).get('viewer', {})
                
                # 將快捷鍵轉換為扁平化的字典
                for action, keys in viewer_shortcuts.items():
                    if isinstance(keys, list) and keys:
                        # 只取第一個快捷鍵
                        self.shortcuts[action] = keys[0]
                    elif isinstance(keys, str):
                        self.shortcuts[action] = keys
        except Exception as e:
            print(f"載入快捷鍵設定失敗: {e}")
            # 使用預設值
            self.shortcuts = {
                'nav.prev': 'PageUp',
                'nav.next': 'PageDown',
                'save.selected': 'Ctrl+S',
                'view.reset': 'R'
            }
    
    def save_shortcuts(self) -> None:
        """儲存快捷鍵設定到 config.yaml"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 更新快捷鍵設定
            if 'shortcuts' not in config:
                config['shortcuts'] = {}
            if 'viewer' not in config['shortcuts']:
                config['shortcuts']['viewer'] = {}
            
            for action, key in self.shortcuts.items():
                config['shortcuts']['viewer'][action] = [key]
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
        except Exception as e:
            print(f"儲存快捷鍵設定失敗: {e}")
    
    def get_shortcut(self, action: str) -> str:
        """取得指定動作的快捷鍵"""
        return self.shortcuts.get(action, '')
    
    def set_shortcut(self, action: str, key: str) -> None:
        """設定指定動作的快捷鍵"""
        self.shortcuts[action] = key
    
    def get_all_shortcuts(self) -> Dict[str, str]:
        """取得所有快捷鍵"""
        return self.shortcuts.copy()
