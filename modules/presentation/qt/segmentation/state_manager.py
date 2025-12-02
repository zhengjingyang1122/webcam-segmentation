from __future__ import annotations
from typing import Dict, List, Set, Optional
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class StateManager:
    """Handles annotation state, selection, and undo/redo history."""
    
    def __init__(self, max_history: int = 20):
        self.selected_indices: Set[int] = set()
        self.annotations: Dict[int, int] = {}  # {mask_index: class_id}
        self.annotation_history: List[Dict] = []
        self.annotation_redo_stack: List[Dict] = []
        self.max_history = max_history
        
        # Per-image state cache
        self.per_image_state: Dict[Path, Dict] = {}

    def clear_current_state(self):
        self.selected_indices.clear()
        self.annotations.clear()
        self.annotation_history.clear()
        self.annotation_redo_stack.clear()

    def save_history(self):
        """Save current state to history."""
        state = {
            'selected_indices': self.selected_indices.copy(),
            'annotations': self.annotations.copy()
        }
        self.annotation_history.append(state)
        if len(self.annotation_history) > self.max_history:
            self.annotation_history.pop(0)
        self.annotation_redo_stack.clear()

    def undo(self) -> bool:
        """Perform undo. Returns True if successful."""
        if not self.annotation_history:
            return False
        
        # Save current to redo
        current_state = {
            'selected_indices': self.selected_indices.copy(),
            'annotations': self.annotations.copy()
        }
        self.annotation_redo_stack.append(current_state)
        
        # Restore from history
        state = self.annotation_history.pop()
        self.selected_indices = state['selected_indices']
        self.annotations = state['annotations']
        return True

    def redo(self) -> bool:
        """Perform redo. Returns True if successful."""
        if not self.annotation_redo_stack:
            return False
        
        # Save current to history
        current_state = {
            'selected_indices': self.selected_indices.copy(),
            'annotations': self.annotations.copy()
        }
        self.annotation_history.append(current_state)
        
        # Restore from redo
        state = self.annotation_redo_stack.pop()
        self.selected_indices = state['selected_indices']
        self.annotations = state['annotations']
        return True

    def can_undo(self) -> bool:
        return len(self.annotation_history) > 0

    def can_redo(self) -> bool:
        return len(self.annotation_redo_stack) > 0

    def save_image_state(self, path: Path):
        """Save state for a specific image path."""
        self.per_image_state[path] = {
            'selected_indices': self.selected_indices.copy(),
            'annotations': self.annotations.copy()
        }

    def load_image_state(self, path: Path) -> bool:
        """Load state for a specific image path. Returns True if loaded from memory."""
        if path in self.per_image_state:
            state = self.per_image_state[path]
            self.selected_indices = state['selected_indices'].copy()
            self.annotations = state['annotations'].copy()
            self.annotation_history.clear()
            self.annotation_redo_stack.clear()
            return True
        return False

    def load_from_json(self, json_path: Path) -> bool:
        """Load annotations from a JSON file."""
        if not json_path.exists():
            return False
            
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.selected_indices.clear()
            self.annotations.clear()
            
            if 'annotations' in data:
                for ann in data['annotations']:
                    idx = ann['index']
                    class_id = ann.get('class_id', 0)
                    self.selected_indices.add(idx)
                    self.annotations[idx] = class_id
            elif 'selected_indices' in data:
                # Legacy format
                self.selected_indices = set(data['selected_indices'])
                self.annotations = {idx: 0 for idx in self.selected_indices}
            
            self.annotation_history.clear()
            self.annotation_redo_stack.clear()
            return True
        except Exception as e:
            logger.error(f"Failed to load annotations from {json_path}: {e}")
            return False
