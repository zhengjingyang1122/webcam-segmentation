from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Set
import cv2
import numpy as np
from PySide6.QtWidgets import QMessageBox

from .utils import compute_bbox, compute_polygon

logger = logging.getLogger(__name__)

class Exporter:
    """Handles exporting segmentation results to various formats."""

    @staticmethod
    def save_union(
        parent_widget,
        image_path: Path,
        bgr: np.ndarray,
        masks: List[np.ndarray],
        selected_indices: Set[int],
        annotations: Dict[int, int],
        output_path_str: str,
        format_str: str,
        crop_bbox: bool,
        chk_coco: bool,
        chk_voc: bool,
        chk_labelme: bool,
        chk_yolo_det: bool,
        chk_yolo_seg: bool
    ) -> None:
        """Save the union of multiple masks as a single image."""
        indices = sorted(list(selected_indices))
        
        # Determine output directory
        custom_path = output_path_str.strip()
        if custom_path:
            out_dir = Path(custom_path)
        else:
            out_dir = Path(image_path).parent
        
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save annotations JSON
        Exporter.save_annotations_json(image_path, out_dir, selected_indices, annotations)

        H, W = bgr.shape[:2]
        union_mask = np.zeros((H, W), dtype=np.uint8)
        for i in indices:
            if 0 <= i < len(masks):
                union_mask = np.maximum(union_mask, (masks[i] > 0).astype(np.uint8))

        base_name = f"{image_path.stem}_union"
        
        # Prepare output image (BGRA)
        bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = union_mask * 255

        if crop_bbox:
            # Crop to union bbox
            x, y, w, h = compute_bbox(union_mask > 0)
            crop = bgra[y : y + h, x : x + w]
            img_h, img_w = h, w
            # Annotations relative to cropped image
            boxes = [(0, 0, w, h)]
            poly = compute_polygon(union_mask[y : y + h, x : x + w])
            polys = [poly]
        else:
            # Full image
            crop = bgra
            img_h, img_w = H, W
            x, y, w, h = compute_bbox(union_mask > 0)
            boxes = [(x, y, w, h)]
            poly = compute_polygon(union_mask)
            polys = [poly]

        # Format handling
        fmt = format_str.lower()
        if fmt == "jpg":
            save_img = cv2.cvtColor(crop, cv2.COLOR_BGRA2BGR)
            ext = ".jpg"
        elif fmt == "bmp":
            save_img = cv2.cvtColor(crop, cv2.COLOR_BGRA2BGR)
            ext = ".bmp"
        else:
            save_img = crop
            ext = ".png"

        save_path = out_dir / f"{base_name}{ext}"
        ok, buf = cv2.imencode(ext, save_img)
        if ok:
            save_path.write_bytes(buf.tobytes())
            
            # Write annotation formats
            Exporter.write_yolo_labels(out_dir, base_name, boxes, polys, img_w, img_h, indices, annotations, chk_yolo_det, chk_yolo_seg)
            Exporter.write_coco_json(out_dir, base_name, boxes, polys, img_w, img_h, indices, annotations, chk_coco)
            Exporter.write_voc_xml(out_dir, base_name, boxes, img_w, img_h, save_path.name, indices, annotations, chk_voc)
            Exporter.write_labelme_json(out_dir, base_name, polys, img_w, img_h, save_path.name, indices, annotations, chk_labelme)
            
            QMessageBox.information(parent_widget, "完成", f"已儲存聯集影像至：\n{save_path}")
        else:
            QMessageBox.warning(parent_widget, "失敗", "影像編碼失敗")

    @staticmethod
    def save_indices(
        parent_widget,
        image_path: Path,
        bgr: np.ndarray,
        masks: List[np.ndarray],
        selected_indices: Set[int],
        annotations: Dict[int, int],
        output_path_str: str,
        format_str: str,
        crop_bbox: bool,
        chk_coco: bool,
        chk_voc: bool,
        chk_labelme: bool,
        chk_yolo_det: bool,
        chk_yolo_seg: bool
    ) -> int:
        """Save selected masks as individual images and export combined annotations."""
        indices = sorted(list(selected_indices))
        
        custom_path = output_path_str.strip()
        if custom_path:
            out_dir = Path(custom_path)
        else:
            out_dir = Path(image_path).parent
        
        out_dir.mkdir(parents=True, exist_ok=True)
        
        Exporter.save_annotations_json(image_path, out_dir, selected_indices, annotations)
        
        saved_count = 0
        H, W = bgr.shape[:2]
        
        fmt = format_str.lower()
        ext = f".{fmt}"

        all_boxes = []
        all_polys = []
        valid_indices = []

        for i in indices:
            if not (0 <= i < len(masks)):
                continue
            m = masks[i] > 0
            
            x_orig, y_orig, w_orig, h_orig = compute_bbox(m)
            poly_orig = compute_polygon(m)
            
            all_boxes.append((x_orig, y_orig, w_orig, h_orig))
            all_polys.append(poly_orig)
            valid_indices.append(i)
            
            bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
            bgra[:, :, 3] = m.astype(np.uint8) * 255
            
            base_name = f"{image_path.stem}_{i:03d}"
            
            if crop_bbox:
                crop = bgra[y_orig : y_orig + h_orig, x_orig : x_orig + w_orig]
            else:
                crop = bgra
            
            if fmt in ["jpg", "bmp"]:
                save_img = cv2.cvtColor(crop, cv2.COLOR_BGRA2BGR)
            else:
                save_img = crop
                
            save_path = out_dir / f"{base_name}{ext}"
            ok, buf = cv2.imencode(ext, save_img)
            if ok:
                save_path.write_bytes(buf.tobytes())
                saved_count += 1
        
        if saved_count > 0:
            base_name_orig = image_path.stem
            
            Exporter.write_yolo_labels(out_dir, base_name_orig, all_boxes, all_polys, W, H, valid_indices, annotations, chk_yolo_det, chk_yolo_seg)
            Exporter.write_coco_json(out_dir, base_name_orig, all_boxes, all_polys, W, H, valid_indices, annotations, chk_coco)
            Exporter.write_voc_xml(out_dir, base_name_orig, all_boxes, W, H, image_path.name, valid_indices, annotations, chk_voc)
            Exporter.write_labelme_json(out_dir, base_name_orig, all_polys, W, H, image_path.name, valid_indices, annotations, chk_labelme)
            
            QMessageBox.information(parent_widget, "完成", f"已儲存 {saved_count} 個物件影像及標註檔案")
        else:
            QMessageBox.warning(parent_widget, "提示", "沒有儲存任何檔案")
            
        return saved_count

    @staticmethod
    def save_annotations_json(image_path: Path, out_dir: Path, selected_indices: Set[int], annotations: Dict[int, int]) -> None:
        """Save current annotations to a JSON file."""
        try:
            ann_list = []
            for idx in sorted(selected_indices):
                class_id = annotations.get(idx, 0)
                ann_list.append({
                    "index": idx,
                    "class_id": class_id
                })
            
            data = {
                "image_path": image_path.name,
                "annotations": ann_list
            }
            
            save_path = out_dir / f"{image_path.stem}_annotations.json"
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"已儲存標註狀態: {save_path} ({len(ann_list)} 個物件)")
        except Exception as e:
            logger.error(f"儲存標註狀態失敗: {e}")

    @staticmethod
    def write_coco_json(out_dir, base_name, boxes, polys, img_w, img_h, indices, annotations, enabled):
        if not enabled:
            return
            
        coco_data = {
            "images": [
                {"id": 1, "width": img_w, "height": img_h, "file_name": f"{base_name}.png"}
            ],
            "annotations": [],
            "categories": []
        }
        
        used_classes = set()
        for idx in indices:
            cls_id = annotations.get(idx, 0)
            used_classes.add(cls_id)
            
        for cls_id in sorted(used_classes):
            coco_data["categories"].append({
                "id": cls_id,
                "name": f"class_{cls_id}",
                "supercategory": "object"
            })
        
        for i, (box, poly) in enumerate(zip(boxes, polys)):
            obj_idx = indices[i] if i < len(indices) else 0
            cls_id = annotations.get(obj_idx, 0)
            
            x, y, bw, bh = box
            segmentation = []
            if poly is not None and len(poly) > 0:
                segmentation = [poly.flatten().tolist()]
                
            ann = {
                "id": i + 1,
                "image_id": 1,
                "category_id": cls_id,
                "bbox": [x, y, bw, bh],
                "segmentation": segmentation,
                "area": bw * bh,
                "iscrowd": 0
            }
            coco_data["annotations"].append(ann)
            
        (out_dir / f"{base_name}_coco.json").write_text(json.dumps(coco_data, indent=2), encoding="utf-8")

    @staticmethod
    def write_voc_xml(out_dir, base_name, boxes, w, h, filename, indices, annotations, enabled):
        if not enabled:
            return
            
        import xml.etree.ElementTree as ET
        
        root = ET.Element("annotation")
        ET.SubElement(root, "folder").text = out_dir.name
        ET.SubElement(root, "filename").text = filename
        ET.SubElement(root, "path").text = filename
        
        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(w)
        ET.SubElement(size, "height").text = str(h)
        ET.SubElement(size, "depth").text = "3"
        
        for i, (x, y, bw, bh) in enumerate(boxes):
            obj_idx = indices[i] if i < len(indices) else 0
            cls_id = annotations.get(obj_idx, 0)
            cls_name = f"class_{cls_id}"
            
            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text = cls_name
            ET.SubElement(obj, "pose").text = "Unspecified"
            ET.SubElement(obj, "truncated").text = "0"
            ET.SubElement(obj, "difficult").text = "0"
            
            bndbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(x)
            ET.SubElement(bndbox, "ymin").text = str(y)
            ET.SubElement(bndbox, "xmax").text = str(x + bw)
            ET.SubElement(bndbox, "ymax").text = str(y + bh)
            
        tree = ET.ElementTree(root)
        tree.write(out_dir / f"{base_name}.xml", encoding="utf-8", xml_declaration=True)

    @staticmethod
    def write_labelme_json(out_dir, base_name, polys, w, h, filename, indices, annotations, enabled):
        if not enabled:
            return
            
        shapes = []
        for i, poly in enumerate(polys):
            if poly is not None and len(poly) > 0:
                obj_idx = indices[i] if i < len(indices) else 0
                cls_id = annotations.get(obj_idx, 0)
                cls_name = f"class_{cls_id}"
                
                shape = {
                    "label": cls_name,
                    "points": poly.tolist(),
                    "group_id": None,
                    "shape_type": "polygon",
                    "flags": {}
                }
                shapes.append(shape)
                
        data = {
            "version": "4.5.6",
            "flags": {},
            "shapes": shapes,
            "imagePath": filename,
            "imageData": None,
            "imageHeight": h,
            "imageWidth": w
        }
        
        (out_dir / f"{base_name}_labelme.json").write_text(json.dumps(data, indent=2), encoding="utf-8")

    @staticmethod
    def write_yolo_labels(out_dir, base_name, boxes, polys, img_w, img_h, indices, annotations, det_enabled, seg_enabled):
        if det_enabled:
            lines = []
            for idx, (x, y, w, h) in enumerate(boxes):
                if w <= 0 or h <= 0:
                    continue
                obj_idx = indices[idx] if idx < len(indices) else 0
                cls_id = annotations.get(obj_idx, 0)
                xc = (x + w / 2.0) / img_w
                yc = (y + h / 2.0) / img_h
                nw = w / img_w
                nh = h / img_h
                lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
            if lines:
                (out_dir / f"{base_name}_yolo.txt").write_text("\n".join(lines), encoding="utf-8")

        if seg_enabled:
            lines = []
            for idx, poly in enumerate(polys):
                if poly is None or len(poly) == 0:
                    continue
                obj_idx = indices[idx] if idx < len(indices) else 0
                cls_id = annotations.get(obj_idx, 0)
                pts = []
                for px, py in poly:
                    pts.append(f"{px / img_w:.6f} {py / img_h:.6f}")
                lines.append(f"{cls_id} " + " ".join(pts))
            if lines:
                (out_dir / f"{base_name}_seg.txt").write_text("\n".join(lines), encoding="utf-8")
