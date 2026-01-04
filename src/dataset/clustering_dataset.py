from __future__ import annotations

import os
import glob
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


# =========================================================
# Unicode -> class id
# =========================================================

HIRAGANA_RANGE = (0x3040, 0x309F)
KATAKANA_RANGE = (0x30A0, 0x30FF)


def parse_unicode_cell(u: object) -> Optional[int]:
    """
    DataFrameの 'Unicode' セルが以下のどれでも int codepoint にする:
      - "U+3042" 形式
      - "3042"   形式
      - "あ"     1文字
      - NaN / None -> None
    """
    if u is None:
        return None
    if isinstance(u, float) and pd.isna(u):
        return None

    s = str(u).strip()
    if not s:
        return None

    # "U+3042"
    if s.upper().startswith("U+"):
        try:
            return int(s[2:], 16)
        except ValueError:
            return None

    # "3042" (hexっぽい)
    if all(c in "0123456789abcdefABCDEF" for c in s) and len(s) in (4, 5, 6):
        try:
            return int(s, 16)
        except ValueError:
            pass

    # "あ" のような 1 文字
    if len(s) == 1:
        return ord(s)

    return None


def is_kana(codepoint: int) -> bool:
    return (HIRAGANA_RANGE[0] <= codepoint <= HIRAGANA_RANGE[1]) or (
        KATAKANA_RANGE[0] <= codepoint <= KATAKANA_RANGE[1]
    )


@dataclass
class UnicodeClassMapper:
    """
    クラス定義:
      - class 0: 漢字（その他全部をまとめる）
      - class 1..: ひらがな/カタカナ を Unicode codepoint ごとにユニーク割当
    """
    kanji_class_id: int = 0
    _kana_to_id: Dict[int, int] = None  # codepoint -> class_id

    def __post_init__(self) -> None:
        if self._kana_to_id is None:
            self._kana_to_id = {}

    @property
    def num_classes(self) -> int:
        # 0:kanji + unique kana
        return 1 + len(self._kana_to_id)

    def fit_from_dfs(self, dfs: Iterable[pd.DataFrame], unicode_col: str = "Unicode") -> None:
        """データ全体から出現するかなを収集し、安定した class_id を確定する。"""
        kana_set = set()
        for df in dfs:
            if unicode_col not in df.columns:
                continue
            for v in df[unicode_col].tolist():
                cp = parse_unicode_cell(v)
                if cp is None:
                    continue
                if is_kana(cp):
                    kana_set.add(cp)

        # 安定化のため昇順で割当
        for cp in sorted(kana_set):
            if cp not in self._kana_to_id:
                self._kana_to_id[cp] = 1 + len(self._kana_to_id)

    def to_class_id(self, unicode_cell: object) -> int:
        cp = parse_unicode_cell(unicode_cell)
        if cp is None:
            return self.kanji_class_id
        if not is_kana(cp):
            return self.kanji_class_id
        # fit済み想定。未知かなが来た場合は追加（ただしクラス数が動くので基本はfit推奨）
        if cp not in self._kana_to_id:
            self._kana_to_id[cp] = 1 + len(self._kana_to_id)
        return self._kana_to_id[cp]


# =========================================================
# Mask generation (bbox only)
# =========================================================

def _clip_xyxy(x1: int, y1: int, x2: int, y2: int, W: int, H: int) -> Tuple[int, int, int, int]:
    x1 = max(0, min(x1, W))
    x2 = max(0, min(x2, W))
    y1 = max(0, min(y1, H))
    y2 = max(0, min(y2, H))
    return x1, y1, x2, y2


def make_text_region_mask(
    ann_df: pd.DataFrame,
    H: int,
    W: int,
    *,
    x_col: str = "X",
    y_col: str = "Y",
    w_col: str = "Width",
    h_col: str = "Height",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    本文領域マスク: bbox領域を 1 とする 2値マスク (H, W)
    """
    mask = torch.zeros((H, W), dtype=dtype)
    if ann_df is None or len(ann_df) == 0:
        return mask

    for _, row in ann_df.iterrows():
        x = int(row[x_col])
        y = int(row[y_col])
        w = int(row[w_col])
        h = int(row[h_col])

        x1, y1 = x, y
        x2, y2 = x + w, y + h
        x1, y1, x2, y2 = _clip_xyxy(x1, y1, x2, y2, W, H)
        if x1 < x2 and y1 < y2:
            mask[y1:y2, x1:x2] = 1.0
    return mask


def make_affinity_mask(
    ann_df: pd.DataFrame,
    H: int,
    W: int,
    *,
    expand_ratio_y: float = 0.2,
    x_col: str = "X",
    y_col: str = "Y",
    w_col: str = "Width",
    h_col: str = "Height",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    アフィニティマスク:
      - bboxを縦方向に expand_ratio_y だけ上下に拡張
      - 拡張bbox同士の重なり領域を 1 とする

    実装は count-map を作って (count>=2) を affinity とする（O(N)描画 + threshold）。
    """
    if ann_df is None or len(ann_df) == 0:
        return torch.zeros((H, W), dtype=dtype)

    count = torch.zeros((H, W), dtype=torch.int16)

    for _, row in ann_df.iterrows():
        x = int(row[x_col])
        y = int(row[y_col])
        w = int(row[w_col])
        h = int(row[h_col])

        dy = int(round(h * expand_ratio_y))

        x1 = x
        x2 = x + w
        y1 = y - dy
        y2 = y + h + dy

        x1, y1, x2, y2 = _clip_xyxy(x1, y1, x2, y2, W, H)
        if x1 < x2 and y1 < y2:
            count[y1:y2, x1:x2] += 1

    affinity = (count >= 2).to(dtype)
    return affinity


def make_final_text_mask(text_region_mask: torch.Tensor, affinity_mask: torch.Tensor) -> torch.Tensor:
    """
    final_text_mask = text_region_mask - affinity_mask を 2値化して返す
    """
    final = (text_region_mask - affinity_mask).clamp(min=0.0)
    final = (final > 0.5).to(text_region_mask.dtype)
    return final


# =========================================================
# Pixel-level one-hot label map
# =========================================================

def make_one_hot_char_label_map(
    ann_df: pd.DataFrame,
    H: int,
    W: int,
    mapper: UnicodeClassMapper,
    *,
    unicode_col: str = "Unicode",
    x_col: str = "X",
    y_col: str = "Y",
    w_col: str = "Width",
    h_col: str = "Height",
    affinity_mask: Optional[torch.Tensor] = None,
    mask_out_affinity: bool = True,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    ピクセル単位 one-hot 文字種ラベルマップ: (H, W, C)
      - bbox領域に該当クラスを 1
      - mask_out_affinity=True の場合、affinity領域は 0 に戻す（曖昧領域の除外）
    """
    C = mapper.num_classes
    label = torch.zeros((H, W, C), dtype=dtype)

    if ann_df is None or len(ann_df) == 0:
        return label

    for _, row in ann_df.iterrows():
        x = int(row[x_col])
        y = int(row[y_col])
        w = int(row[w_col])
        h = int(row[h_col])
        cls = mapper.to_class_id(row.get(unicode_col, None))

        x1, y1 = x, y
        x2, y2 = x + w, y + h
        x1, y1, x2, y2 = _clip_xyxy(x1, y1, x2, y2, W, H)
        if x1 < x2 and y1 < y2:
            # 重なりがあっても最大値で 1 を保持
            label[y1:y2, x1:x2, cls] = 1.0

    if mask_out_affinity and affinity_mask is not None:
        # affinity領域の全チャネルを 0
        am = affinity_mask.to(dtype)  # (H, W)
        label[am > 0.5] = 0.0

    return label


# =========================================================
# Image / annotation loading (project structure: folder/images + folder/*.csv)
# =========================================================

def list_folders(root_dir: str) -> List[str]:
    return [
        name for name in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, name))
    ]


def load_doc_csv(doc_dir: str) -> pd.DataFrame:
    csv_files = glob.glob(os.path.join(doc_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"CSV not found in: {doc_dir}")
    return pd.read_csv(csv_files[0], encoding="utf-8")


def list_images(images_dir: str) -> List[str]:
    return [
        name for name in os.listdir(images_dir)
        if os.path.isfile(os.path.join(images_dir, name))
    ]


def image_id_from_filename(name: str) -> str:
    # "xxxx.png" -> "xxxx"
    return os.path.splitext(name)[0]


def load_and_resize_pad_image(
    image_path: str,
    *,
    target_width: int,
    patch_size: int = 256,
) -> Tuple[torch.Tensor, int, int, int, int]:
    """
    画像を読み込み:
      - 幅 target_width に等倍スケール（縦横比維持）
      - 高さを patch_size の倍数に下方向ゼロパディング
    return:
      img: (3, H_pad, W) float32 [0..1]
      orig_w, orig_h, new_w, new_h (new_hはパディング前のリサイズ後高さ)
    """
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size

    new_w = int(target_width)
    new_h = int(orig_h * new_w / orig_w)

    img = img.resize((new_w, new_h), Image.BILINEAR)

    padded_h = ((new_h + patch_size - 1) // patch_size) * patch_size
    if padded_h > new_h:
        padded = Image.new("RGB", (new_w, padded_h), (0, 0, 0))
        padded.paste(img, (0, 0))
        img = padded

    t = torch.from_numpy(__import__("numpy").array(img)).permute(2, 0, 1).float() / 255.0
    return t, orig_w, orig_h, new_w, new_h


def scale_annotation_df(
    ann_df: pd.DataFrame,
    *,
    scale: float,
    x_col: str = "X",
    y_col: str = "Y",
    w_col: str = "Width",
    h_col: str = "Height",
) -> pd.DataFrame:
    """bboxを画像のスケールに合わせて整数化して返す（元dfは壊さない）。"""
    if ann_df is None or len(ann_df) == 0:
        return ann_df

    df = ann_df.copy()
    df[x_col] = (df[x_col] * scale).round().astype(int)
    df[y_col] = (df[y_col] * scale).round().astype(int)
    df[w_col] = (df[w_col] * scale).round().astype(int)
    df[h_col] = (df[h_col] * scale).round().astype(int)
    return df


# =========================================================
# Dataset
# =========================================================

class ClusteringDataset(Dataset):
    """
    返り値（1サンプル）:
      {
        "image": (3,H,W) float32 0..1
        "final_text_mask": (H,W) float32 {0,1}
        "affinity_mask": (H,W) float32 {0,1}
        "label_map": (H,W,C) float32 {0,1}
        "meta": dict
      }
    """
    def __init__(
        self,
        root_dir: str,
        *,
        canvas_width: int = 2048,
        patch_size: int = 256,
        test_mode: bool = False,
        test_docs: Sequence[str] = (),
        mapper: Optional[UnicodeClassMapper] = None,
        image_dirname: str = "images",
        image_key_col: str = "Image",
        unicode_col: str = "Unicode",
    ) -> None:
        self.root_dir = root_dir
        self.canvas_width = int(canvas_width)
        self.patch_size = int(patch_size)
        self.test_mode = bool(test_mode)
        self.test_docs = set(test_docs)
        self.image_dirname = image_dirname
        self.image_key_col = image_key_col
        self.unicode_col = unicode_col

        # 1) フォルダを走査して samples を作る + dfを保持
        self.samples: List[Tuple[str, str, pd.DataFrame]] = []  # (doc_id, image_path, doc_df)
        doc_dfs: List[pd.DataFrame] = []

        for doc_id in list_folders(root_dir):
            in_test = doc_id in self.test_docs
            if self.test_mode and not in_test:
                continue
            if (not self.test_mode) and in_test:
                continue

            doc_dir = os.path.join(root_dir, doc_id)
            images_dir = os.path.join(doc_dir, image_dirname)
            if not os.path.isdir(images_dir):
                continue

            df = load_doc_csv(doc_dir)
            doc_dfs.append(df)

            for fname in list_images(images_dir):
                img_path = os.path.join(images_dir, fname)
                self.samples.append((doc_id, img_path, df))

        # 2) mapper が無ければ作って fit（クラス数固定化のため推奨）
        self.mapper = mapper or UnicodeClassMapper()
        # fit済みでない場合は全dfからかなを確定
        if self.mapper.num_classes == 1:
            self.mapper.fit_from_dfs(doc_dfs, unicode_col=self.unicode_col)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        doc_id, image_path, df = self.samples[idx]
        image_name = os.path.basename(image_path)
        image_id = image_id_from_filename(image_name)

        # 画像ロード（リサイズ+パディング）
        img, orig_w, orig_h, new_w, new_h = load_and_resize_pad_image(
            image_path, target_width=self.canvas_width, patch_size=self.patch_size
        )
        _, H, W = img.shape

        # この画像の annotation 抽出
        if self.image_key_col in df.columns:
            # print(f'check {image_id=}')
            # print(f'check {self.image_key_col=}')
            # print(f'check {df[self.image_key_col]=}')
            # print(f'check {df[df[self.image_key_col] == image_id]}')
            ann_df = df[df[self.image_key_col] == image_id]
        else:
            ann_df = df  # 互換性のため（必要ならここは要調整）
        # print(f'{ann_df=}')
        # bbox スケーリング（x/y/w/h 全部同じ比率でOK：縦横比維持の等倍拡縮）
        scale = new_w / orig_w
        ann_df_s = scale_annotation_df(ann_df, scale=scale)

        # マスク生成
        text_region_mask = make_text_region_mask(ann_df_s, H=H, W=W)
        affinity_mask = make_affinity_mask(ann_df_s, H=H, W=W, expand_ratio_y=0.2)
        final_text_mask = make_final_text_mask(text_region_mask, affinity_mask)

        # one-hot ラベル
        label_map = make_one_hot_char_label_map(
            ann_df_s,
            H=H,
            W=W,
            mapper=self.mapper,
            unicode_col=self.unicode_col,
            affinity_mask=affinity_mask,
            mask_out_affinity=True,
        )

        return {
            "image": img,  # (3,H,W)
            "final_text_mask": final_text_mask,  # (H,W)
            "affinity_mask": affinity_mask,  # (H,W)
            "label_map": label_map,  # (H,W,C)
            "meta": {
                "doc_id": doc_id,
                "image_id": image_id,
                "image_path": image_path,
                "num_classes": self.mapper.num_classes,
            },
        }