from PIL import Image
import numpy as np
import torch
from matplotlib import pyplot as plt
import glob
import pandas as pd
import os




# ...existing code...
from torch.utils.data import Dataset, DataLoader

class KuzushijiCanvasDataset(Dataset):
    def get_folders(self, target_dir):
        # ディレクトリだけを抽出
        folders = [
            name for name in os.listdir(target_dir)
            if os.path.isdir(os.path.join(target_dir, name))
        ]
        return folders

    def get_images_path(self, target_dir, folders_id):
        target_doc = target_dir + '/' + folders_id
        target_images_path = target_doc + '/images'
        return target_doc, target_images_path

    def get_doc_df(self, target_doc):
            # target_doc 内の CSV ファイルをすべて取得
        csv_files = glob.glob(os.path.join(target_doc, "*.csv"))

        # 1つ目の CSV を読み込む例
        if csv_files:
            df = pd.read_csv(csv_files[0], encoding="utf-8")  # 必要なら encoding を調整
            # display(df)
        else:
            print("CSV ファイルが見つかりませんでした。")
        return df

    def get_images_id(self, target_dir):
        # ディレクトリだけを抽出
        folders = [
            name for name in os.listdir(target_dir)
            if os.path.isfile(os.path.join(target_dir, name))
        ]
        return folders

    def get_image_path(self, target_images_path, image_id):
        image_path = target_images_path + '/' + image_id
        return image_path
    def get_image_id(self, image_path):
        return image_path.split('.')[0]

    def get_annotation_data(self, df, image_id):
        annotation_data = df[df['Image'] == image_id]
        return annotation_data

    def get_image(self, image_path, target_width=1024, patch_size=256):
        """
        image_path から画像を読み込み、
        - 横幅を target_width にリサイズ
        - 高さを patch_size(=256) の倍数になるように下側にゼロパディング
        を行ってから torch.Tensor(C, H, W) を返す。
        """
        # 画像読み込み (RGB)
        img = Image.open(image_path).convert("RGB")

        # 元のサイズ
        orig_w, orig_h = img.size  # (W, H)

        # 横幅を target_width に変更し、高さは縦横比を維持してスケーリング
        new_w = target_width
        new_h = int(orig_h * new_w / orig_w)
        img = img.resize((new_w, new_h), Image.BILINEAR)

        # 高さを patch_size の倍数にパディング
        padded_h = ((new_h + patch_size - 1) // patch_size) * patch_size  # 切り上げ
        pad_bottom = padded_h - new_h

        if pad_bottom > 0:
            # (W, H+C) へのパディング用キャンバス（黒で埋める）
            padded = Image.new("RGB", (new_w, padded_h), (0, 0, 0))
            padded.paste(img, (0, 0))
            img = padded

        # Tensor 変換: (H, W, C) -> (C, H, W), float32
        tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float()
        return tensor, orig_w, orig_h

    def draw_rects_from_df(self, df, canvas_width, orig_w, orig_h):
        """
        df から矩形を読み取り、(H, W) の torch.float32 テンソルを返す。
        矩形領域は 1、それ以外は 0。
        """
        x_col="X"
        y_col="Y"
        w_col="Width"
        h_col="Height"

        org_height = orig_h
        org_width = orig_w
        canvas_height = int(org_height * canvas_width / org_width)
        canvas = torch.zeros((canvas_height, canvas_width), dtype=torch.float32)

        rate = canvas_width / org_width

        for _, row in df.iterrows():
            x = int(row[x_col] * rate)
            y = int(row[y_col] * rate)
            w = int(row[w_col] * rate)
            h = int(row[h_col] * rate)

            # 範囲チェック（キャンバス外にはみ出る場合の対策）
            x1 = max(x, 0)
            y1 = max(y, 0)
            x2 = min(x + w, canvas_width)
            y2 = min(y + h, canvas_height)

            if x1 < x2 and y1 < y2:
                canvas[y1:y2, x1:x2] = 1.0

        return canvas


    def draw_glue_from_df(self, df, canvas_width, orig_w, orig_h):
        """
        df から矩形を読み取り、(H, W) の torch.float32 テンソルを返す。
        矩形領域は 1、それ以外は 0。
        """
        x_col="X"
        y_col="Y"
        w_col="Width"
        h_col="Height"

        org_height = orig_h
        org_width = orig_w
        canvas_height = int(org_height * canvas_width / org_width)
        canvas = torch.zeros((canvas_height, canvas_width), dtype=torch.float32)

        rate = canvas_width / org_width

        for _, row in df.iterrows():
            x = int(row[x_col] * rate)
            y = int(row[y_col] * rate)
            w = int(row[w_col] * rate)
            h = int(row[h_col] * rate)
            min_l = int(min(w, h)*0.2)

            # 範囲チェック（キャンバス外にはみ出る場合の対策）
            x1 = max(x, 0)
            y1 = max(y, 0) - min_l
            x2 = min(x + w , canvas_width)
            y2 = min(y + h + min_l, canvas_height)

            if x1 < x2 and y1 < y2:
                canvas[y1:y2, x1:x2] += 1.0
        canvas = (canvas == 2).to(torch.float32)  # or .to(tensor.dtype)

        return canvas
    def __init__(self, root_dir, canvas_width=1024, transform=None, test_mode= False, test_docs=[]):
        """
        root_dir: char_sep_datas のパス
        canvas_width: キャンバス幅（高さは画像縦横比から計算）
        transform: 画像に対する torchvision.transforms など（任意）
        """
        self.root_dir = root_dir
        self.canvas_width = canvas_width
        self.transform = transform
        test_docs = set(test_docs)

        # ファイル一覧を作る
        # 形式: [(image_path, df_for_that_folder), ...] とする
        self.samples = []

        folders = self.get_folders(root_dir)
        for folder in folders:
            if test_mode:
                if folder not in test_docs:
                    continue
            else:
                if folder in test_docs:
                    continue
            # target_doc : /char_sep_datas/xxxxxx
            # target_image_path : /char_sep_datas/xxxxxx/images
            target_doc, target_images_path = self.get_images_path(root_dir, folder)
            df = self.get_doc_df(target_doc)
            image_name_list = self.get_images_id(target_images_path)
            for image_name in image_name_list:
                image_path = self.get_image_path(target_images_path, image_name)
                image_id = self.get_image_id(image_name)  # 拡張子なし
                # DataFrame 側の Image 列と合わせる (必要に応じて調整)
                # 例: df['Image'] が拡張子なしならそのまま、拡張子付きなら ".png" を足すなど
                self.samples.append({
                    "image_path": image_path,
                    "image_id": image_id,
                    "df": df
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = sample["image_path"]
        image_id = sample["image_id"]
        df = sample["df"]

        # 画像ロード
        img, orig_w, orig_h = self.get_image(image_path, target_width=self.canvas_width,patch_size=256)  # (C, H, W) tensor

        img = img / 255.0  # 0-1 正規化
        # 必要なら変換
        if self.transform is not None:
            img = self.transform(img)

        # アノテーション抽出
        ann_df = self.get_annotation_data(df, image_id)

        # キャンバス生成
        canvas_main = self.draw_rects_from_df(ann_df, self.canvas_width, orig_w, orig_h)
        canvas_glue = self.draw_glue_from_df(ann_df, self.canvas_width, orig_w, orig_h)
        canvas = torch.stack([canvas_main, canvas_glue], dim=0)  # (2, Hc, Wc)

        return img, canvas

target_dir = r"C:/Users/kotat/MyPrograms/MyKuzushiji/kuzushiji-recognition/char_sep_datas"
test_docs = [
    '200021637',
    '100249371',
    '100249537',
    '200005598',
    '200014740',
    '200020019',
    '200021712',
    '200021869'
]

# データセットとデータローダの作成例
train_dataset = KuzushijiCanvasDataset(
    root_dir=target_dir,
    canvas_width=2048,
    transform=None,  # ここに正規化などを入れてもよい
    test_mode=False,
    test_docs=test_docs
)
test_dataset = KuzushijiCanvasDataset(
    root_dir=target_dir,
    canvas_width=2048,
    transform=None,  # ここに正規化などを入れてもよい
    test_mode=True,
    test_docs=test_docs
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=0  # Windows なら最初は 0 推奨
)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0  # Windows なら最初は 0 推奨
)
