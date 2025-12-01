#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DM Lab2 Emotion Classification - DistilBERT Baseline (for V100)

使用說明：
1. 準備資料夾結構：
   KaggleData/
     ├─ final_posts.json
     ├─ emotion.csv
     └─ data_identification.csv

2. 安裝套件：
   pip install --upgrade transformers datasets accelerate scikit-learn pandas numpy

3. 執行：
   python dm_lab2_bert_baseline.py

4. 輸出：
   會產生 submission_bert.csv，可直接上傳到 Kaggle。
"""

import os
import json
import re
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)


# ==============================
# 全域設定
# ==============================
DATA_DIR = "KaggleData"
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
EPOCHS = 4
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 32
OUTPUT_DIR = os.path.join(DATA_DIR, "bert_output")


# ==============================
# 1. Preprocessing
#    - 讀取 JSON / CSV
#    - 展開 nested 結構
#    - 合併成 train/test
# ==============================

def load_posts(json_path: str) -> pd.DataFrame:
    """
    從 final_posts.json 讀出 post_id 與 text。
    JSON 結構範例：
    [
      {
        "root": {
          "_type": "post",
          "_source": {
            "post": {
              "post_id": "0x61fc95",
              "text": "...",
              "hashtags": []
            }
          }
        }
      },
      ...
    ]
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for item in data:
        post = item["root"]["_source"]["post"]
        rows.append(
            {
                "post_id": post["post_id"],
                "text": post["text"],
            }
        )

    posts = pd.DataFrame(rows)
    return posts


def clean_text(s: str) -> str:
    """
    簡單英文前處理：
    - 轉小寫
    - 移除 URL
    - 移除 @mention
    - 壓縮多空白
    （不刪標點給 BERT，會保留原句子結構比較自然）
    """
    s = str(s).lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"@\w+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def preprocessing(data_dir: str = DATA_DIR) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    讀取 / 合併 / 清理資料，回傳 train_df, test_df

    train_df 欄位：id, text, label, clean_text, usage
    test_df  欄位：id, text, clean_text, usage
    """
    print(">>> Loading posts from JSON...")
    posts = load_posts(os.path.join(data_dir, "final_posts.json"))
    print(f"posts shape: {posts.shape}")

    print(">>> Loading CSVs (emotion, data_identification)...")
    emotions = pd.read_csv(os.path.join(data_dir, "emotion.csv"))
    split_info = pd.read_csv(os.path.join(data_dir, "data_identification.csv"))

    # 根據你提供的 columns：
    # emotion.csv columns: ['id', 'emotion']
    # data_identification.csv columns: ['id', 'split']
    emotions = emotions.rename(columns={"id": "post_id"})
    split_info = split_info.rename(columns={"id": "post_id", "split": "usage"})

    print("emotion head:")
    print(emotions.head())
    print("split_info head:")
    print(split_info.head())

    # 合併 posts + split_info（train/test）+ emotions（label）
    df = posts.merge(split_info, on="post_id", how="inner")
    df = df.merge(emotions, on="post_id", how="left")  # test 會是 NaN

    # 統一欄位名
    df = df.rename(columns={"post_id": "id", "emotion": "label"})
    print("Merged df head:")
    print(df.head())

    # 文字清理
    print(">>> Cleaning text...")
    df["clean_text"] = df["text"].apply(clean_text)

    # 切 train / test
    train_df = df[df["usage"] == "train"].copy()
    test_df = df[df["usage"] == "test"].copy()

    # 防呆：移除缺 label 的 train 資料
    train_df = train_df.dropna(subset=["clean_text", "label"])
    test_df = test_df.dropna(subset=["clean_text"])

    print(f"Train size: {len(train_df)}")
    print(f"Test  size: {len(test_df)}")

    return train_df, test_df


# ==============================
# 2. HF Dataset & Tokenizer 準備
# ==============================

def make_label_mapping(train_df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    將文字 label (anger, joy, ...) 映射成整數 id
    """
    unique_labels = sorted(train_df["label"].unique())
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {v: k for k, v in label2id.items()}
    print("Label mapping:", label2id)
    return label2id, id2label


def remove_if_exists(dataset: Dataset, col_name: str) -> Dataset:
    """
    from_pandas 會自動加一個 __index_level_0__ 欄位，把它移掉
    """
    if col_name in dataset.column_names:
        dataset = dataset.remove_columns(col_name)
    return dataset


def build_hf_datasets(train_df: pd.DataFrame,
                      test_df: pd.DataFrame,
                      tokenizer,
                      max_length: int = MAX_LENGTH):
    """
    將 pandas DataFrame 轉成 HuggingFace Dataset，
    並進行 tokenization。
    """
    # train/test text & label
    train_df = train_df.copy()
    test_df = test_df.copy()

    # 轉成 id label
    label2id, id2label = make_label_mapping(train_df)
    train_df["label_id"] = train_df["label"].map(label2id)

    # 先切出 train/eval（這裡用 0.1 當 eval）
    train_texts, eval_texts, train_labels, eval_labels = train_test_split(
        train_df["clean_text"].values,
        train_df["label_id"].values,
        test_size=0.1,
        random_state=42,
        stratify=train_df["label_id"].values,
    )

    train_df_split = pd.DataFrame({"clean_text": train_texts, "label_id": train_labels})
    eval_df_split = pd.DataFrame({"clean_text": eval_texts, "label_id": eval_labels})

    # 轉成 HF Dataset
    train_dataset = Dataset.from_pandas(train_df_split)
    eval_dataset = Dataset.from_pandas(eval_df_split)
    test_dataset = Dataset.from_pandas(test_df[["clean_text"]])

    # Tokenization 函數
    def tokenize(batch):
        return tokenizer(
            batch["clean_text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    print(">>> Tokenizing train/eval/test ...")
    train_dataset = train_dataset.map(tokenize, batched=True)
    eval_dataset = eval_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)

    # 移除原始文字欄位
    train_dataset = train_dataset.remove_columns(["clean_text"])
    eval_dataset = eval_dataset.remove_columns(["clean_text"])
    test_dataset = test_dataset.remove_columns(["clean_text"])

    # 移除自動 index 欄位（若存在）
    train_dataset = remove_if_exists(train_dataset, "__index_level_0__")
    eval_dataset = remove_if_exists(eval_dataset, "__index_level_0__")
    test_dataset = remove_if_exists(test_dataset, "__index_level_0__")

    # 把 label_id 欄位改名成 labels（Trainer 預設使用）
    train_dataset = train_dataset.rename_column("label_id", "labels")
    eval_dataset = eval_dataset.rename_column("label_id", "labels")

    # 設定 tensor 格式
    train_dataset.set_format("torch")
    eval_dataset.set_format("torch")
    test_dataset.set_format("torch")

    return train_dataset, eval_dataset, test_dataset, label2id, id2label


# ==============================
# 3. 模型訓練 & 評估 & 預測
# ==============================

def compute_metrics(eval_pred):
    """
    Trainer 用來計算 Macro F1 的函式
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    macro_f1 = f1_score(labels, preds, average="macro")
    return {"macro_f1": macro_f1}


def train_and_predict(train_dataset: Dataset,
                      eval_dataset: Dataset,
                      test_dataset: Dataset,
                      label2id: Dict[str, int],
                      id2label: Dict[int, str],
                      output_dir: str = OUTPUT_DIR) -> np.ndarray:
    """
    使用 DistilBERT 訓練並對 test 做預測，回傳 test_pred_label_words
    """
    print(">>> Loading tokenizer & model:", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    print(">>> Setting up TrainingArguments...")
    # 為了相容舊版 transformers，不使用 evaluation_strategy 這種較新的參數
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        fp16=True,              # V100 可用
        logging_steps=100,
        save_steps=500,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    print(">>> Start training...")
    trainer.train()

    print(">>> Evaluate on eval_dataset:")
    eval_result = trainer.evaluate()
    print("Eval results:", eval_result)

    print(">>> Predict on test_dataset...")
    preds = trainer.predict(test_dataset)
    test_logits = preds.predictions
    test_pred_ids = np.argmax(test_logits, axis=-1)
    test_pred_labels = [id2label[i] for i in test_pred_ids]

    return np.array(test_pred_labels)


# ==============================
# main
# ==============================

def main():
    # 1. 前處理
    train_df, test_df = preprocessing(DATA_DIR)

    # 2. 準備 tokenizer & HF Dataset
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset, eval_dataset, test_dataset, label2id, id2label = build_hf_datasets(
        train_df, test_df, tokenizer, max_length=MAX_LENGTH
    )

    # 3. 訓練 & 預測
    test_pred_labels = train_and_predict(
        train_dataset,
        eval_dataset,
        test_dataset,
        label2id,
        id2label,
        output_dir=OUTPUT_DIR,
    )

    # 4. 產生 submission
    submission = pd.DataFrame(
        {
            "id": test_df["id"].values,
            "emotion": test_pred_labels,
        }
    )

    out_path = "submission_bert.csv"
    submission.to_csv(out_path, index=False)
    print(">>> Saved submission to:", out_path)
    print(submission.head())


if __name__ == "__main__":
    main()
