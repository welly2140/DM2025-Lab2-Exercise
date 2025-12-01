#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DM Lab2 Emotion Classification - Optimized BERT
優化版本 - 針對 Mean F1 Score 優化

主要改進：
1. 使用更強的預訓練模型（RoBERTa emotion-specific）
2. 改進文本預處理（保留emoji、重複字符等情感特徵）
3. 添加類別權重處理不平衡
4. 優化訓練參數（learning rate scheduler, warmup, longer training）
5. 支持多模型ensemble
6. 使用更長的max_length
7. K-Fold Cross-Validation
8. 兼容新舊版本的 transformers

作者優化版本
最後更新：2024
"""

import os
import json
import re
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    get_linear_schedule_with_warmup,
)
import torch

# ==============================
# 全域設定
# ==============================
DATA_DIR = "KaggleData"

# 模型選擇 - 可以嘗試以下幾個：
# 1. "cardiffnlp/twitter-roberta-base-emotion" - Twitter情感分類專用
# 2. "j-hartmann/emotion-english-distilroberta-base" - 6類情感專用
# 3. "roberta-base" - 通用RoBERTa
# 4. "microsoft/deberta-v3-base" - DeBERTa（更強但更慢）
#MODEL_NAME = "microsoft/deberta-v3-base"
MODEL_NAME = "roberta-large"

# 訓練參數
MAX_LENGTH = 256  # 增加到256
#EPOCHS = 8  # 增加訓練輪數
EPOCHS = 12
TRAIN_BATCH_SIZE = 16  # 降低batch size以便用更大的模型
EVAL_BATCH_SIZE = 32
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1  # warmup 10%的步數
#WEIGHT_DECAY = 0.01
WEIGHT_DECAY = 0.05

# K-Fold設定
USE_KFOLD = True # 是否使用K-Fold（更準確但訓練時間長）
N_FOLDS = 5

OUTPUT_DIR = os.path.join(DATA_DIR, "bert_optimized_output")


# ==============================
# 1. 改進的文本預處理
# ==============================

def clean_text_improved(s: str) -> str:
    """
    改進的文本清理 - 保留情感相關特徵
    
    保留：
    - emoji（情感強相關）
    - 重複字符（如 "soooo happy" 表示強烈程度）
    - 標點符號（！！！表達強烈情感）
    
    移除：
    - URL
    - @mention
    """
    s = str(s)
    
    # 移除URL
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    
    # 移除@mention
    s = re.sub(r"@\w+", " ", s)
    
    # 標準化重複字符（但保留一些）- 例如 "soooooo" -> "sooo"
    # 保留最多3個重複字符，這樣可以保留情感強度信息
    s = re.sub(r'(.)\1{3,}', r'\1\1\1', s)
    
    # 壓縮多空白
    s = re.sub(r"\s+", " ", s).strip()
    
    # 不轉小寫 - 大寫字母可能表示強烈情感（如 "I LOVE THIS"）
    # 如果使用的是 uncased 模型，這裡可以轉小寫
    # s = s.lower()
    
    return s


def load_posts(json_path: str) -> pd.DataFrame:
    """從 final_posts.json 讀取資料"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for item in data:
        post = item["root"]["_source"]["post"]
        rows.append({
            "post_id": post["post_id"],
            "text": post["text"],
        })

    return pd.DataFrame(rows)


def preprocessing(data_dir: str = DATA_DIR) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    讀取、合併、清理資料
    """
    print(">>> Loading data...")
    posts = load_posts(os.path.join(data_dir, "final_posts.json"))
    emotions = pd.read_csv(os.path.join(data_dir, "emotion.csv"))
    split_info = pd.read_csv(os.path.join(data_dir, "data_identification.csv"))

    # 重新命名欄位
    emotions = emotions.rename(columns={"id": "post_id"})
    split_info = split_info.rename(columns={"id": "post_id", "split": "usage"})

    # 合併資料
    df = posts.merge(split_info, on="post_id", how="inner")
    df = df.merge(emotions, on="post_id", how="left")
    df = df.rename(columns={"post_id": "id", "emotion": "label"})

    # 改進的文字清理
    print(">>> Cleaning text (improved)...")
    df["clean_text"] = df["text"].apply(clean_text_improved)

    # 切分 train/test
    train_df = df[df["usage"] == "train"].copy()
    test_df = df[df["usage"] == "test"].copy()

    train_df = train_df.dropna(subset=["clean_text", "label"])
    test_df = test_df.dropna(subset=["clean_text"])

    print(f"Train size: {len(train_df)}")
    print(f"Test  size: {len(test_df)}")
    print(f"\nLabel distribution:")
    print(train_df["label"].value_counts())

    return train_df, test_df


# ==============================
# 2. 計算類別權重
# ==============================

def compute_class_weights(train_df: pd.DataFrame, label2id: Dict[str, int]) -> Dict[int, float]:
    """
    計算類別權重以處理不平衡資料
    """
    labels = train_df["label"].values
    label_ids = np.array([label2id[l] for l in labels])
    
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(label_ids),
        y=label_ids
    )
    
    weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    print(f"Class weights: {weight_dict}")
    
    return weight_dict


# ==============================
# 3. 自定義 Trainer with weighted loss
# ==============================

class WeightedTrainer(Trainer):
    """
    自定義Trainer，使用類別權重
    兼容新舊版本的 transformers
    """
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        計算損失函數
        **kwargs 用於接受新版本 transformers 可能傳入的額外參數（如 num_items_in_batch）
        """
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if self.class_weights is not None:
            # 使用加權的交叉熵損失
            weight_tensor = torch.tensor(
                [self.class_weights[i] for i in range(len(self.class_weights))],
                dtype=torch.float32
            ).to(logits.device)
            loss_fct = torch.nn.CrossEntropyLoss(weight=weight_tensor)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        else:
            loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss


# ==============================
# 4. Dataset 準備
# ==============================

def make_label_mapping(train_df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[int, str]]:
    """建立標籤映射"""
    unique_labels = sorted(train_df["label"].unique())
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {v: k for k, v in label2id.items()}
    print(f"Label mapping: {label2id}")
    return label2id, id2label


def prepare_datasets(train_texts, train_labels, eval_texts, eval_labels, 
                    test_texts, tokenizer, max_length):
    """
    準備 HuggingFace Dataset
    """
    train_df = pd.DataFrame({"clean_text": train_texts, "labels": train_labels})
    eval_df = pd.DataFrame({"clean_text": eval_texts, "labels": eval_labels})
    test_df = pd.DataFrame({"clean_text": test_texts})

    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)
    test_dataset = Dataset.from_pandas(test_df)

    def tokenize(batch):
        return tokenizer(
            batch["clean_text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    train_dataset = train_dataset.map(tokenize, batched=True)
    eval_dataset = eval_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)

    train_dataset = train_dataset.remove_columns(["clean_text"])
    eval_dataset = eval_dataset.remove_columns(["clean_text"])
    test_dataset = test_dataset.remove_columns(["clean_text"])

    # 移除 pandas index
    for col in ["__index_level_0__"]:
        if col in train_dataset.column_names:
            train_dataset = train_dataset.remove_columns([col])
        if col in eval_dataset.column_names:
            eval_dataset = eval_dataset.remove_columns([col])
        if col in test_dataset.column_names:
            test_dataset = test_dataset.remove_columns([col])

    train_dataset.set_format("torch")
    eval_dataset.set_format("torch")
    test_dataset.set_format("torch")

    return train_dataset, eval_dataset, test_dataset


# ==============================
# 5. 評估指標
# ==============================

def compute_metrics(eval_pred):
    """
    計算 Macro F1 (比賽使用的指標)
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    
    # Macro F1 (比賽指標)
    macro_f1 = f1_score(labels, preds, average="macro")
    
    # 也計算 micro 和 weighted 作為參考
    micro_f1 = f1_score(labels, preds, average="micro")
    weighted_f1 = f1_score(labels, preds, average="weighted")
    
    return {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "weighted_f1": weighted_f1,
    }


# ==============================
# 6. 單次訓練函數
# ==============================

def train_single_model(train_dataset, eval_dataset, test_dataset,
                      label2id, id2label, class_weights, 
                      output_dir, seed=42):
    """
    訓練單一模型
    """
    print(f"\n>>> Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,  # 如果使用emotion專用模型可能需要這個
    )

    # 訓練參數 (兼容新舊版本的 transformers)
    try:
        # 嘗試使用新版本參數
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=LEARNING_RATE,
            per_device_train_batch_size=TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=EVAL_BATCH_SIZE,
            num_train_epochs=EPOCHS,
            weight_decay=WEIGHT_DECAY,
            warmup_ratio=WARMUP_RATIO,  # warmup
            
            # 評估策略 (新版本參數)
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            
            # 其他設置
            logging_steps=50,
            save_total_limit=2,
            seed=seed,
            fp16=True,  # 混合精度訓練
            report_to="none",  # 不使用wandb等
        )
    except TypeError:
        # 舊版本 transformers，使用舊參數名
        print(">>> 檢測到舊版本 transformers，使用兼容參數")
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=LEARNING_RATE,
            per_device_train_batch_size=TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=EVAL_BATCH_SIZE,
            num_train_epochs=EPOCHS,
            weight_decay=WEIGHT_DECAY,
            
            # 其他設置
            logging_steps=50,
            save_steps=500,
            save_total_limit=2,
            seed=seed,
            fp16=True,
        )

    # 使用自定義 Trainer（帶類別權重）
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    print(">>> Training...")
    trainer.train()

    print(">>> Final evaluation:")
    eval_result = trainer.evaluate()
    print(f"Eval Macro F1: {eval_result.get('eval_macro_f1', 'N/A'):.4f}")

    print(">>> Predicting on test set...")
    preds = trainer.predict(test_dataset)
    test_logits = preds.predictions
    test_pred_ids = np.argmax(test_logits, axis=-1)
    test_pred_labels = [id2label[i] for i in test_pred_ids]

    return np.array(test_pred_labels), test_logits, eval_result.get('eval_macro_f1', 0.0)


# ==============================
# 7. K-Fold Cross-Validation
# ==============================

def train_kfold(train_df, test_df, tokenizer):
    """
    使用 K-Fold Cross-Validation 訓練多個模型並ensemble
    """
    label2id, id2label = make_label_mapping(train_df)
    class_weights = compute_class_weights(train_df, label2id)
    
    # 準備資料
    train_df = train_df.copy()
    train_df["label_id"] = train_df["label"].map(label2id)
    
    # K-Fold split
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    
    all_test_logits = []
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df["label_id"])):
        print(f"\n{'='*60}")
        print(f"Training Fold {fold + 1}/{N_FOLDS}")
        print(f"{'='*60}")
        
        # 分割資料
        train_texts = train_df.iloc[train_idx]["clean_text"].values
        train_labels = train_df.iloc[train_idx]["label_id"].values
        eval_texts = train_df.iloc[val_idx]["clean_text"].values
        eval_labels = train_df.iloc[val_idx]["label_id"].values
        test_texts = test_df["clean_text"].values
        
        # 準備 dataset
        train_dataset, eval_dataset, test_dataset = prepare_datasets(
            train_texts, train_labels, eval_texts, eval_labels,
            test_texts, tokenizer, MAX_LENGTH
        )
        
        # 訓練
        fold_output_dir = os.path.join(OUTPUT_DIR, f"fold_{fold}")
        _, test_logits, eval_f1 = train_single_model(
            train_dataset, eval_dataset, test_dataset,
            label2id, id2label, class_weights,
            fold_output_dir, seed=42+fold
        )
        
        all_test_logits.append(test_logits)
        fold_scores.append(eval_f1)
    
    print(f"\n{'='*60}")
    print("K-Fold Results:")
    for i, score in enumerate(fold_scores):
        print(f"Fold {i+1}: {score:.4f}")
    print(f"Mean: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    print(f"{'='*60}")
    
    # Ensemble: 平均所有fold的logits
    avg_logits = np.mean(all_test_logits, axis=0)
    final_pred_ids = np.argmax(avg_logits, axis=-1)
    final_pred_labels = [id2label[i] for i in final_pred_ids]
    
    return final_pred_labels


# ==============================
# 8. 單次訓練（不使用K-Fold）
# ==============================

def train_single(train_df, test_df, tokenizer):
    """
    單次訓練（更快但可能略不準確）
    """
    label2id, id2label = make_label_mapping(train_df)
    class_weights = compute_class_weights(train_df, label2id)
    
    # 準備資料
    train_df = train_df.copy()
    train_df["label_id"] = train_df["label"].map(label2id)
    
    # 分割 train/eval
    from sklearn.model_selection import train_test_split
    train_texts, eval_texts, train_labels, eval_labels = train_test_split(
        train_df["clean_text"].values,
        train_df["label_id"].values,
        test_size=0.1,
        random_state=42,
        stratify=train_df["label_id"].values,
    )
    
    test_texts = test_df["clean_text"].values
    
    # 準備 dataset
    train_dataset, eval_dataset, test_dataset = prepare_datasets(
        train_texts, train_labels, eval_texts, eval_labels,
        test_texts, tokenizer, MAX_LENGTH
    )
    
    # 訓練
    pred_labels, _, _ = train_single_model(
        train_dataset, eval_dataset, test_dataset,
        label2id, id2label, class_weights,
        OUTPUT_DIR, seed=42
    )
    
    return pred_labels


# ==============================
# Main
# ==============================

def main():
    print("="*60)
    print("DM Lab2 - Optimized BERT for Emotion Classification")
    print("="*60)
    print(f"Model: {MODEL_NAME}")
    print(f"Max Length: {MAX_LENGTH}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch Size: {TRAIN_BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Use K-Fold: {USE_KFOLD}")
    print("="*60)
    
    # 1. 資料預處理
    train_df, test_df = preprocessing(DATA_DIR)
    
    # 2. Tokenizer
    print(f"\n>>> Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # 3. 訓練
    if USE_KFOLD:
        print(f"\n>>> Using {N_FOLDS}-Fold Cross-Validation")
        test_pred_labels = train_kfold(train_df, test_df, tokenizer)
    else:
        print("\n>>> Using single train/val split")
        test_pred_labels = train_single(train_df, test_df, tokenizer)
    
    # 4. 產生 submission
    submission = pd.DataFrame({
        "id": test_df["id"].values,
        "emotion": test_pred_labels,
    })
    
    out_path = os.path.join(DATA_DIR, "submission_bert_optimized_kfolf_large_epoch.csv")
    submission.to_csv(out_path, index=False)
    
    print(f"\n{'='*60}")
    print(f">>> Submission saved to: {out_path}")
    print("="*60)
    print("\nSubmission preview:")
    print(submission.head(10))
    print(f"\nPrediction distribution:")
    print(submission["emotion"].value_counts())


if __name__ == "__main__":
    main()




