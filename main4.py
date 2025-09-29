# -*- coding: utf-8 -*-
"""
WB Image Relevance — ViT-H/14 @224, fine-tune "last2" + bagging + text-ensemble
-------------------------------------------------------------------------------
- Модель: open_clip CLIP-ViT-H-14 (224).
- Частичная разморозка: последние 2 блока визуального трансформера + ln_post + logit_scale.
- Обучение по BCE на логитах: logit = (cosine(image,text) * exp(logit_scale)).
- CV: GroupKFold по card_identifier_id, честный OOF, сохранение лучших чекпойнтов.
- Улучшения:
    (1) Бэггинг по сид-значениям: SEEDS = [42, 43, 44] (усредняем предсказания).
    (2) Энсамбль текстовых шаблонов (RU/EN, краткий/подробный) — усреднение эмбеддингов.
- Валидация/тест: TTA (оригинал + горизонтальный флип).
- Атомарное сохранение чекпойнтов (tmp→replace), совместимо с NFS.
- Выход: submission.csv, oof_report_vith14_last2_bag.json, чекпойнты в _outputs/.

Ожидаемая структура данных:
BASE_DIR/
  ├─ train.csv (id, card_identifier_id, title, description, label)
  ├─ test.csv  (id, card_identifier_id, title, description)
  └─ images/{id}.jpg
"""

import os
import json
import time
import math
import logging
import traceback
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

from tqdm.auto import tqdm

# ----------------------------
# Конфигурация (без argparse)
# ----------------------------
BASE_DIR = "clip_model"
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
TEST_CSV  = os.path.join(BASE_DIR, "test.csv")
IMAGES_DIR = os.path.join(BASE_DIR, "images")
OUTPUTS_DIR = os.path.join(BASE_DIR, "_outputs")
SUBMISSION_PATH = os.path.join(BASE_DIR, "submission.csv")
OOF_JSON_PATH   = os.path.join(BASE_DIR, "oof_report_vith14_last2_bag.json")

os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Модель/пре-тренировка в open_clip
MODEL_NAME = "ViT-H-14"
PRETRAINED = "laion2b_s32b_b79k"

# Тренировка
N_FOLDS = 5
EPOCHS = 2
BATCH_SIZE = 96             # при OOM уменьшайте до 64/48/32
GRAD_ACCUM_STEPS = 1
INIT_LR = 1e-5
WEIGHT_DECAY = 0.1
WARMUP_RATIO = 0.1

# Бэггинг
SEEDS = [42, 43, 44]

# TTA (валидация/тест)
USE_TTA = True
MIXED_PRECISION = True

# Текстовые шаблоны (усредняем эмбеддинги по шаблонам)
POS_TEMPLATES = [
    "Название товара: {t}. Описание товара: {d}. Реальное фото предмета без текста, схем и таблиц.",
    "Фото товара: {t}. {d}. Без баннеров, без надписей на изображении.",
    "Product photograph: {t}. {d}. Natural photo, no text overlays, no charts or tables.",
    "Image of the product: {t}. {d}. Clean background, no banners or captions."
]

# Логирование
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("vith14_last2_bag")

# ----------------------------
# open_clip
# ----------------------------
try:
    import open_clip
except Exception as e:
    logger.error("Не найден пакет open_clip_torch. Установите: pip install open_clip_torch")
    raise

# ----------------------------
# Утилиты
# ----------------------------
def set_seeds(seed: int = 42):
    try:
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = get_device()
logger.info(f"DEVICE={DEVICE}")

def read_csv_auto(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        if df.shape[1] == 1:
            df = pd.read_csv(path, sep=";")
        return df
    except Exception as e:
        logger.error(f"Ошибка чтения CSV {path}: {e}")
        raise

def safe_image_open(path: str) -> Image.Image:
    try:
        with Image.open(path) as img:
            return img.convert("RGB")
    except (FileNotFoundError, UnidentifiedImageError) as e:
        logger.warning(f"Проблема с изображением {path}: {e}")
        return Image.fromarray(np.full((224, 224, 3), 255, dtype=np.uint8))

def atomic_torch_save(state_obj, final_path: str):
    try:
        d = os.path.dirname(final_path)
        os.makedirs(d, exist_ok=True)
        tmp_path = final_path + ".tmp"
        torch.save(state_obj, tmp_path, _use_new_zipfile_serialization=False)
        os.replace(tmp_path, final_path)
        return True
    except Exception as e:
        logger.error(f"Не удалось сохранить чекпойнт: {e}")
        try:
            base = os.path.basename(final_path)
            tmp_path2 = os.path.join("/tmp", base + ".tmp")
            torch.save(state_obj, tmp_path2, _use_new_zipfile_serialization=False)
            os.replace(tmp_path2, final_path)
            return True
        except Exception as e2:
            logger.error(f"Повторная попытка сохранения провалилась: {e2}")
            return False

# ----------------------------
# Модель
# ----------------------------
def load_model_and_preprocess():
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED, device=DEVICE
    )
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    model.eval()
    return model, preprocess, tokenizer

def unfreeze_last2_visual_blocks(model: torch.nn.Module):
    try:
        for p in model.parameters():
            p.requires_grad = False

        resblocks = model.visual.transformer.resblocks
        for blk in resblocks[-2:]:
            for p in blk.parameters():
                p.requires_grad = True

        if hasattr(model.visual, "ln_post"):
            for p in model.visual.ln_post.parameters():
                p.requires_grad = True

        if hasattr(model, "logit_scale") and isinstance(model.logit_scale, torch.nn.Parameter):
            model.logit_scale.requires_grad = True
        else:
            try:
                model.logit_scale.requires_grad_(True)
            except Exception:
                pass

        trainable = sum(p.requires_grad for p in model.parameters())
        total = sum(1 for _ in model.parameters())
        logger.info(f"Градиентов включено: {trainable}/{total} параметров")
    except Exception as e:
        logger.error(f"Не удалось частично разморозить: {e}. Размораживаем всё.")
        for p in model.parameters():
            p.requires_grad = True

# ----------------------------
# Тексты → эмбеддинги (с ансамблем шаблонов)
# ----------------------------
@torch.no_grad()
def encode_texts(model, tokenizer, texts: List[str], batch: int = 256) -> np.ndarray:
    embs = []
    rng = range(0, len(texts), batch)
    for i in tqdm(rng, desc="Тексты→эмбеддинги", leave=False):
        chunk = texts[i:i+batch]
        tok = tokenizer(chunk)
        if DEVICE != "cpu":
            tok = tok.to(DEVICE)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEVICE=="cuda" and MIXED_PRECISION)):
            feat = model.encode_text(tok)
        feat = F.normalize(feat, dim=-1).detach().cpu().numpy()
        embs.append(feat)
    return np.concatenate(embs, axis=0) if embs else np.zeros((0, 1024), dtype=np.float32)

def build_texts_for_df_multi(df: pd.DataFrame, templates: List[str]) -> Tuple[List[str], int]:
    all_texts = []
    for _, r in df.iterrows():
        t = str(r.get("title", "") or "")
        d = str(r.get("description", "") or "")
        for tpl in templates:
            all_texts.append(tpl.format(t=t, d=d))
    return all_texts, len(templates)

# ----------------------------
# Датасеты
# ----------------------------
class WBTrainDataset(Dataset):
    def __init__(self, df: pd.DataFrame, txt_emb: np.ndarray, preprocess, train: bool):
        self.df = df.reset_index(drop=True)
        self.txt = torch.from_numpy(txt_emb).float()
        self.preprocess = preprocess
        self.train = train

    def __len__(self): return len(self.df)

    def __getitem__(self, idx: int):
        rid = int(self.df.loc[idx, "id"])
        lbl = float(self.df.loc[idx, "label"])
        img_path = os.path.join(IMAGES_DIR, f"{rid}.jpg")
        img = safe_image_open(img_path)
        if self.train and np.random.rand() < 0.5:
            img = ImageOps.mirror(img)
        img_t = self.preprocess(img)
        txt_t = self.txt[idx]
        return img_t, txt_t, torch.tensor(lbl, dtype=torch.float32)

class WBInferDataset(Dataset):
    def __init__(self, df: pd.DataFrame, txt_emb: np.ndarray, preprocess, tta: bool):
        self.df = df.reset_index(drop=True)
        self.txt = torch.from_numpy(txt_emb).float()
        self.preprocess = preprocess
        self.tta = tta

    def __len__(self): return len(self.df)

    def __getitem__(self, idx: int):
        rid = int(self.df.loc[idx, "id"])
        img_path = os.path.join(IMAGES_DIR, f"{rid}.jpg")
        img = safe_image_open(img_path)
        if self.tta:
            x1 = self.preprocess(img)
            x2 = self.preprocess(ImageOps.mirror(img))
            x = torch.stack([x1, x2], dim=0)  # [2,3,H,W]
        else:
            x = self.preprocess(img).unsqueeze(0)      # [1,3,H,W]
        txt_t = self.txt[idx]
        return x, txt_t

# ----------------------------
# Обучение / валидация / инференс
# ----------------------------
def make_optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=INIT_LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.98))

def cosine_warmup_lr(step, total_steps, base_lr, warmup_steps):
    if step < warmup_steps:
        return base_lr * (step / max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    min_lr = base_lr * 0.1
    return min_lr + 0.5*(base_lr - min_lr)*(1 + math.cos(math.pi * progress))

def train_one_fold(fold_id: int, train_df: pd.DataFrame, val_df: pd.DataFrame,
                   txt_train: np.ndarray, txt_val: np.ndarray,
                   preprocess, seed: int) -> Tuple[str, float, np.ndarray]:
    model, preprocess_loaded, _ = load_model_and_preprocess()
    preprocess_used = preprocess_loaded
    unfreeze_last2_visual_blocks(model)
    model.train()

    ds_tr = WBTrainDataset(train_df, txt_train, preprocess_used, train=True)
    ds_va = WBInferDataset(val_df, txt_val, preprocess_used, tta=USE_TTA)

    num_workers = min(8, os.cpu_count() or 2)
    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers,
                       pin_memory=(DEVICE=="cuda"), drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=math.ceil(BATCH_SIZE/2), shuffle=False, num_workers=num_workers,
                       pin_memory=(DEVICE=="cuda"), drop_last=False)

    optimizer = make_optimizer(model)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE=="cuda" and MIXED_PRECISION))
    total_steps = EPOCHS * max(1, len(dl_tr))
    warmup_steps = int(WARMUP_RATIO * total_steps)
    step_id = 0
    bce = torch.nn.BCEWithLogitsLoss()

    best_auc = -1.0
    best_path = os.path.join(OUTPUTS_DIR, f"vit_h14_224_last2_seed{seed}_fold{fold_id}_best.pt")
    val_preds_best = None

    for epoch in range(1, EPOCHS+1):
        model.train()
        pbar = tqdm(dl_tr, desc=f"[seed{seed}|fold{fold_id}] epoch {epoch}/{EPOCHS}", leave=False)
        optimizer.zero_grad(set_to_none=True)
        for it, (imgs, txts, labels) in enumerate(pbar, start=1):
            imgs = imgs.to(DEVICE, non_blocking=True)
            txts = txts.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEVICE=="cuda" and MIXED_PRECISION)):
                img_feat = F.normalize(model.encode_image(imgs), dim=-1)
                txt_feat = F.normalize(txts, dim=-1)
                sim = torch.sum(img_feat * txt_feat, dim=1)
                logits = sim * model.logit_scale.exp()
                loss = bce(logits, labels)
            scaler.scale(loss).backward()
            if it % GRAD_ACCUM_STEPS == 0:
                lr = cosine_warmup_lr(step_id, total_steps, INIT_LR, warmup_steps)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                step_id += 1

        # валидация по эпохе
        val_probs = infer_validation(model, dl_va)
        val_auc = roc_auc_score(val_df["label"].values.astype(int), val_probs)
        logger.info(f"[seed{seed}|fold{fold_id}] epoch {epoch}: val AUC = {val_auc:.6f}")
        if val_auc > best_auc:
            best_auc = val_auc
            state = {"model_state": model.state_dict(), "best_auc": float(best_auc), "epoch": epoch, "seed": seed, "fold": fold_id}
            ok = atomic_torch_save(state, best_path)
            if not ok:
                logger.error("Не удалось сохранить чекпойнт (best).")
            val_preds_best = val_probs.copy()

    # safety: если не сохранилось — оставим финальный стейт
    if not os.path.exists(best_path):
        state = {"model_state": model.state_dict(), "best_auc": float(best_auc), "seed": seed, "fold": fold_id}
        atomic_torch_save(state, best_path)

    if val_preds_best is None:
        val_preds_best = val_probs
    return best_path, float(best_auc), val_preds_best

@torch.no_grad()
def infer_validation(model, dl_va: DataLoader) -> np.ndarray:
    model.eval()
    parts = []
    pbar = tqdm(dl_va, desc="eval", leave=False)
    for x, txt in pbar:
        x = x.to(DEVICE, non_blocking=True)     # [B, T, 3, H, W] или [B,1,3,H,W]
        txt = txt.to(DEVICE, non_blocking=True) # [B, D]
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.view(B*T, C, H, W)
        else:
            B1, C, H, W = x.shape
            T = 1
            B = B1
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEVICE=="cuda" and MIXED_PRECISION)):
            img_feat = F.normalize(model.encode_image(x), dim=-1).view(B, T, -1).mean(dim=1)
            txt_feat = F.normalize(txt, dim=-1)
            sim = torch.sum(img_feat * txt_feat, dim=1)
            logits = sim * model.logit_scale.exp()
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            parts.append(probs)
    return np.concatenate(parts, axis=0) if parts else np.zeros(0, dtype=float)

@torch.no_grad()
def infer_test(checkpoints: List[str], test_df: pd.DataFrame, txt_test: np.ndarray) -> np.ndarray:
    preds_all = []
    for ckpt in checkpoints:
        try:
            model, preprocess, _ = load_model_and_preprocess()
            state = torch.load(ckpt, map_location=DEVICE)
            msd = state.get("model_state", state)
            model.load_state_dict(msd, strict=False)
            model.eval()

            ds_te = WBInferDataset(test_df, txt_test, preprocess, tta=USE_TTA)
            dl_te = DataLoader(ds_te, batch_size=math.ceil(BATCH_SIZE/2), shuffle=False,
                               num_workers=min(8, os.cpu_count() or 2), pin_memory=(DEVICE=="cuda"))
            part = []
            pbar = tqdm(dl_te, desc=f"test {os.path.basename(ckpt)}", leave=False)
            for x, txt in pbar:
                x = x.to(DEVICE, non_blocking=True)
                txt = txt.to(DEVICE, non_blocking=True)
                if x.dim() == 5:
                    B, T, C, H, W = x.shape
                    x = x.view(B*T, C, H, W)
                else:
                    B1, C, H, W = x.shape
                    T = 1
                    B = B1
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEVICE=="cuda" and MIXED_PRECISION)):
                    img_feat = F.normalize(model.encode_image(x), dim=-1).view(B, T, -1).mean(dim=1)
                    txt_feat = F.normalize(txt, dim=-1)
                    sim = torch.sum(img_feat * txt_feat, dim=1)
                    logits = sim * model.logit_scale.exp()
                    probs = torch.sigmoid(logits).detach().cpu().numpy()
                    part.append(probs)
            preds_all.append(np.concatenate(part, axis=0))
        except Exception as e:
            logger.error(f"Ошибка инференса с чекпойнтом {ckpt}: {e}")
    return np.mean(np.stack(preds_all, axis=0), axis=0) if preds_all else np.zeros(len(test_df), dtype=float)

# ----------------------------
# Основной скрипт
# ----------------------------
def main():
    t0 = time.time()
    # 0) Читаем данные
    train = read_csv_auto(TRAIN_CSV)
    test  = read_csv_auto(TEST_CSV)
    need_train = {"id", "card_identifier_id", "title", "description", "label"}
    need_test  = {"id", "card_identifier_id", "title", "description"}
    if not need_train.issubset(train.columns):
        logger.error(f"train.csv должен содержать {need_train}. Найдено: {list(train.columns)}"); return
    if not need_test.issubset(test.columns):
        logger.error(f"test.csv должен содержать {need_test}. Найдено: {list(test.columns)}"); return

    for df, nm in [(train, "train"), (test, "test")]:
        df["id"] = df["id"].astype(int)
        df["card_identifier_id"] = df["card_identifier_id"].astype(str)
        df["title"] = df["title"].fillna("")
        df["description"] = df["description"].fillna("")
    logger.info(f"Размеры: train={len(train)}, test={len(test)}")

    # 1) Предвычисление текстовых эмбеддингов с ансамблем шаблонов (делаем один раз для всех сидов)
    model_txt, preprocess, tokenizer = load_model_and_preprocess()
    model_txt.eval()

    tr_texts_flat, T = build_texts_for_df_multi(train, POS_TEMPLATES)
    te_texts_flat, _ = build_texts_for_df_multi(test,  POS_TEMPLATES)

    tr_txt_flat = encode_texts(model_txt, tokenizer, tr_texts_flat, batch=256)  # [N*T, D]
    te_txt_flat = encode_texts(model_txt, tokenizer, te_texts_flat, batch=256)  # [M*T, D]
    del model_txt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    tr_txt = tr_txt_flat.reshape(len(train), T, -1).mean(axis=1)  # [N, D]
    te_txt = te_txt_flat.reshape(len(test),  T, -1).mean(axis=1)  # [M, D]

    # 2) Сплиты
    y = train["label"].astype(int).values
    groups = train["card_identifier_id"].values
    gkf = GroupKFold(n_splits=N_FOLDS)

    # 3) Бэггинг по сид-значениям
    seed_reports: Dict[int, dict] = {}
    oof_bag = np.zeros(len(train), dtype=float)
    ckpts_all: List[str] = []

    for seed in SEEDS:
        set_seeds(seed)
        oof_pred_seed = np.zeros(len(train), dtype=float)
        fold_aucs = []
        ckpts_seed: List[str] = []

        for fold_id, (tr_idx, va_idx) in enumerate(gkf.split(train, y, groups=groups), start=1):
            tr_df = train.iloc[tr_idx].copy().reset_index(drop=True)
            va_df = train.iloc[va_idx].copy().reset_index(drop=True)
            try:
                best_path, best_auc, val_preds = train_one_fold(
                    fold_id, tr_df, va_df, tr_txt[tr_idx], tr_txt[va_idx], preprocess, seed
                )
                fold_aucs.append(best_auc)
                oof_pred_seed[va_idx] = val_preds
                if os.path.exists(best_path):
                    ckpts_seed.append(best_path)
                else:
                    logger.error(f"Чекпойнт не найден для seed{seed}|fold{fold_id}: {best_path}")
            except Exception as e:
                logger.error(f"Ошибка на seed{seed}|fold{fold_id}: {e}\n{traceback.format_exc()}")

        # метрика по сид-эксперименту
        try:
            oof_auc_seed = roc_auc_score(y, oof_pred_seed)
        except Exception:
            oof_auc_seed = float("nan")

        seed_reports[seed] = {
            "fold_aucs": [float(x) for x in fold_aucs],
            "oof_auc": float(oof_auc_seed),
            "checkpoints": ckpts_seed
        }
        oof_bag += oof_pred_seed / max(1, len(SEEDS))
        ckpts_all.extend(ckpts_seed)

    # 4) Общая OOF метрика после бэггинга
    try:
        oof_auc = roc_auc_score(y, oof_bag)
    except Exception:
        oof_auc = float("nan")
    logger.info("OOF AUC (bagging) = {:.6f}".format(oof_auc))
    for s, rep in seed_reports.items():
        logger.info(f"  seed {s}: OOF={rep['oof_auc']:.6f} | folds=" + ", ".join([f"{x:.4f}" for x in rep["fold_aucs"]]))

    # 5) Инференс на тесте (среднее по всем чекпойнтам всех сидов)
    test_pred = infer_test(ckpts_all, test, te_txt)
    sub = pd.DataFrame({"id": test["id"].astype(int), "y_pred": test_pred.astype(float)})
    sub.to_csv(SUBMISSION_PATH, index=False)
    logger.info(f"Сабмит сохранён: {SUBMISSION_PATH}")

    # 6) Отчёт
    report = {
        "model": f"{MODEL_NAME}@224 last2 (BCE, TTA={USE_TTA})",
        "seeds": SEEDS,
        "seed_reports": seed_reports,
        "oof_auc_bagging": float(oof_auc),
        "n_checkpoints": len(ckpts_all),
        "submission": os.path.basename(SUBMISSION_PATH)
    }
    with open(OOF_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"OOF-репорт сохранён: {OOF_JSON_PATH}")
    logger.info(f"Готово. Время: {(time.time()-t0)/60.1:.1f} мин.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Фатальная ошибка: {e}\n{traceback.format_exc()}")
