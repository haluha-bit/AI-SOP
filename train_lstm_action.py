"""
train_lstm_action.py
===================
LSTM 动作识别模型训练脚本

功能：从视频和标注数据中提取时序特征，训练 LSTM 动作分类模型

使用方法：
    python train_lstm_action.py

输出：
    lstm_runs_fine/
    ├── best_lstm_fine.pt    # 最佳模型权重
    └── config.json          # 模型配置文件
"""

import json
import csv
import shutil
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import deque
from pathlib import Path
from datetime import datetime

try:
    from ultralytics import YOLO
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    from tqdm import tqdm
except ImportError as e:
    raise ImportError("请安装依赖: pip install ultralytics mediapipe torch torchvision tqdm") from e


class Config:
    BASE_DIR = Path(__file__).parent.resolve()
    VIDEO_DIR = BASE_DIR / "video"
    TIMELINE_CSV = BASE_DIR / "full_timeline_10videos_template.csv"
    YOLO_MODEL_PATH = BASE_DIR / "runs_detect/yolov8s_mirror_v14_no_earlystop/weights/best.pt"
    OUTPUT_DIR = BASE_DIR / "lstm_runs_fine"

    INPUT_SIZE = 146
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    NUM_CLASSES = 6
    DROPOUT = 0.2

    WINDOW_SIZE = 48
    STRIDE = 16
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 100
    PATIENCE = 15

    VAL_RATIO = 0.2
    RANDOM_SEED = 42

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


CLASS_NAMES = ["base", "frame", "mirror", "screw"]
SOURCE_WINDOWS = [48, 72, 96]

LABEL_MAP = {
    "background": 0,
    "S1_base_frame_joint": 1,
    "S2_frame_mirror_fix": 2,
    "S3_right_screw": 3,
    "S4_left_screw": 4,
    "S5_final_check": 5
}

LABEL_MAP_INVERSE = {v: k for k, v in LABEL_MAP.items()}

STEP_ID_TO_LABEL = {
    "S1": "S1_base_frame_joint",
    "S2": "S2_frame_mirror_fix",
    "S3": "S3_right_screw",
    "S4": "S4_left_screw",
    "S5": "S5_final_check",
}


class ActionDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class ActionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def build_feature_row(detections, hands, frame_w, frame_h):
    det_feat = {k: [0.0, 0.0, 0.0, 0.0, 0.0] for k in CLASS_NAMES}

    for d in detections:
        cls_name = d.get("cls_name")
        if cls_name not in det_feat:
            continue
        conf = float(d.get("conf", 0.0))
        x1, y1, x2, y2 = d.get("xyxy", [0, 0, 0, 0])
        w = max(0.0, float(x2) - float(x1))
        h = max(0.0, float(y2) - float(y1))
        cx = float(x1) + w / 2.0
        cy = float(y1) + h / 2.0

        fw = max(float(frame_w), 1.0)
        fh = max(float(frame_h), 1.0)
        if conf > det_feat[cls_name][0]:
            det_feat[cls_name] = [conf, cx / fw, cy / fh, w / fw, h / fh]

    det_vec = []
    for cname in CLASS_NAMES:
        det_vec.extend(det_feat[cname])

    hand_vec = [0.0] * (2 * 21 * 3)
    for hi, hand in enumerate(hands[:2]):
        lms = hand.get("landmarks", [])
        for li, lm in enumerate(lms[:21]):
            base = hi * 63 + li * 3
            hand_vec[base] = float(lm[0]) if len(lm) > 0 else 0.0
            hand_vec[base + 1] = float(lm[1]) if len(lm) > 1 else 0.0
            hand_vec[base + 2] = float(lm[2]) if len(lm) > 2 else 0.0

    return np.array(det_vec + hand_vec, dtype=np.float32)


def step_id_to_fine_label(step_id):
    return STEP_ID_TO_LABEL.get(str(step_id), "background")


def extract_features_from_video(video_path, timeline_df, yolo_model, hands_model, window_size, stride):
    sequences = []
    labels = []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  警告: 无法打开视频 {video_path.name}")
        return sequences, labels

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"  处理: {video_path.name}, FPS={fps:.1f}, 帧数={total_frames}")

    frame_id = 0
    feat_queue = deque(maxlen=max(SOURCE_WINDOWS))
    pbar = tqdm(total=total_frames, desc=f"  提取特征", unit="帧", leave=False)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_h, frame_w = frame.shape[:2]
        t_sec = frame_id / fps

        pbar.update(1)

        res = yolo_model.predict(frame, verbose=False, conf=0.25, device="cpu")[0]
        detections = []
        if res.boxes is not None:
            for box in res.boxes:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                detections.append({
                    "cls_name": yolo_model.names.get(cls_id, str(cls_id)),
                    "conf": float(box.conf[0]),
                    "xyxy": [x1, y1, x2, y2],
                })

        rgb_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hr = hands_model.process(rgb_raw)
        hands_out = []
        if hr.multi_hand_landmarks:
            for hand_lm in hr.multi_hand_landmarks:
                lms = [[float(lm.x), float(lm.y), float(lm.z)] for lm in hand_lm.landmark]
                hands_out.append({"landmarks": lms})

        feat = build_feature_row(detections, hands_out, frame_w, frame_h)
        feat_queue.append(feat)

        current_label = "background"
        for _, row in timeline_df.iterrows():
            if row["start_sec"] <= t_sec <= row["end_sec"]:
                current_label = step_id_to_fine_label(row["step_id"])
                break

        if len(feat_queue) >= window_size:
            for i in range(0, len(feat_queue) - window_size + 1, stride):
                window_feats = list(feat_queue)[i:i + window_size]
                seq = np.stack(window_feats, axis=0)
                sequences.append(seq)
                labels.append(LABEL_MAP[current_label])

        frame_id += 1

    pbar.close()
    cap.release()
    print(f"  完成: 提取 {len(sequences)} 个训练样本")
    return sequences, labels


def calculate_class_weights(labels):
    class_counts = [0] * len(LABEL_MAP)
    for label_idx in labels:
        class_counts[label_idx] += 1

    total = len(labels)
    weights = []
    for count in class_counts:
        if count > 0:
            weights.append(total / (len(class_counts) * count))
        else:
            weights.append(1.0)

    max_weight = max(weights)
    weights = [w / max_weight for w in weights]

    return torch.FloatTensor(weights)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def train_lstm():
    print("=" * 60)
    print("LSTM 动作识别模型训练")
    print("=" * 60)

    cfg = Config()

    print(f"\n配置:")
    print(f"  设备: {cfg.DEVICE}")
    print(f"  视频目录: {cfg.VIDEO_DIR}")
    print(f"  标注文件: {cfg.TIMELINE_CSV}")
    print(f"  输出目录: {cfg.OUTPUT_DIR}")
    print(f"  窗口大小: {cfg.WINDOW_SIZE}")
    print(f"  步长: {cfg.STRIDE}")
    print(f"  批量大小: {cfg.BATCH_SIZE}")
    print(f"  学习率: {cfg.LEARNING_RATE}")
    print(f"  训练轮数: {cfg.NUM_EPOCHS}")

    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not cfg.TIMELINE_CSV.exists():
        print(f"\n错误: 标注文件不存在: {cfg.TIMELINE_CSV}")
        print("请创建 timeline CSV 文件或检查路径")
        return

    if not cfg.YOLO_MODEL_PATH.exists():
        print(f"\n警告: YOLO模型不存在: {cfg.YOLO_MODEL_PATH}")
        print("将使用随机初始化的YOLO模型")

    print("\n" + "-" * 40)
    print("Step 1: 加载模型")
    print("-" * 40)

    if cfg.YOLO_MODEL_PATH.exists():
        print(f"加载 YOLO 模型: {cfg.YOLO_MODEL_PATH}")
        yolo = YOLO(str(cfg.YOLO_MODEL_PATH))
    else:
        print("创建 YOLO 模型 (无预训练权重)")
        yolo = YOLO("yolov8s.pt")

    print("初始化 MediaPipe Hands...")
    hands_model = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    print("\n" + "-" * 40)
    print("Step 2: 读取标注数据")
    print("-" * 40)

    timeline_df = pd.read_csv(cfg.TIMELINE_CSV)
    print(f"标注文件包含 {len(timeline_df)} 条记录")

    available_videos = list(cfg.VIDEO_DIR.glob("*.mp4"))
    print(f"视频目录包含 {len(available_videos)} 个视频文件")

    if not available_videos:
        print("\n错误: 视频目录为空")
        return

    print("\n" + "-" * 40)
    print("Step 3: 提取训练特征")
    print("-" * 40)

    all_sequences = []
    all_labels = []

    for video_path in tqdm(available_videos, desc="处理视频", unit="个"):
        video_annotations = timeline_df[timeline_df["video_name"] == video_path.name]

        if video_annotations.empty:
            print(f"\n跳过 (无标注): {video_path.name}")
            continue

        video_annotations = video_annotations.sort_values(["start_sec", "end_sec"])

        seqs, lbls = extract_features_from_video(
            video_path,
            video_annotations,
            yolo,
            hands_model,
            cfg.WINDOW_SIZE,
            cfg.STRIDE
        )

        if seqs:
            all_sequences.extend(seqs)
            all_labels.extend(lbls)

    hands_model.close()

    if not all_sequences:
        print("\n错误: 未能提取任何训练样本")
        print("请检查: 1) 视频文件是否有效 2) 标注是否正确")
        return

    all_sequences = np.array(all_sequences)
    all_labels = np.array(all_labels)

    print(f"\n特征提取完成:")
    print(f"  总样本数: {len(all_sequences)}")
    print(f"  特征形状: {all_sequences.shape}")
    print(f"  类别分布:")

    for label_idx, label_name in LABEL_MAP_INVERSE.items():
        count = (all_labels == label_idx).sum()
        percentage = count / len(all_labels) * 100
        print(f"    {label_name}: {count} ({percentage:.1f}%)")

    print("\n" + "-" * 40)
    print("Step 4: 划分数据集")
    print("-" * 40)

    np.random.seed(cfg.RANDOM_SEED)
    indices = np.arange(len(all_sequences))
    np.random.shuffle(indices)

    split_idx = int(len(indices) * (1 - cfg.VAL_RATIO))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_seq = all_sequences[train_indices]
    train_lbl = all_labels[train_indices]
    val_seq = all_sequences[val_indices]
    val_lbl = all_labels[val_indices]

    print(f"  训练集: {len(train_seq)} 样本")
    print(f"  验证集: {len(val_seq)} 样本")

    train_dataset = ActionDataset(train_seq, train_lbl)
    val_dataset = ActionDataset(val_seq, val_lbl)

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)

    print("\n" + "-" * 40)
    print("Step 5: 初始化模型")
    print("-" * 40)

    model = ActionLSTM(
        cfg.INPUT_SIZE,
        cfg.HIDDEN_SIZE,
        cfg.NUM_LAYERS,
        cfg.NUM_CLASSES,
        cfg.DROPOUT
    ).to(cfg.DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  模型参数量: {total_params:,} (可训练: {trainable_params:,})")

    class_weights = calculate_class_weights(all_labels).to(cfg.DEVICE)
    print(f"  类别权重: {class_weights.cpu().numpy()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    print("\n" + "-" * 40)
    print("Step 6: 开始训练")
    print("-" * 40)

    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0

    for epoch in tqdm(range(cfg.NUM_EPOCHS), desc="训练", unit="轮"):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, cfg.DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, cfg.DEVICE)

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch [{epoch+1:3d}/{cfg.NUM_EPOCHS}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"LR: {current_lr:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0

            torch.save(model.state_dict(), cfg.OUTPUT_DIR / "best_lstm_fine.pt")
            print(f"  -> 保存最佳模型 (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= cfg.PATIENCE:
                print(f"\n早停: 验证损失连续 {cfg.PATIENCE} 轮未下降")
                break

    print("\n" + "-" * 40)
    print("Step 7: 保存配置文件")
    print("-" * 40)

    config_data = {
        "input_size": cfg.INPUT_SIZE,
        "hidden_size": cfg.HIDDEN_SIZE,
        "num_layers": cfg.NUM_LAYERS,
        "num_classes": cfg.NUM_CLASSES,
        "label_map": [
            {"id": v, "label": k} for k, v in LABEL_MAP.items()
        ],
        "training_info": {
            "best_epoch": best_epoch,
            "best_val_loss": float(best_val_loss),
            "train_samples": len(train_seq),
            "val_samples": len(val_seq),
            "window_size": cfg.WINDOW_SIZE,
            "stride": cfg.STRIDE,
        }
    }

    config_path = cfg.OUTPUT_DIR / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, ensure_ascii=False, indent=2)

    print(f"配置文件已保存: {config_path}")
    print(f"模型权重已保存: {cfg.OUTPUT_DIR / 'best_lstm_fine.pt'}")

    print("\n" + "=" * 60)
    print("训练完成!")
    print(f"最佳验证损失: {best_val_loss:.4f} (Epoch {best_epoch})")
    print("=" * 60)


if __name__ == "__main__":
    train_lstm()
