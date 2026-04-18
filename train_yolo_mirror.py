from pathlib import Path

from ultralytics import YOLO


def main():
    project_root = Path(__file__).resolve().parent
    data_yaml = project_root / "yolo" / "data.yaml"

    model = YOLO("yolov8s.pt")
    model.train(
        data=str(data_yaml),
        epochs=80,
        imgsz=960,
        batch=8,
        workers=4,
        device=0,
        project=str(project_root / "runs_detect"),
        name="yolov8s_mirror_v14_no_earlystop",
        patience=0,  # 关闭EarlyStopping，完整跑完epochs
        pretrained=True,
        optimizer="auto",
        amp=True,
        cache=False,
        verbose=True,
    )


if __name__ == "__main__":
    main()
