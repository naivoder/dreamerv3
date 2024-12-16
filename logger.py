from tensorboardX import SummaryWriter
import os

class TrainingLogger:
    def __init__(self, log_dir, task):
        save_dir = os.path.join(log_dir, "_".join(task.split(".")))
        os.makedirs(save_dir, exist_ok=True)
        self._writer = SummaryWriter(save_dir)

    def log_evaluation_summary(self, summary, step):
        for k, v in summary.items():
            self._writer.add_scalar(k, float(v), step)
        self._writer.flush()

    def log_metrics(self, summary, step):
        print("\n----Training step {} summary----".format(step))
        for k, v in summary.items():
            val = float(v)
            print("{:<40} {:<.2f}".format(k, val))
            self._writer.add_scalar(k, val, step)
        self._writer.flush()

    # (N, T, C, H, W)
    def log_video(self, images, step=None, name="policy", fps=30):
        self._writer.add_video(name, images, step, fps=fps)
        self._writer.flush()

    def log_images(self, images, step=None, name="policy"):
        self._writer.add_images(name, images, step, dataformats="NHWC")
        self._writer.flush()

    def log_figure(self, figure, step=None, name="policy"):
        self._writer.add_figure(name, figure, step)
        self._writer.flush()
