"""Training history logging + plotting + HF sync.

Usage:
    from plot import PlotManager
    pm = PlotManager(hf, save_dir)
    pm.log(step, loss, lr, tps)
    pm.plot(step)    # saves plot_train_step_NNNN.png
    pm.upload(step)  # pushes plot + history JSON to HF
"""

import os, json, time
from pathlib import Path


class PlotManager:
    """Logs training metrics to JSON, generates loss plots, syncs to HuggingFace.

    JSON is updated locally after each log. Plot is generated + uploaded
    periodically (every plot_interval steps).
    """

    def __init__(self, hf, save_dir=".", history_file="train_history.json",
                 plot_interval=256):
        self.hf = hf
        self.save_dir = Path(save_dir)
        self.history_file = self.save_dir / history_file
        self.plot_interval = plot_interval
        self.last_uploaded_step = 0
        self.history = self._load_or_download()

    def _load_or_download(self):
        """Load local JSON, or fall back to HF download, or start fresh."""
        if self.history_file.exists():
            with open(self.history_file) as f:
                return json.load(f)
        if self.hf:
            try:
                from huggingface_hub import hf_hub_download
                path = hf_hub_download(
                    repo_id=self.hf.repo_id,
                    filename=self.history_file.name,
                    revision=self.hf.revision,
                    token=self.hf._get_token(),
                )
                import shutil
                shutil.copy2(path, self.history_file)
                print(f"Downloaded {self.history_file.name} from HF")
                with open(self.history_file) as f:
                    return json.load(f)
            except Exception as e:
                print(f"No remote history ({e}), starting fresh")
        return []

    def log(self, step, loss, lr=None, tps=None, aux_loss=None):
        """Append one row to history and save JSON."""
        entry = {"step": step, "loss": loss, "time": time.time()}
        if lr is not None:
            entry["lr"] = lr
        if tps is not None:
            entry["tps"] = tps
        if aux_loss is not None:
            entry["aux_loss"] = aux_loss
        self.history.append(entry)
        with open(self.history_file, "w") as f:
            json.dump(self.history, f, indent=2)

    def plot(self, step):
        """Generate loss plot as plot_train_step_{step}.png. No-op if no matplotlib."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return
        if len(self.history) < 2:
            return
        steps = [e["step"] for e in self.history]
        losses = [e["loss"] for e in self.history]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(steps, losses, label="loss")
        if "aux_loss" in self.history[0]:
            aux = [e.get("aux_loss", 0) for e in self.history]
            ax.plot(steps, aux, label="aux_loss", alpha=0.5)
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.set_title(f"Training loss (step {step})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        path = self.save_dir / f"plot_train_step_{step}.png"
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)

    def upload(self, step):
        """Upload JSON (overwrite) + plot (step-prefixed) to HF."""
        if not self.hf:
            return
        api = self.hf._get_api()
        self.hf.ensure_repo()
        self.hf.ensure_revision()
        msg = f"Training step {step}"

        # Upload history JSON (overwrites)
        if self.history_file.exists():
            api.upload_file(
                path_or_fileobj=str(self.history_file),
                path_in_repo=self.history_file.name,
                repo_id=self.hf.repo_id,
                revision=self.hf.revision,
                token=self.hf._get_token(),
                commit_message=msg,
            )

        # Upload step plot
        plot_path = self.save_dir / f"plot_train_step_{step}.png"
        if plot_path.exists():
            api.upload_file(
                path_or_fileobj=str(plot_path),
                path_in_repo=plot_path.name,
                repo_id=self.hf.repo_id,
                revision=self.hf.revision,
                token=self.hf._get_token(),
                commit_message=msg,
            )
            self.last_uploaded_step = step
