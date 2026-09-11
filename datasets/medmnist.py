"""Optional MedMNIST dataset integration for cross-domain PEFT experiments."""
from __future__ import annotations

from torch.utils.data import Dataset


class _TargetAdapter(Dataset):
    def __init__(self, base, transform=None):
        self.base = base
        self.transform = transform
    def __len__(self):
        return len(self.base)
    def __getitem__(self, index):
        image, target = self.base[index]
        if self.transform is not None:
            image = self.transform(image)
        try:
            target = int(target[0])
        except Exception:
            target = int(target)
        return image, target


def build_medmnist(name: str, root: str, split: str, *, download: bool, transform=None):
    try:
        import medmnist
    except Exception as exc:
        raise RuntimeError(
            "MedMNIST support is optional. Install it with `pip install medmnist` "
            "or `pip install -r requirements-optional.txt`."
        ) from exc

    key = str(name).lower().replace("-", "_")
    class_names = {
        "pathmnist": "PathMNIST",
        "dermamnist": "DermaMNIST",
        "bloodmnist": "BloodMNIST",
        "pneumoniamnist": "PneumoniaMNIST",
        "organamnist": "OrganAMNIST",
        "organamnist_axial": "OrganAMNIST",
        "tissuemnist": "TissueMNIST",
    }
    if key not in class_names:
        raise KeyError(key)
    cls = getattr(medmnist, class_names[key])
    med_split = {"train": "train", "val": "val", "test": "test"}[split]
    # size=224 is supported in modern MedMNIST and is preferable for ImageNet
    # backbones. Fall back gracefully for older package versions.
    try:
        base = cls(split=med_split, root=root, download=download, size=224)
    except TypeError:
        base = cls(split=med_split, root=root, download=download)
    info = medmnist.INFO.get(key if key != "organamnist" else "organamnist_axial", {})
    labels = info.get("label", {})
    num_classes = len(labels) if labels else 0
    if num_classes <= 0:
        # MedMNIST exposes n_classes on newer versions.
        num_classes = int(getattr(base, "n_classes", 2))
    return _TargetAdapter(base, transform=transform), num_classes
