"""Single source of truth for dataset capabilities used by the research code."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    aliases: tuple[str, ...] = ()
    task: str = "single_label"
    num_classes: int | None = None
    source: str = "torchvision"
    download_policy: str = "safe_auto"
    split_policy: str = "official_or_deterministic"
    default_metric: str = "accuracy"
    supports_kaggle: bool = True
    requires_auth: bool = False
    notes: str = ""

    def matches(self, value: str) -> bool:
        key = normalize_dataset_name(value)
        return key == self.name or key in self.aliases

    def to_dict(self) -> dict:
        return asdict(self)


def normalize_dataset_name(value: str) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


# Keep this registry descriptive: builder implementation remains in build.py so
# the submitted-paper experiment path is not rewritten.
_SPECS = [
    DatasetSpec("fake", ("fakedata", "synthetic"), source="synthetic", download_policy="manual"),
    DatasetSpec("csv", ("csv_dataset",), source="local", download_policy="manual"),
    DatasetSpec("imagefolder", ("folder", "imagenet", "imagenet1k", "imagenet_a", "imagenet_r", "imagenet_sketch", "tiny_imagenet", "tinyimagenet"), source="local", download_policy="manual"),
    DatasetSpec("cub200", ("cub_200", "cub_200_2011", "cub"), source="local", download_policy="manual"),
    DatasetSpec("nabirds", ("na_birds",), source="local", download_policy="manual"),
    DatasetSpec("stanford_dogs", ("stanforddogs", "dogs"), source="local", download_policy="manual"),
    DatasetSpec("vtab", ("vtab1k", "vtab_1k"), source="local", download_policy="manual"),
    DatasetSpec("fewshot", ("few_shot", "metadata_fewshot"), source="local", download_policy="manual"),
    DatasetSpec("cifar10", num_classes=10), DatasetSpec("cifar100", ("cifar_100",), num_classes=100),
    DatasetSpec("mnist", num_classes=10), DatasetSpec("fashion_mnist", ("fashionmnist",), num_classes=10),
    DatasetSpec("emnist", ("emnist_byclass",), num_classes=62), DatasetSpec("kmnist", num_classes=10),
    DatasetSpec("qmnist", num_classes=10), DatasetSpec("usps", num_classes=10), DatasetSpec("svhn", num_classes=10),
    DatasetSpec("stl10", num_classes=10), DatasetSpec("food101", ("food_101",), num_classes=101),
    DatasetSpec("oxfordiiitpet", ("oxford_iiit_pet", "pets", "oxford_pets"), num_classes=37),
    DatasetSpec("flowers102", ("oxfordflowers102", "oxford_flowers102"), num_classes=102),
    DatasetSpec("stanford_cars", ("stanfordcars", "cars"), num_classes=196),
    DatasetSpec("caltech101", ("caltech_101",), num_classes=101),
    DatasetSpec("dtd", ("describable_textures", "textures"), num_classes=47),
    DatasetSpec("eurosat", ("euro_sat",), num_classes=10),
    DatasetSpec("fgvc_aircraft", ("fgvca", "aircraft"), num_classes=100),
    DatasetSpec("sun397", ("sun_397",), num_classes=397), DatasetSpec("gtsrb", ("traffic_signs", "german_traffic_signs"), num_classes=43),
    DatasetSpec("fer2013", ("fer_2013",), num_classes=7), DatasetSpec("pcam", ("patch_camelyon",), num_classes=2),
    DatasetSpec("country211", ("country_211",), num_classes=211), DatasetSpec("rendered_sst2", ("renderedsst2",), num_classes=2),
    DatasetSpec("places365", ("places_365",), num_classes=365, download_policy="large_auto"),
    DatasetSpec("inaturalist", ("inat", "inat2021"), download_policy="large_auto"),
    DatasetSpec("coco", ("coco2017", "mscoco", "mscoco2017"), task="multilabel", num_classes=80, download_policy="large_auto"),
    DatasetSpec("voc2007", ("pascal_voc", "pascal_voc2007"), task="multilabel", num_classes=20),
    DatasetSpec("celeba", ("celeba_attributes",), task="multilabel", num_classes=40),
    DatasetSpec("voc2012_segmentation", ("voc_segmentation", "pascal_voc_segmentation"), task="semantic_segmentation", num_classes=21),
    DatasetSpec("oxford_pet_segmentation", ("pet_segmentation", "oxfordiiitpet_segmentation"), task="semantic_segmentation", num_classes=3),
    DatasetSpec("sbd_segmentation", ("berkeley_sbd",), task="semantic_segmentation", num_classes=21),
    DatasetSpec("cityscapes_segmentation", ("cityscapes",), task="semantic_segmentation", num_classes=19, download_policy="manual", requires_auth=True),
    DatasetSpec("segmentation_folder", ("mask_folder",), task="semantic_segmentation", source="local", download_policy="manual"),
    DatasetSpec("depth_folder", ("monocular_depth_folder",), task="depth_estimation", source="local", download_policy="manual", default_metric="rmse"),
    DatasetSpec("fake_segmentation", task="semantic_segmentation", source="synthetic", download_policy="manual"),
    DatasetSpec("fake_depth", task="depth_estimation", source="synthetic", download_policy="manual"),
    DatasetSpec("voc_detection", ("voc2007_detection", "voc2012_detection"), task="object_detection", num_classes=20),
    DatasetSpec("coco_detection", ("coco2017_detection",), task="object_detection", num_classes=80, download_policy="large_auto"),
    DatasetSpec("fake_detection", task="object_detection", source="synthetic", download_policy="manual"),
    # New optional open medical-image routes. The integration uses MedMNIST's
    # 224x224 flag where supported so the experiment is not based on 28x28 upsampling.
    DatasetSpec("pathmnist", task="single_label", source="medmnist", download_policy="safe_auto", default_metric="accuracy"),
    DatasetSpec("dermamnist", task="single_label", source="medmnist", download_policy="safe_auto", default_metric="accuracy"),
    DatasetSpec("bloodmnist", task="single_label", source="medmnist", download_policy="safe_auto", default_metric="accuracy"),
    DatasetSpec("pneumoniamnist", task="single_label", source="medmnist", download_policy="safe_auto", default_metric="accuracy"),
    DatasetSpec("organamnist", ("organamnist_axial",), task="single_label", source="medmnist", download_policy="safe_auto", default_metric="accuracy"),
    DatasetSpec("tissuemnist", task="single_label", source="medmnist", download_policy="safe_auto", default_metric="accuracy"),
]

DATASET_SPECS = {spec.name: spec for spec in _SPECS}


def dataset_specs() -> tuple[DatasetSpec, ...]:
    return tuple(_SPECS)


def get_dataset_spec(name: str) -> DatasetSpec:
    key = normalize_dataset_name(name)
    for spec in _SPECS:
        if spec.matches(key):
            return spec
    raise KeyError(f"Unknown dataset {name!r}")


def available_dataset_names() -> list[str]:
    return sorted(DATASET_SPECS)


def registry_as_dicts() -> list[dict]:
    return [spec.to_dict() for spec in _SPECS]
