"""Shared, explicitly reviewed dataset download and partition selection.

Unknown licenses are excluded from open-only mode. See DATA_LICENSES.md.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Dataset:
    id: str
    files: tuple  # (URL, local filename) pairs
    license: str
    license_source: str
    attribution: str
    open_use: bool = False
    reviewed_on: str = "2026-09-05"


DATASETS = (
    Dataset('mango', (('https://data.mendeley.com/public-files/datasets/46htwnp833/files/747b2613-4d0a-4628-8aaf-8fc5547d286e/file_downloaded', 'mango_data.csv'),),
            'CC-BY-4.0', 'https://data.mendeley.com/datasets/46htwnp833/6',
            'Credit Anderson, Walsh and Subedi and the dataset DOI.', open_use=True),
    Dataset('melamine', (('https://raw.githubusercontent.com/RNL1/Melamine-Dataset/master/Melamine_Dataset.pkl', 'melamine_data.pkl'),),
            'MIT', 'https://github.com/RNL1/Melamine-Dataset',
            'Retain the repository copyright and MIT notice; cite the two papers in its README.', open_use=True),
    Dataset('cgl', (('https://eigenvector.com/wp-content/uploads/2021/04/CGL_nir.mat_.zip', 'CGL_nir.mat_.zip'),),
            'Unspecified', 'https://eigenvector.com/resources/data-sets/',
            'No explicit open reuse license identified on the dataset landing page.', open_use=False),
    Dataset('corn', (('https://eigenvector.com/wp-content/uploads/2019/06/corn.mat_.zip', 'corn.mat_.zip'),),
            'Unspecified', 'https://eigenvector.com/resources/data-sets/',
            'No explicit open reuse license identified on the dataset landing page.', open_use=False),
    Dataset('diesel', (('https://eigenvector.com/wp-content/uploads/2019/06/SWRI_Diesel_NIR_CSV.zip', 'SWRI_Diesel_NIR_CSV.zip'),),
            'Unspecified', 'https://eigenvector.com/resources/data-sets/',
            'No explicit open reuse license identified on the dataset landing page.', open_use=False),
    Dataset('shootout', (('https://eigenvector.com/wp-content/uploads/2019/06/nir_shootout_2002.mat_.zip', 'nir_shootout_2002.mat_.zip'),),
            'Unspecified', 'https://eigenvector.com/resources/data-sets/',
            'No explicit open reuse license identified on the dataset landing page.', open_use=False),
    Dataset('eggs', (('https://data.mendeley.com/public-files/datasets/6hn67h2trb/files/0604423d-785c-4076-badb-b3fab8ec8367/file_downloaded', 'eggs.csv'),),
            'CC-BY-4.0', 'https://data.mendeley.com/datasets/6hn67h2trb/1',
            'Credit Ivan Ramirez-Morales and DOI 10.17632/6hn67h2trb.1.', open_use=True),
    Dataset('wheat', (('https://ndownloader.figshare.com/files/6932732', 'wheat_kernel.xlsx'),),
            'CC-BY-4.0', 'https://api.figshare.com/v2/articles/4252217',
            'Credit Wenya Liu and DOI 10.6084/m9.figshare.4252217.v1.', open_use=True),
    Dataset('ossl', (('https://storage.googleapis.com/soilspec4gg-public/ossl_all_L0_v1.2.csv.gz', 'ossl_all_L0_v1.2.csv.gz'), ('https://storage.googleapis.com/soilspec4gg-public/ossl_all_L1_v1.2.csv.gz', 'ossl_all_L1_v1.2.csv.gz')),
            'MIT / CC-BY (publisher declaration)', 'https://soilspectroscopy.org/introducing-the-open-soil-spectral-library/',
            'Credit OSSL and contributing libraries; retain source attribution and license metadata. See DATA_LICENSES.md for scope.', open_use=True),
)

# Alternative public endpoints for the same files in the publisher's storage.
DOWNLOAD_FALLBACKS = {
    next(d for d in DATASETS if d.id == 'mango').files[0][0]: (
        'https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/0f6f0a8b-a671-446d-85ef-bb8fcc34bc5d',
    ),
    next(d for d in DATASETS if d.id == 'eggs').files[0][0]: (
        'https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/ec03af06-005d-40c6-a207-681cc4aabdf4',
    ),
}

PARTITION_SOURCES = {
    "diesel": "diesel", "corn": "corn", "melamine": "melamine",
    "eggs": "eggs", "soil_nir": "ossl", "soil_mir": "ossl",
    "mango": "mango", "cgl": "cgl", "shootout": "shootout", "wheat": "wheat",
}
TRIP_PREFIXES = {
    "diesel": "Diesel_", "corn": "Corn_", "melamine": "Melamine_",
    "eggs": "Eggs", "cgl": "CGL_", "shootout": "Shootout_", "wheat": "Wheat",
}


def select_datasets(open_only=False):
    return tuple(d for d in DATASETS if not open_only or d.open_use)


def select_partitions(dataset="all", open_only=False):
    if dataset != "all" and dataset not in PARTITION_SOURCES:
        raise ValueError(f"Unknown dataset: {dataset}")
    allowed = {d.id for d in select_datasets(open_only)}
    if dataset != "all" and PARTITION_SOURCES[dataset] not in allowed:
        raise ValueError(f"{dataset} is excluded by --open-only: no verified open reuse license")
    return tuple(name for name, source in PARTITION_SOURCES.items()
                 if source in allowed and dataset in ("all", name))
