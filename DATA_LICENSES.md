# Dataset licenses and the open-only subset

Review date: 2026-09-05. The executable selection is in `dataset_catalog.py`.
`--open-only` selects sources with an explicit open license allowing commercial
reuse, subject to its conditions. It does not mean “unlicensed” or “no obligations”.
Unknown or unverified permissions are excluded. The repository's MIT license
covers its code; it does not replace third-party dataset terms.

| Dataset | Included | License evidence and attribution |
|---|---|---|
| Mango | Yes | [Mendeley version 6](https://data.mendeley.com/datasets/46htwnp833/6) declares CC BY 4.0. Credit Nicholas Anderson, Kerry Walsh and Phul Subedi, cite the dataset and accompanying publication. The existing file URL is retained. |
| Melamine | Yes | The [dataset repository](https://github.com/RNL1/Melamine-Dataset) contains the downloaded PKL alongside an MIT LICENSE and states that publication has the owner's explicit consent. We apply that repository license to this bundled dataset, with no separate exception identified. Retain its copyright and license notice and cite the two publications requested in its README. |
| Eggs | Yes | [Mendeley version 1](https://data.mendeley.com/datasets/6hn67h2trb/1), CC BY 4.0. Credit Ivan Ramírez-Morales; DOI 10.17632/6hn67h2trb.1. |
| Wheat Kernel | Yes | [Figshare API record](https://api.figshare.com/v2/articles/4252217) explicitly identifies file 6932732 and CC BY 4.0. Credit Wenya Liu; DOI 10.6084/m9.figshare.4252217.v1. |
| OSSL | Yes | The [publisher's data licensing declaration](https://soilspectroscopy.org/introducing-the-open-soil-spectral-library/) describes compiled data under MIT and the Zenodo backup under CC BY. Credit OSSL and its contributing libraries and preserve their source/license metadata. See scope below. |
| Diesel, Corn, CGL, NIR Shootout | No | The [Eigenvector dataset page](https://eigenvector.com/resources/data-sets/) provides downloads but no explicit open reuse license was identified for these four datasets. Their availability alone is not used as evidence of open licensing. |

For CC BY 4.0, retain attribution, link the license and indicate changes; see the
[license summary](https://creativecommons.org/licenses/by/4.0/). For MIT, preserve
the applicable copyright and permission notice; see the source repository's
[license](https://github.com/RNL1/Melamine-Dataset/blob/master/LICENSE).
The scripts print attribution instructions and evidence links before downloading.
No dataset is mirrored by this repository.

## OSSL scope and reproducibility

Inclusion follows OSSL's explicit declaration for its compiled data, not the MIT
license of its documentation/code alone. The existing L0 and L1 v1.2 URLs are
retained, as listed by the [official access documentation](https://docs.soilspectroscopy.org/db-access.html).
This is a source-level selection, not a per-row license audit. OSSL retains
[source-specific scan license fields](https://docs.soilspectroscopy.org/db-desc.html);
users must preserve applicable source notices when reusing the compilation.

The publisher states that bucket files may change without notice. This filter
does not pin file hashes or certify that today's downloads match the historical
benchmark snapshot. Do not replace the bucket files with a different Zenodo
snapshot without checking compatibility with the stored partition indices.
Future catalog changes or changes in upstream terms require a fresh review.

## Relation to the thesis

The subset follows the five open sources classified in the article, excluding
the four Eigenvector collections. In TRIP it retains Melamine (four tasks), Eggs
(one) and Wheat (one), preserving the original train/validation/test assignments.
Mango and soil remain separate benchmark groups. This is a reduced benchmark;
its results must not be presented as a reproduction of the complete benchmark.
The historical 109,858/111,767 sample counts are not recomputed or guaranteed by
this option.
