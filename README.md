# About

Code for our paper about time series anomaly detection evaluation protocols published at TPCTC 2023 ([Multivariate Time Series Anomaly Detection: Fancy Algorithms and Flawed Evaluation Methodology](https://arxiv.org/abs/2308.13068)).

# Datasets

Please download the SWaT, WADI and PSM datasets and place original files in the
corresponding directory under `notebooks`. For WADI, run the provided script
(`prepare_WADI.sh`) to remove comments from data and shorten columns names.

- For SWaT and WADI: https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/
- For PSM: https://github.com/eBay/RANSynCoders



# Performance with PCA-based baseline

| Dataset | `F1` (point-wise) | `F1_c` (composite) | `F1_ew` (event-wise) |
| ------- | ----------------- | ------------------ | -------------------- |
| SWaT    | 0.810             | 0.596              | 0.555                |
| WADI    | 0.374             | 0.655              | 0.608                |
| PSM     | 0.538             | 0.484              | 0.20                 |


As mentioned in the notebooks, better `F1_c` scores can be achieved by disabling
score smoothing or by using a smaller smoothing window.
