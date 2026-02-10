# neosr (Debug Fork)

This repository is a personal fork of `neosr`, used for local debugging and experiments.

It is **not** the official project version.

For full documentation, updates, and official instructions, please refer to:

- Official repository: https://github.com/neosr-project/neosr
- Official wiki: https://github.com/neosr-project/neosr/wiki

## Model Analysis

### Project Structure & Unified Framework

The project `neosr` is a unified framework for training and testing super-resolution models, based on PyTorch.

**Core Components:**
*   **`neosr/archs/`**: This directory contains the source code for all network architectures. Each file typically defines one or more model classes and registers them.
*   **`neosr/models/`**: Defines high-level model logic (e.g., `ImageModel`, `VideoModel`). These classes handle the training loop, validation, and inference, wrapping the network architecture.
*   **`neosr/data/`**: Handles data loading and processing.
*   **`neosr/utils/registry.py`**: Implements a registry system (`ARCH_REGISTRY`, `MODEL_REGISTRY`, etc.). This allows models to be instantiated by name from configuration files, decoupling architecture definition from instantiation logic.
*   **Configuration**: The project uses TOML files (in `options/`) to configure training and testing. The `[network_g]` section in these files specifies the generator network architecture (`type`) and its hyperparameters.

**Unified Framework Mechanism:**
1.  **Registration**: Network architectures in `neosr/archs/` differ in implementation but are registered to `ARCH_REGISTRY` using the `@ARCH_REGISTRY.register()` decorator.
2.  **Instantiation**: The `build_network` function in `neosr/archs/__init__.py` takes a configuration dictionary (from TOML), looks up the model class in the registry, and instantiates it with the provided arguments.
3.  **Execution**: The `ImageModel` class (in `neosr/models/image.py`) uses `build_network` to create the network and then manages the forward pass, loss calculation, and optimization in a standardized way, allowing easy switching between different architectures.

### Model Parameters and FLOPs

The following table lists the Parameters (Params) in Millions [M] and Floating Point Operations (FLOPs) in Giga [G] for available super-resolution models.
*   **Input Resolution**: 64x64 (Low Resolution).
*   **Scale Factor**: 4x (unless otherwise noted or fixed by model).
*   **FLOPs**: Calculated on the 64x64 input.

| Model | Params [M] | FLOPs [G] |
| :--- | :--- | :--- |
| **asid** | 0.31 | 1.19 |
| **asid_d8** | 0.74 | 2.83 |
| **atd** | 19.88 | 77.73 |
| **atd_light** | 0.66 | 2.92 |
| **catanet** | 0.54 | 2.97 |
| **cfsr** | 0.30 | 1.22 |
| **compact** | 0.62 | 2.54 |
| **craft** | 0.75 | 4.90 |
| **cugan** | 1.41 | 8.29 |
| **dat_s** | 11.21 | 46.62 |
| **dat_m** | 14.80 | 61.30 |
| **dct** | 1.12 | 4.58 |
| **dctlsa** | 0.89 | 3.63 |
| **ditn** | 0.44 | 1.80 |
| **drct_s** | 2.38 | 11.90 |
| **drct** | 14.01 | 59.64 |
| **drct_l** | 27.33 | 114.51 |
| **drct_xl** | 31.77 | 132.80 |
| **eimn** | 1.00 | 3.90 |
| **eimn_a** | 0.88 | 3.43 |
| **esrgan** | 16.70 | 73.43 |
| **flexnet** | 2.87 | 11.72 |
| **grformer** | 0.80 | 3.40 |
| **grformer_medium** | 9.50 | 39.25 |
| **grformer_large** | 18.62 | 76.96 |
| **hasn** | 0.44 | 1.66 |
| **hat_s** | 9.36 | 40.83 |
| **hat_m** | 20.51 | 86.41 |
| **hat_l** | 40.32 | 168.04 |
| **hit_srf** | 0.87 | 3.73 |
| **hma** | 5.50 | 18.25 |
| **hma_medium** | 69.63 | 201.91 |
| **hma_large** | 138.56 | 399.04 |
| **light_safmnpp** | 0.07 | 0.27 |
| **lmlt** | 0.67 | 2.64 |
| **lmlt_tiny** | 0.25 | 0.98 |
| **lmlt_large** | 1.29 | 5.10 |
| **man** | 8.67 | 35.23 |
| **man_light** | 0.83 | 3.35 |
| **man_tiny** | 0.15 | 0.60 |
| **mfghmoe** | 26.83 | 110.41 |
| **microsr** | 14.01 | 59.23 |
| **microsr_light** | 2.38 | 11.62 |
| **moesr** | 16.53 | 49.77 |
| **mosrv2** | 4.18 | 17.08 |
| **msdan** | 0.23 | 1.46 |
| **ninasr** | 1.03 | 3.99 |
| **ninasr_b0** | 0.11 | 0.41 |
| **ninasr_b2** | 10.06 | 39.00 |
| **omnisr** | 0.79 | 3.15 |
| **plainusr** | 1.25 | 4.68 |
| **plainusr_large** | 3.04 | 11.64 |
| **plainusr_ultra** | 0.21 | 0.83 |
| **plksr** | 7.39 | 30.21 |
| **plksr_tiny** | 2.37 | 9.70 |
| **rcan** | 15.59 | 65.25 |
| **realplksr** | 7.39 | 30.21 |
| **realplksr_s** | 2.37 | 9.70 |
| **realplksr_l** | 16.59 | 67.89 |
| **rgt** | 13.36 | 51.91 |
| **rgt_s** | 10.19 | 40.12 |
| **safmn** | 0.24 | 0.96 |
| **safmn_l** | 5.59 | 22.81 |
| **sebica** | 0.04 | 0.17 |
| **sebica_mini** | 0.01 | 0.03 |
| **span** | 0.43 | 1.74 |
| **span_fast** | 0.18 | 0.75 |
| **spanplus_st** | 0.43 | 1.74 |
| **spanplus_sts** | 0.14 | 0.57 |
| **srformer_light** | 0.84 | 3.53 |
| **srformer_medium** | 10.43 | 56.85 |
| **swinir_small** | 0.90 | 3.77 |
| **swinir_medium** | 11.85 | 50.55 |
| **swinir_large** | 27.92 | 119.60 |

**Excluded / Not Profiled:**
*   **esc, spanplus (base), dunet, krgn**: Require CUDA or custom kernels.
*   **ea2fpn**: Requires external weight downloads.
