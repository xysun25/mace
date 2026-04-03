# MACE 微调改进：逐 Epoch 吸附能基准测试

本文档记录对 `/home/xysun/work/mace` 仓库的所有修改，以及新增功能的完整使用方法。

---

## 一、修改概述

在 MACE 微调训练流程中集成了**逐 epoch 吸附能基准测试**功能：每完成一轮 `eval_interval` 的训练，
自动使用当前模型权重计算所有吸附位点的预测吸附能，与 DFT 参考值对比，
并将结果（数据表 + parity 图）保存到指定目录，同时在训练 log 中打印汇总信息。

---

## 二、新增 / 修改的文件

### 2.1 新增文件

#### `mace/tools/adsorption_benchmark.py`

核心基准测试模块，包含以下关键函数：

| 函数 | 说明 |
|------|------|
| `get_fixed_atoms(atoms, structure_dir)` | 获取固定原子索引（见下节） |
| `_parse_cp2k_fixed_atoms(inp_path)` | 解析 CP2K `.inp` 文件中的 `&FIXED_ATOMS LIST` |
| `extract_cp2k_pdb_energy(pdb_path)` | 从 CP2K PDB 轨迹的 REMARK 行读取最终能量（Hartree → eV） |
| `load_adsorption_system(surface_dir, gas_dir, ads_dir)` | 加载表面、气体分子、吸附结构，计算 DFT 参考吸附能 |
| `run_adsorption_benchmark(calc, structures, dft_df, ...)` | 单次基准计算（一种 slab_ref/gas_ref/ads_mode 组合） |
| `save_parity_plot(df, plot_path)` | 保存 MACE vs DFT parity plot（PNG） |
| `make_adsorption_benchmark_fn(...)` | **工厂函数**，返回可传入训练循环的回调 `fn(epoch, model)` |

**固定原子判断逻辑（优先级从高到低）：**

1. 结构文件（PDB/extxyz）中已包含 `FixAtoms` 约束 → 直接使用
2. 结构目录下存在 `*.inp` 文件 → 解析 `&FIXED_ATOMS … LIST … &END FIXED_ATOMS`
   - 支持空格分隔整数：`LIST 1 2 3 4`
   - 支持范围写法：`LIST 1..10 15 20..25`
   - 支持多个 `&FIXED_ATOMS` 块
   - CP2K 1-indexed 自动转换为 ASE 0-indexed
3. 以上均未找到 → **无固定原子**（适用于纳米颗粒等非周期性体系）

固定原子信息仅从 `surface_dir` 读取一次，然后统一应用到所有吸附结构。

---

### 2.2 修改文件

#### `mace/tools/train.py`

`train()` 函数新增参数：

```python
def train(
    ...
    adsorption_benchmark_fn=None,   # 新增：fn(epoch, model) → None
):
```

在每个 `eval_interval` 对应的验证步骤完成后，调用该回调（仅 rank 0 执行）：

```python
if adsorption_benchmark_fn is not None and rank == 0:
    try:
        adsorption_benchmark_fn(epoch, model_to_evaluate)
    except Exception as e:
        logging.warning(f"Adsorption benchmark failed at epoch {epoch}: {e}")
```

---

#### `mace/tools/arg_parser.py`

`build_default_arg_parser()` 新增 6 个命令行参数：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--adsorption_benchmark` | bool | `False` | 是否启用逐 epoch 基准测试 |
| `--adsorption_surface_dir` | str | `None` | 清洁表面结构目录（含 `*-pos-1.pdb`，可选含 `*.inp`） |
| `--adsorption_gas_dir` | str | `None` | 气体分子结构目录 |
| `--adsorption_ads_dir` | str | `None` | 吸附结构根目录（含多个 `opt_*` 子目录） |
| `--adsorption_output_dir` | str | `None` | 输出目录（默认 `<results_dir>/adsorption_benchmark/`） |
| `--adsorption_fmax` | float | `0.05` | relax 模式的力收敛标准（eV/Å） |
| `--adsorption_benchmark_device` | str | `cpu` | 基准计算设备（建议 `cpu`，避免与训练 GPU 冲突） |

**注意**：不需要指定 `slab_ref`、`gas_ref`、`ads_mode`，程序会自动运行全部 8 种组合。

---

#### `mace/cli/run_train.py`

- 导入 `make_adsorption_benchmark_fn`
- 在 `tools.train()` 调用前，若 `--adsorption_benchmark=True` 则构造回调函数
- 将回调作为 `adsorption_benchmark_fn` 参数传入训练循环

---

## 三、吸附能计算方法

基准测试自动遍历 **8 种 (slab_ref × gas_ref × ads_mode) 组合**：

```
slab_ref ∈ {mlff, dft}  ×  gas_ref ∈ {mlff, dft}  ×  ads_mode ∈ {sp, relax}
```

| combo_tag | slab_ref | gas_ref | ads_mode | 说明 |
|-----------|----------|---------|----------|------|
| `slab-mlff_gas-mlff_sp` | MLFF | MLFF | 单点 | 所有能量均用当前模型计算 |
| `slab-mlff_gas-mlff_relax` | MLFF | MLFF | 弛豫 | 从 DFT 几何出发用 MLFF 弛豫后算能量 |
| `slab-mlff_gas-dft_sp` | MLFF | DFT | 单点 | 气体分子用 DFT 能量 |
| `slab-mlff_gas-dft_relax` | MLFF | DFT | 弛豫 | |
| `slab-dft_gas-mlff_sp` | DFT | MLFF | 单点 | 表面用 DFT 能量 |
| `slab-dft_gas-mlff_relax` | DFT | MLFF | 弛豫 | |
| `slab-dft_gas-dft_sp` | DFT | DFT | 单点 | 参考能量全用 DFT（最常用） |
| `slab-dft_gas-dft_relax` | DFT | DFT | 弛豫 | |

吸附能公式：

```
E_ads = E(slab+adsorbate) - E(slab) - E(gas)
```

---

## 四、输入目录结构

```
surface_dir/               ← 清洁表面
    *-pos-1.pdb            ← CP2K 几何优化轨迹（必须）
    cp2k.inp               ← CP2K 输入文件（可选，用于读取固定原子）

gas_dir/                   ← 气体分子
    *-pos-1.pdb

ads_dir/                   ← 吸附结构根目录
    opt_1/
        *-pos-1.pdb
    opt_2/
        *-pos-1.pdb
    ...
    opt_N/
        *-pos-1.pdb
```

`ads_dir` 允许与 `surface_dir`/`gas_dir` 的父目录相同（程序不重复加载表面和气体）。
例如，原 `FTuMLP1` 项目的 `small/` 目录包含 `opt_1..opt_18`，可直接这样指定：

```bash
--adsorption_surface_dir=".../small/opt_17"   # 清洁表面
--adsorption_gas_dir=".../small/opt_18"        # CO 气体
--adsorption_ads_dir=".../small"               # 吸附结构（程序自动跳过 opt_17/opt_18）
```

---

## 五、输出文件

每个 epoch，在 `--adsorption_output_dir` 下生成：

```
adsorption_benchmark/
    adsorption_benchmark_epoch0001_slab-mlff_gas-mlff_sp.csv
    adsorption_benchmark_epoch0001_slab-mlff_gas-mlff_sp.png
    adsorption_benchmark_epoch0001_slab-mlff_gas-mlff_relax.csv
    adsorption_benchmark_epoch0001_slab-mlff_gas-mlff_relax.png
    ...（共 8 对 CSV + PNG）
    adsorption_benchmark_epoch0001_all.csv   ← 8 种组合合并
    adsorption_benchmark_epoch0002_...
    ...
```

**CSV 列说明：**

| 列名 | 说明 |
|------|------|
| `model` | 模型标签（含 epoch 和 combo 信息） |
| `opt_dir` | 吸附结构目录名（如 `opt_1`） |
| `site` | 位点标签（如 `site_01`） |
| `slab_ref` | 表面参考能量来源 |
| `gas_ref` | 气体参考能量来源 |
| `ads_mode` | 能量计算模式 |
| `E_slab_eV` | 表面参考能量（eV） |
| `E_gas_eV` | 气体参考能量（eV） |
| `E_slab_ads_eV` | 吸附体系总能量（eV） |
| `E_ads_eV` | MACE 预测吸附能（eV） |
| `E_ads_dft_eV` | DFT 参考吸附能（eV） |
| `error_eV` | 误差 = `E_ads_eV - E_ads_dft_eV`（eV） |

**训练 log 中输出示例：**

```
═════════════════════════════════════════════════════════════════
AdsorptionBenchmark — Epoch 3  (8 combos)
  [slab-mlff_gas-mlff_sp]      MAE=0.123 eV  RMSE=0.145 eV  Max=0.231 eV
  [slab-mlff_gas-mlff_relax]   MAE=0.118 eV  RMSE=0.139 eV  Max=0.220 eV
  [slab-dft_gas-dft_sp]        MAE=0.089 eV  RMSE=0.102 eV  Max=0.167 eV
  ...

  ── Epoch 3 Summary ──
  combo                                MAE     RMSE
  -------------------------------------------------------
  slab-mlff_gas-mlff_sp              0.123    0.145
  slab-dft_gas-dft_sp                0.089    0.102
  ...
  Combined CSV: adsorption_benchmark_epoch0003_all.csv
```

---

## 六、使用方法

### 6.1 SLURM 脚本（推荐）

参考 `/home/xysun/work/ftmace/mh-1/omat-pbe/bs_16/ft_mace.sh`：

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=24:10:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --partition=h100
#SBATCH --job-name=train_mace

source /scratch/junchen/xysun/mace/.venv/bin/activate

mace_run_train \
    --name="MACE_lora" \
    --foundation_model="mh-1" \
    --foundation_head="omat_pbe" \
    --train_file="/scratch/junchen/xysun/ftmace/data/data1/train1.extxyz" \
    --valid_fraction=0.05 \
    --test_file="/scratch/junchen/xysun/ftmace/data/data1/test1.extxyz" \
    --lora=True \
    --lora_rank=4 \
    --lora_alpha=1.0 \
    --energy_weight=1.0 \
    --forces_weight=1.0 \
    --E0s="estimated" \
    --lr=0.005 \
    --weight_decay=0.0 \
    --ema \
    --ema_decay=0.995 \
    --amsgrad \
    --clip_grad=10.0 \
    --batch_size=16 \
    --max_num_epochs=12 \
    --restart_latest \
    --default_dtype="float64" \
    --device=cuda \
    --seed=3 \
    --adsorption_benchmark=True \
    --adsorption_surface_dir="/scratch/junchen/xysun/FTuMLP1/small/opt_17" \
    --adsorption_gas_dir="/scratch/junchen/xysun/FTuMLP1/small/opt_18" \
    --adsorption_ads_dir="/scratch/junchen/xysun/FTuMLP1/small" \
    --adsorption_output_dir="results/adsorption_benchmark" \
    --adsorption_fmax=0.05 \
    --adsorption_benchmark_device=cpu
```

**路径说明（根据实际情况修改）：**

| 参数 | 当前值 | 说明 |
|------|--------|------|
| `--adsorption_surface_dir` | `.../FTuMLP1/small/opt_17` | 清洁表面目录，内含 `*-pos-1.pdb` 和 `*.inp` |
| `--adsorption_gas_dir` | `.../FTuMLP1/small/opt_18` | 气体分子目录 |
| `--adsorption_ads_dir` | `.../FTuMLP1/small` | 含所有 `opt_*` 的父目录 |
| `--adsorption_output_dir` | `results/adsorption_benchmark` | 相对于工作目录的输出路径 |

### 6.2 纯命令行（无 SLURM）

```bash
mace_run_train \
    --name="test_run" \
    --foundation_model="mace-mp-0" \
    --train_file="train.extxyz" \
    --valid_fraction=0.1 \
    --E0s="average" \
    --max_num_epochs=10 \
    --device=cuda \
    --adsorption_benchmark=True \
    --adsorption_surface_dir="/path/to/slab_dir" \
    --adsorption_gas_dir="/path/to/gas_dir" \
    --adsorption_ads_dir="/path/to/adsorption_dirs"
```

### 6.3 不启用基准测试（默认行为不变）

直接省略所有 `--adsorption_*` 参数即可，训练流程与原版完全一致。

---

## 七、注意事项

1. **基准测试在 CPU 上运行**（`--adsorption_benchmark_device=cpu`），不占用训练 GPU，
   但对于 `ads_mode=relax` 的 4 种组合，CPU 弛豫可能较慢。如果 epoch 很多，
   可以通过提高 `--eval_interval` 来降低调用频率（默认每 epoch 调用一次）。

2. **模型权重是实时复制**：每次回调都对当前模型做 `deepcopy`，不影响训练状态。
   如果启用了 EMA（`--ema`），则使用 EMA 平均后的权重进行基准计算。

3. **固定原子从表面目录自动读取**：`surface_dir` 中的 `*.inp` 文件决定哪些原子固定，
   同样的约束会自动应用到所有吸附结构。气体分子不施加任何约束。

4. **ads_dir 与 surface_dir/gas_dir 可共用父目录**：程序通过目录名（`opt_*` 格式）
   遍历子目录，实际加载时根据 `surface_dir` 和 `gas_dir` 的内容自动识别，
   不会把表面/气体当作吸附位点重复计算。

5. **输出目录会自动创建**，无需提前 `mkdir`。

---

## 八、依赖

基准测试模块依赖以下包（均已在 MACE 环境中安装）：

- `ase`（结构读写、弛豫）
- `numpy`、`pandas`（数据处理）
- `matplotlib`（绘图，可选，缺失时跳过绘图但不报错）
