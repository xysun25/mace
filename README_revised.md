# MACE 微调改进说明

本文档记录对 `/home/xysun/work/mace`（`my-mods` 分支）的全部修改，包含两部分独立功能：

1. **固定原子 & 吸附质力权重**（训练过程）
2. **逐 Epoch 吸附能基准测试**（训练监控）

---

## 第一部分：固定原子与吸附质力权重

### 背景

吸附体系的训练数据中，表面底层原子在 CP2K 中被固定（`fixed=1`），其受力恒为零。
若将这些零力纳入损失函数，会稀释对吸附质原子力的学习信号。
同时，吸附质原子数量少但物理上更关键，需要额外加权。

### 修改的文件

#### `mace/data/utils.py`

从 extxyz 读取两个额外的逐原子字段：

```python
# fixed:I:1 —— 1=固定原子，0=自由原子
fixed = atoms.arrays.get("fixed", None)
if fixed is not None:
    properties["fixed"] = np.array(fixed, dtype=np.int32)

# atom_forces_weight:R:1 —— 每个原子力的权重
atom_forces_weight = atoms.arrays.get("atom_forces_weight", None)
if atom_forces_weight is not None:
    properties["atom_forces_weight"] = np.array(atom_forces_weight, dtype=np.float64)
```

#### `mace/data/atomic_data.py`

将 `fixed` 和 `atom_forces_weight` 存入 `AtomicData` 对象，
无 `fixed` 字段时自动填充全零张量（视为全部自由）：

```python
# fixed: [n_atoms] long tensor，无标签时默认全 0（全部自由原子）
if config.properties.get("fixed") is not None:
    cls_kwargs["fixed"] = torch.tensor(config.properties["fixed"], dtype=torch.long)
else:
    cls_kwargs["fixed"] = torch.zeros(num_atoms, dtype=torch.long)
```

#### `mace/modules/loss.py`

新增辅助函数 `_get_free_atom_mask()`，在所有损失函数中：

- **排除固定原子**：`fixed==1` 的原子不参与力损失计算
- **应用逐原子权重**：吸附质原子可赋予更高权重

影响的损失函数类：`WeightedEnergyForcesLoss`、`WeightedHuberEnergyForcesStressLoss`、`UniversalLoss`。

```python
def _get_free_atom_mask(ref: Batch) -> Optional[torch.Tensor]:
    """返回自由原子的布尔掩码 [n_atoms]，无 fixed 字段时返回 None。"""
    if hasattr(ref, "fixed") and ref.fixed is not None:
        fixed = ref.fixed
        if fixed.dim() > 1:
            fixed = fixed.squeeze(-1)
        return fixed == 0
    return None
```

#### `mace/tools/train.py`（`MACELoss.update`）

验证指标（RMSE/MAE）同样只统计自由原子的力误差：

```python
if hasattr(batch, "fixed") and batch.fixed is not None:
    free_mask = batch.fixed.squeeze(-1) == 0
    ref_forces = batch.forces[free_mask]
    pred_forces = output["forces"][free_mask]
    # 只用自由原子计算 RMSE/MAE
```

#### `mace/tools/scripts_utils.py`

在 `log_dataset_contents` 中兼容处理 `fixed` 和 `atom_forces_weight`
不存在时的 KeyError。

### 训练数据准备

#### extxyz 格式要求

训练数据需包含 `fixed` 和（可选）`atom_forces_weight` 列：

```
246
Lattice="..." Properties=species:S:1:pos:R:3:fixed:I:1:forces:R:3 energy=...
Rh  15.58  1.916  11.293  0  -0.241  -0.937   0.346
Rh  18.287 1.899  11.497  0  -0.286  -0.129  -0.237
...
```

- `fixed=0`：自由原子（参与力损失）
- `fixed=1`：固定原子（排除在力损失之外）

若加入 `atom_forces_weight`：

```
Properties=species:S:1:pos:R:3:fixed:I:1:atom_forces_weight:R:1:forces:R:3
Rh  ...  0  1.0  ...    # 表面原子，权重 1.0
C   ...  0  5.0  ...    # 吸附质 C，权重 5.0
O   ...  0  5.0  ...    # 吸附质 O，权重 5.0
```

#### `add_weights.py`（工具脚本）

批量为训练/测试数据添加 `atom_forces_weight` 字段：

```python
ADSORBATE_ELEMENTS = {"C", "H", "O"}   # 吸附质元素
ADSORBATE_WEIGHT = 5.0                  # 吸附质原子的力权重

# 用法
python add_weights.py
# 输入: train1.extxyz, test1.extxyz
# 输出: train1_weighted.extxyz, test1_weighted.extxyz
```

---

## 第二部分：逐 Epoch 吸附能基准测试

### 背景

微调过程中难以直观判断吸附能预测质量。
新增回调函数在每个 `eval_interval` 结束时自动运行基准测试：
用当前模型权重计算所有吸附位点的预测吸附能，与 DFT 参考值对比，
结果写入训练 log 并保存图表。

### 新增文件

#### `mace/tools/adsorption_benchmark.py`

| 函数 | 说明 |
|------|------|
| `_parse_cp2k_fixed_atoms(inp_path)` | 解析 CP2K `.inp` 中 `&FIXED_ATOMS LIST`，1-indexed → 0-indexed |
| `get_fixed_atoms(atoms, structure_dir)` | 获取固定原子索引（见下节优先级） |
| `extract_cp2k_pdb_energy(pdb_path)` | 从 CP2K PDB 轨迹 REMARK 行读取最终能量（Hartree → eV） |
| `load_adsorption_system(surface_dir, gas_dir, ads_dir)` | 加载表面/气体/吸附结构，计算 DFT 参考吸附能 |
| `run_adsorption_benchmark(calc, structures, dft_df, ...)` | 单次基准计算（一种组合） |
| `save_parity_plot(df, plot_path)` | 保存 MACE vs DFT parity plot（PNG） |
| `make_adsorption_benchmark_fn(...)` | **工厂函数**，返回回调 `fn(epoch, model)` |

**固定原子判断优先级：**

1. 结构文件（PDB/extxyz）中已有 `FixAtoms` 约束 → 直接使用
2. 结构目录下存在 `*.inp` → 解析 `&FIXED_ATOMS LIST`
   - 支持：`LIST 1 2 3 4`，`LIST 1..10 15 20..25`，多个块
   - CP2K 1-indexed 自动转为 ASE 0-indexed
3. 均未找到 → **无固定原子**（适用于纳米颗粒等非周期性体系）

已验证：`opt_17/cp2k.inp` 解析结果与原脚本硬编码的 32 个索引完全一致。

### 修改的文件

#### `mace/tools/train.py`

`train()` 新增可选参数：

```python
def train(..., adsorption_benchmark_fn=None):
    ...
    # 在每个 eval_interval 的验证之后、rank 0 上调用
    if adsorption_benchmark_fn is not None and rank == 0:
        try:
            adsorption_benchmark_fn(epoch, model_to_evaluate)
        except Exception as e:
            logging.warning(f"Adsorption benchmark failed at epoch {epoch}: {e}")
```

EMA 启用时，使用 EMA 平均后的权重（`model_to_evaluate` 已在 `param_context` 内）。

#### `mace/tools/arg_parser.py`

新增 7 个 CLI 参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--adsorption_benchmark` | `False` | 是否启用 |
| `--adsorption_surface_dir` | `None` | 清洁表面目录（含 `*-pos-1.pdb`，可选 `*.inp`） |
| `--adsorption_gas_dir` | `None` | 气体分子目录 |
| `--adsorption_ads_dir` | `None` | 吸附结构根目录（含 `opt_*` 子目录） |
| `--adsorption_output_dir` | `None` | 输出目录（默认 `<results_dir>/adsorption_benchmark/`） |
| `--adsorption_fmax` | `0.05` | relax 模式力收敛标准（eV/Å） |
| `--adsorption_benchmark_device` | `cpu` | 基准计算设备 |

#### `mace/cli/run_train.py`

训练启动时构造基准回调并传入训练循环。

### 8 种计算组合

自动遍历全部 `slab_ref × gas_ref × ads_mode` 组合，无需手动指定：

| combo_tag | slab_ref | gas_ref | ads_mode |
|-----------|----------|---------|----------|
| `slab-mlff_gas-mlff_sp` | MLFF | MLFF | 单点 |
| `slab-mlff_gas-mlff_relax` | MLFF | MLFF | 弛豫 |
| `slab-mlff_gas-dft_sp` | MLFF | DFT | 单点 |
| `slab-mlff_gas-dft_relax` | MLFF | DFT | 弛豫 |
| `slab-dft_gas-mlff_sp` | DFT | MLFF | 单点 |
| `slab-dft_gas-mlff_relax` | DFT | MLFF | 弛豫 |
| `slab-dft_gas-dft_sp` | DFT | DFT | 单点 |
| `slab-dft_gas-dft_relax` | DFT | DFT | 弛豫 |

吸附能公式：`E_ads = E(slab+ads) - E(slab) - E(gas)`

---

## 使用方法

### 训练数据准备

```bash
# 1. 为数据添加 atom_forces_weight（可选但推荐）
python add_weights.py
# 输出 train1_weighted.extxyz, test1_weighted.extxyz

# 2. 确认 extxyz 包含 fixed 列（应在 CP2K→extxyz 转换时已写入）
head -2 train1_weighted.extxyz
# Properties=...fixed:I:1:atom_forces_weight:R:1:forces:R:3...
```

### 微调训练脚本（SLURM）

参考 `ft_mace.sh`：

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

### 输入目录结构

```
surface_dir/          ← 清洁表面（e.g. small/opt_17）
    *-pos-1.pdb       必须
    cp2k.inp          可选，用于读取固定原子

gas_dir/              ← 气体分子（e.g. small/opt_18）
    *-pos-1.pdb

ads_dir/              ← 吸附结构根目录（e.g. small/）
    opt_1/
        *-pos-1.pdb
    opt_2/
        *-pos-1.pdb
    ...
```

`ads_dir` 可与 `surface_dir`/`gas_dir` 共用父目录。

### 输出文件

每个 epoch 在 `--adsorption_output_dir` 下生成：

```
adsorption_benchmark/
    adsorption_benchmark_epoch0001_slab-mlff_gas-mlff_sp.csv
    adsorption_benchmark_epoch0001_slab-mlff_gas-mlff_sp.png
    ...（共 8 对 CSV + PNG）
    adsorption_benchmark_epoch0001_all.csv
    adsorption_benchmark_epoch0002_...
```

训练 log 中同步打印：

```
═══════════════════════════════════════════════════════════════
AdsorptionBenchmark — Epoch 3  (8 combos)
  [slab-dft_gas-dft_sp]       MAE=0.089 eV  RMSE=0.102 eV  Max=0.167 eV
  [slab-mlff_gas-mlff_sp]     MAE=0.123 eV  RMSE=0.145 eV  Max=0.231 eV
  ...
  ── Epoch 3 Summary ──
  combo                                MAE     RMSE
  slab-dft_gas-dft_sp               0.089    0.102
  slab-mlff_gas-mlff_sp             0.123    0.145
  ...
```

---

## 第三部分：多表面/多吸附质基准测试

### 背景

实际微调场景中往往涉及多种表面（100/111/211/stepped）和多种吸附质（CO、CO2、H 等）。
原有基准测试只支持单一表面，无法同时对比各表面的预测精度。
本次修改扩展为多系统模式，支持任意数量的表面与吸附质组合，每个系统独立输出结果，并提供跨系统汇总图表。

### 修改的文件

#### `mace/tools/adsorption_benchmark.py`

**`make_adsorption_benchmark_fn` 新签名（向后兼容）：**

```python
def make_adsorption_benchmark_fn(
    output_dir: str,
    systems: Optional[List[Dict]] = None,   # 新增：多系统列表
    surface_dir: Optional[str] = None,       # 旧式单系统（保留兼容）
    gas_dir: Optional[str] = None,
    ads_dir: Optional[str] = None,
    device: str = "cpu",
    fmax: float = 0.05,
)
```

每个 `systems` 条目：

```python
{
    "name": "CO/100",          # 显示在日志、文件名、图例中的标签
    "surface_dir": "...",      # 清洁表面目录（含 *-pos-1.pdb）
    "gas_dir": "...",          # 气体分子目录
    "ads_dir": "...",          # 含 opt_* 子目录的根目录
}
```

**新增 `save_multi_system_parity_plot()`：**

在一张图上用不同颜色绘制各系统的 parity plot，标题显示全部系统的总体 MAE/RMSE。

**每 epoch 输出结构变化：**

```
adsorption_benchmark/
    CO_100/                                         ← 每系统独立子目录
        adsorption_benchmark_epoch0001_slab-dft_gas-dft_sp.csv
        adsorption_benchmark_epoch0001_slab-dft_gas-dft_sp.png
        ...（8 对 CSV + PNG）
        adsorption_benchmark_epoch0001_all.csv
    CO_111/
        ...
    CO_211/
        ...
    CO_stepped/
        ...
    adsorption_benchmark_epoch0001_slab-dft_gas-dft_sp_all_systems.png   ← 跨系统对比图
    adsorption_benchmark_epoch0001_slab-mlff_gas-mlff_sp_all_systems.png
    ...（每种 combo 一张跨系统图）
    adsorption_benchmark_epoch0001_all.csv                               ← 全系统全 combo 合并 CSV
```

#### `mace/tools/arg_parser.py`

新增参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--adsorption_benchmark_config` | `None` | JSON 配置文件路径（多系统模式，优先于旧式三参数） |

原有 7 个参数**保持不变**，单系统旧式写法仍然有效。

#### `mace/cli/run_train.py`

- 有 `--adsorption_benchmark_config` 时：读取 JSON 文件，校验每条目含 `name/surface_dir/gas_dir/ads_dir`，调用多系统模式
- 无 `--adsorption_benchmark_config` 时：回退到原有单系统逻辑

### JSON 配置文件格式

```json
[
  {
    "name": "CO/100",
    "surface_dir": "/path/to/small_100/slab",
    "gas_dir": "/path/to/CO_gas_reference",
    "ads_dir": "/path/to/small_100"
  },
  {
    "name": "CO/211",
    "surface_dir": "/path/to/small_211/opt_21",
    "gas_dir": "/path/to/CO_gas_reference",
    "ads_dir": "/path/to/small_211"
  },
  {
    "name": "CO2/100",
    "surface_dir": "/path/to/CO2/small_100/slab",
    "gas_dir": "/path/to/CO2_gas_reference",
    "ads_dir": "/path/to/CO2/small_100"
  }
]
```

**要点：**
- 不同吸附质（CO、CO2、H）各自有独立的 `gas_dir`（CO 气相参考 ≠ CO2 气相参考）
- 同一种吸附质的不同表面可**共享** `gas_dir`（例如 CO 在 100/111/211/stepped 四个表面共用同一个 CO 气相计算目录）
- `ads_dir` 中的 `opt_*` 目录只需含 `*-pos-1.pdb` 即可；代码会自动排除与 `surface_dir` 或 `gas_dir` 指向相同路径的 opt_ 目录

### 当前可用的 CO 配置文件

已生成，可直接使用：

```
/home/xysun/work/FTuMLP1/adsdata/benchmark_config_CO.json
```

该文件定义了 4 个系统：

| name | surface_dir | gas_dir | ads_dir | 站点数 |
|------|-------------|---------|---------|--------|
| CO/100 | `small_100/slab` | `small_steped/opt_18` | `small_100` | 14 |
| CO/111 | `small_111/slab` | `small_steped/opt_18` | `small_111` | 13 |
| CO/211 | `small_211/opt_21` | `small_steped/opt_18` | `small_211` | 20 |
| CO/stepped | `small_steped/opt_17` | `small_steped/opt_18` | `small_steped` | 16 |

> **说明：**
> - `small_100`/`small_111`：表面在各自的 `slab/` 子目录
> - `small_211`：洁净表面在 `opt_21/`（经验证只含 Rh+Fe，无 C/O），共 20 个 CO+slab 站点（opt_1~opt_20）
> - `small_steped`：洁净表面在 `opt_17/`，CO 气相在 `opt_18/`，16 个吸附站点（opt_1~opt_16）
> - 四个表面共用 `small_steped/opt_18` 作为 CO 气相参考

### 训练脚本写法

**多表面模式：**

```bash
mace_run_train \
    ...（其他参数不变）...
    --adsorption_benchmark=True \
    --adsorption_benchmark_config="/home/xysun/work/FTuMLP1/adsdata/benchmark_config_CO.json" \
    --adsorption_output_dir="results/adsorption_benchmark" \
    --adsorption_fmax=0.05 \
    --adsorption_benchmark_device=cpu
```

**单表面模式（旧写法，仍然有效）：**

```bash
mace_run_train \
    ...
    --adsorption_benchmark=True \
    --adsorption_surface_dir="/path/to/opt_17" \
    --adsorption_gas_dir="/path/to/opt_18" \
    --adsorption_ads_dir="/path/to/small" \
    --adsorption_output_dir="results/adsorption_benchmark" \
    --adsorption_benchmark_device=cpu
```

### 训练 log 示例（多系统）

```
═════════════════════════════════════════════════════════════════
AdsorptionBenchmark — Epoch 3  (4 system(s), 8 combos each)

  ── System: CO/100 ──
    [slab-dft_gas-dft_sp]   MAE=0.082 eV  RMSE=0.095 eV  Max=0.154 eV
    ...

  ── System: CO/111 ──
    [slab-dft_gas-dft_sp]   MAE=0.097 eV  RMSE=0.113 eV  Max=0.189 eV
    ...

  ── System: CO/211 ──
    ...

  ── System: CO/stepped ──
    ...

  ── Epoch 3 Cross-System Summary ──
  system                combo                                  MAE     RMSE
  ──────────────────────────────────────────────────────────────────────────
  CO/100                slab-dft_gas-dft_sp                  0.082    0.095
  CO/111                slab-dft_gas-dft_sp                  0.097    0.113
  CO/211                slab-dft_gas-dft_sp                  0.105    0.121
  CO/stepped            slab-dft_gas-dft_sp                  0.078    0.091
  ...
```

### 后续扩展 CO2/H 体系

准备好 DFT 数据后，只需向 JSON 文件追加新条目：

```json
{
  "name": "CO2/100",
  "surface_dir": "/path/to/CO2_data/small_100/slab_or_opt_XX",
  "gas_dir": "/path/to/CO2_data/gas_CO2_opt_XX",
  "ads_dir": "/path/to/CO2_data/small_100"
}
```

模板文件：`/home/xysun/work/FTuMLP1/adsdata/benchmark_config_CO2_H_template.json`

---

## 注意事项

1. `fixed=1` 的原子**不参与力损失和验证指标**，与 CP2K 固定原子受力为零的物理含义一致。
2. `atom_forces_weight` 不影响能量损失，只作用于力损失。
3. 基准测试在 CPU 上运行（`--adsorption_benchmark_device=cpu`），不占用训练 GPU。
4. `ads_mode=relax` 的 4 种组合会在 CPU 上做 ASE 弛豫，耗时较长；可通过提高 `--eval_interval` 降低调用频率。
5. EMA 启用时，基准测试使用 EMA 平均后的模型权重。
6. 多系统模式中，不同吸附质（CO、CO2、H）必须各自提供独立的气相参考目录（`gas_dir`），不可共用。同种吸附质的不同表面可以共用同一个 `gas_dir`。
