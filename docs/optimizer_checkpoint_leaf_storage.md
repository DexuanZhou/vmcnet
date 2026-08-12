# Optimizer PyTree checkpoint 分叶存储改造

## 1. 目的

本次改造解决大型 WSSR optimizer state 无法写入 checkpoint 的问题。修改仅涉及
checkpoint 的序列化和反序列化，不改变 optimizer 更新、低秩求解、训练循环或数值
结果。

相关实现：

- `vmcnet/utils/io.py`
- `tests/units/utils/test_io.py`

## 2. 原始故障

旧实现将完整 optimizer state 包装成一个 Python 对象后直接交给 `numpy.savez`：

```python
optimizer_state = {"single_element": optimizer_state}
np.savez(file_handle, e=epoch, d=data, p=params, o=optimizer_state, k=key)
```

因为 `optimizer_state` 是嵌套 PyTree，NumPy 会把它转成 object array，再通过 pickle
写入一个整体的 `o.npy`。当前环境中该路径使用 pickle protocol 3，单个序列化对象
超过 4 GiB 时会失败：

```text
OverflowError: serializing a bytes object larger than 4 GiB requires
pickle protocol 4 or higher
```

N2 rank-1600 WSSR 的 `sr_o` 形状对应约 740720 个参数和 1600 个历史方向；仅该
float32 数值叶子就约为 4.42 GiB。因此训练计算本身可以运行，但异步 checkpoint
线程在首次保存时失败，留下的 checkpoint 不完整，不能安全恢复或评估。

## 3. 新 checkpoint 格式

optimizer state 现在先通过 JAX PyTree API 拆分：

```python
leaves, tree_definition = jax.tree_util.tree_flatten(optimizer_state)
```

然后在同一个 `.npz` checkpoint 内分别保存：

```text
o                       小型格式标记，不再保存完整 object array
o_format                "pytree_leaves_v1"
o_treedef               PyTree 结构元数据
o_num_leaves            数值叶子数量
o_leaf_000000            第 0 个数值叶子
o_leaf_000001            第 1 个数值叶子
...
```

每个 `o_leaf_*` 都是原生 NumPy 数值数组，而不是 object array。因此大型 `sr_o`
会通过 NumPy 的直接数组写入和 ZIP64 路径保存，不再被转换成一个超过 4 GiB 的
pickle bytes 对象。

PyTree 结构本身很小，使用 pickle protocol 4 序列化为 `uint8` 数组。加载时先读取
所有数值叶子，再调用：

```python
jax.tree_util.tree_unflatten(tree_definition, leaves)
```

恢复原来的 dict、tuple、NamedTuple 和 `None` 等嵌套结构。

## 4. 具体代码改动

### 4.1 新增格式标记

在 `vmcnet/utils/io.py` 中新增：

```python
_OPTIMIZER_TREE_FORMAT = "pytree_leaves_v1"
_OPTIMIZER_TREE_FORMAT_KEY = "o_format"
_OPTIMIZER_TREEDEF_KEY = "o_treedef"
_OPTIMIZER_NUM_LEAVES_KEY = "o_num_leaves"
_OPTIMIZER_LEAF_PREFIX = "o_leaf_"
```

格式标记使加载端可以明确区分新旧 checkpoint。

### 4.2 新增逐叶序列化函数

新增 `_optimizer_state_to_npz_fields()`：

1. 使用 `jax.tree_util.tree_flatten()` 拆分 optimizer PyTree；
2. 将每个叶子转换成 NumPy 数组；
3. 拒绝 object-dtype 叶子，并给出明确错误；
4. 将数值叶子分别写入 `o_leaf_*`；
5. 单独保存叶子数量和 PyTree 结构。

### 4.3 修改 checkpoint 写入

`save_vmc_state()` 不再执行：

```python
o=optimizer_state
```

而是写入 `_optimizer_state_to_npz_fields()` 返回的独立字段：

```python
optimizer_fields = _optimizer_state_to_npz_fields(optimizer_state)
np.savez(..., **optimizer_fields)
```

epoch、walker data、parameters 和 PRNG key 的格式没有改变。

### 4.4 新增新旧格式兼容加载

新增 `_load_optimizer_state()`：

- checkpoint 含 `o_format` 时，按新格式读取数值叶子并重建 PyTree；
- checkpoint 不含 `o_format` 时，继续执行旧的
  `unwrap_if_singleton(npz_data["o"].tolist())` 路径。

因此当前代码可以继续读取已有的 KFAC、SPRING、MinSR 和 WSSR checkpoint。

兼容性方向如下：

| 读取程序 | 旧 checkpoint | 新 checkpoint |
|---|---:|---:|
| 修改后的 VMCNet | 支持 | 支持 |
| 未修改的旧版 VMCNet | 支持 | 不支持 |

## 5. 保持不变的部分

本次没有修改：

- WSSR/SR/SPRING/MinSR/KFAC 的任何更新公式；
- optimizer state 的逻辑结构和字段；
- learning rate、damping、norm constraint 或 clipping；
- checkpoint 的训练调用时机；
- checkpoint 的异步写入机制；
- parameters、walker data 和 PRNG key 的现有格式。

## 6. 测试改动

在 `tests/units/utils/test_io.py` 中增加了三类测试。

### 6.1 嵌套 PyTree 往返

使用包含以下结构的 NamedTuple optimizer state：

- 标量计数器；
- 二维 float64 数组；
- dict；
- tuple；
- JAX array；
- `None`。

验证保存后仍恢复为相同的 NamedTuple 类型，全部数值叶子一致。

### 6.2 独立数值成员检查

直接检查 `.npz` 内容，确认：

- 存在 `o_format`、`o_treedef` 和 `o_num_leaves`；
- `o` 不再是 object dtype；
- 每个 `o_leaf_*` 都不是 object dtype；
- optimizer 的数值 payload 确实被拆成多个 archive member。

### 6.3 旧 checkpoint 兼容

测试构造旧式 object-array checkpoint，再用新 loader 读取，确认原 optimizer tuple 和
数值状态能够正确恢复。

此外还使用实际的 `WSSRSolutionRecurrenceOptimizerState` 做了独立保存/加载验证，
恢复后的 optimizer 类型、`WSSRCoreState` 类型、`sr_o` 和 `solution_state` 均一致。

## 7. 验证结果

执行命令：

```bash
source /home/dexuan1/projects/rrg-ortner/dexuan1/venvs/vmcnet311/bin/activate
export JAX_PLATFORMS=cpu
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
pytest -vv -s tests/units/utils/test_io.py
```

结果：

```text
6 passed, 16 warnings in 115.92s
```

警告均来自现有 `jax.tree_map` 弃用提示，不是本次修改引入的失败。

代码空白检查：

```bash
git diff --check -- vmcnet/utils/io.py tests/units/utils/test_io.py
```

检查通过。

## 8. 使用注意事项

标准训练、恢复和推理路径应使用：

```python
vmcnet.utils.io.reload_vmc_state(...)
```

少数实验脚本过去直接执行：

```python
np.load(checkpoint)["o"].tolist()
```

这种写法只认识旧格式。对新 checkpoint，`o` 是格式标记，实际 optimizer state 位于
`o_leaf_*`。这些脚本若继续使用，需要改成调用 `reload_vmc_state()`。

## 9. 当前验证边界

已经验证了格式、PyTree 类型恢复、逐叶数值存储、旧格式兼容和实际 WSSR state
往返。目前尚未重新运行一次会生成约 4.42 GiB `sr_o` 的 N2 rank-1600 GPU
checkpoint；因此下一次该配置的短程 checkpoint smoke test 将是完整的端到端确认。
