# mesh-normals-tangents

## 题面

接上法线贴图后，硬边被抹平、镜像模型还整片发黑：面法线不归一，光照强度随面积变化；退化三角形产生 NaN，未标记；平滑不加角度阈值，硬边被抹平；切线按位置差算而不看 UV，贴图方向错；切线不做正交化，法线贴图出现拉伸；手性不检查，左右镜像模型法线贴图反向；镜像 UV 不翻转 bitangent，缝处出现黑线；顶点法线平均不按面积加权，小面主导；批量共用切线缓冲，网格之间串；indices 越界不校验，索引错误。mesh/tangent.py 是核心，tests/test_tangent.py 15个红。要修的是：面法线归一化、退化三角形零向量标记、平滑角度阈值、按 UV 求切线、切线正交化、手性检查、镜像 UV 的 bitangent 翻转、面积加权平均、批量缓冲隔离、indices 越界校验。

## 运行方式

在仓库根目录执行：

```powershell
python -m pytest tests -q
```

初始状态应恰好为 `15 failed`，且没有 collection error、import error 或 syntax error：

```text
15 failed in 0.XXs
```

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 图形渲染 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 约定

- 只使用 Python 标准库 `math` 与 `pytest`，离线运行，不使用随机数。
- 右手系：合法切线基满足 `cross(tangent, bitangent) == normal`。
- UV 坐标规范化在 `[0, 1]`。
- 浮点比较容差为 `1e-6`。
- 网格字段固定为 `{"positions", "normals", "uvs", "indices"}`，`indices` 为每三个一组的下标扁平列表。
- 自定义异常类为 `mesh.tangent.TangentError`。

## 不可修改

`tests/test_tangent.py` 不可修改。修复只能发生在 `mesh/tangent.py`，并保持现有公开函数签名与返回结构不变。
