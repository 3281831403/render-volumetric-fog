# render-volumetric-fog

## 题面

场景加体积雾后，远处一片死白、相机后退还能看到雾：ray_box 求交不做近平面裁剪，相机后也继续积分；step_count 向下取整，最后一段丢失；sample_positions 的抖动超过步长，出现条带；density 为负不钳制，透射率大于 1；extinction 不乘步长，结果与步数无关；integrate 用简单累加，不做透射率衰减；内散射不加相位函数，各向同性错误；步长过大不警告，出现明显条带；fog_batch 共用累加器，射线之间串扰；steps<=0 不校验，发生除零。fog/march.py 是核心，tests/test_march.py 15个红。要修的是：按 `fog/march.py` 中的契约修复上述十个缺陷，并让 15 条测试全部通过。

## 运行方式

在仓库根目录执行：

```powershell
python -m pytest tests -q
```

初始状态必须恰好是 `15 failed`，且没有 collection error、import error 或 syntax error。修复后目标为 `15 passed`。运行只需要 Python 3 与 pytest，所有计算均为离线、确定性计算，不使用随机数和外部依赖。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 图形渲染 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 约束

`tests/test_march.py` 是固定的验收测试，不可修改。实现只应改 `fog/march.py`（必要时可补全 `fog/__init__.py` 的导出），并保持所有公开函数签名不变。距离统一使用世界单位，消光系数使用世界单位的倒数，数值断言容差为 `1e-6`。
