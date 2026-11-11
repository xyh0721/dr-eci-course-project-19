# Dr.ECI 句内事件因果识别复现

机器智能课程设计第 4 个项目，第 19 组。基于 [Dr.ECI 作者仓库](https://github.com/DMIRLAB-Group/Dr.ECI) 提供的 EventStoryLine 句内事件对，比较直接提示、分阶段推理、投票阈值和全规则推理。

| 测试配置 | P (%) | R (%) | F1 (%) |
| --- | ---: | ---: | ---: |
| E0 直接提示 | 34.70 | 43.38 | 38.56 |
| E1 完整流程 | 36.21 | 31.41 | 33.64 |
| E2 三票门控 | 39.23 | 26.07 | 31.32 |
| E3 全规则（探索性） | 35.54 | 44.66 | 39.58 |

测试集含 2,678 个事件对和 468 个正例。E3 参考了原测试错误。两个 comparison.csv 文件记录验证和测试的各项计数。

## 使用方法

在仓库根目录使用 Python 3.10 或更高版本。单元测试和数据下载分别运行以下命令。

```sh
python -m unittest discover -s code -v
python code/download_data.py
```

configs 保存原实验配置，data_info 保存审计结果与固定划分，prompts 说明提示及规则。运行新的模型推理需要在本机设置 DR_ECI_API_KEY，服务端会按调用量计费。依次运行 python code/dreci.py validation、python code/dreci.py test 可生成 E0 至 E2 的逐条结果，之后使用 python code/e3.py run 运行 E3（具体子命令见脚本帮助）。code/verify_results.py 和 code/e3.py verify 用于核验本地生成的完整预测及缓存。公开仓库提供源码与指标汇总，逐请求响应保存在实验工作目录。

