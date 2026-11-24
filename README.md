# ViZDoom 敌人检测示例

从 ViZDoom 内置场景采集同帧截图与真值框，微调单类 YOLO 模型，再把截图上的预测输出为 JSON。

## Windows 环境

使用 64 位 Python 3.12。在仓库目录打开 PowerShell。

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

如果使用 NVIDIA 显卡，通过 [PyTorch 安装选择器](https://pytorch.org/get-started/locally/) 选择 Windows、Pip 和兼容的 CUDA 版本，并用 `.\.venv\Scripts\python.exe -m pip` 执行生成的安装命令。然后安装项目依赖。

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
```

## 采集与训练

先采集少量 `basic` 回合，检查 `data/smoke/previews` 里的绿色边框是否与怪物对齐。

```powershell
.\.venv\Scripts\python.exe collect.py --output data/smoke --episodes 12
.\.venv\Scripts\python.exe train.py --data data/smoke/dataset.yaml --epochs 1 --name doom-smoke
```

确认能完成一轮后，再采集更多回合并正式训练。

```powershell
.\.venv\Scripts\python.exe collect.py --scenarios basic defend_the_center defend_the_line --episodes 100
.\.venv\Scripts\python.exe train.py --epochs 50 --batch 8
```

每个场景按回合划分训练、验证和测试集。标注为类别 `0`，名称是 `enemy`。无目标图片会作为背景样本保留。采集目录必须为空，每次采集可以用 `--output` 指定新目录。脚本会打印观察到的对象名称，方便检查场景对应的敌人名称。

训练默认使用第一块可用的 CUDA 显卡，否则使用 CPU。可以用 `--device 0` 或 `--device cpu` 明确指定。显存不足时减小 `--batch`。首次训练会下载预训练权重。训练结束会打印 `best.pt` 的路径和独立测试集的 mAP。

## 预测并输出 JSON

在 `$weights` 变量中填入训练结束时打印的路径。下面以第一次正式训练的目录为例。

```powershell
$weights = 'runs/doom/weights/best.pt'
$image = (Get-ChildItem data/doom/images/test -Filter *.png | Select-Object -First 1).FullName
.\.venv\Scripts\python.exe predict.py --weights $weights --image $image --confidence 0.25
```

输出中的 `bbox` 是相对原图尺寸归一化的中心坐标和宽高。`horizontal_offset` 为负表示目标偏左，为正表示偏右。画面无检测时，`detections` 为空数组。

训练结果、权重和数据集保存在训练电脑本地。更换电脑时，可以重新采集数据，也可以单独复制 `data/doom` 和模型权重。
