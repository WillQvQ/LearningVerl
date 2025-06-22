# 多区域 Loss Mask 功能使用指南

## 概述

本功能实现了复杂长序列多区域 loss_mask 的高级配置，支持对不同文本区域进行精细化的学习控制。

## 支持的数据格式

### 1. 位置标记格式 (position_based)
基于字符位置来标记学习区域：

```python
{
    "text": "完整的文本内容...",
    "learning_regions": [
        {
            "start_char": 100,
            "end_char": 200,
            "weight": 2.0,
            "region_type": "concept"
        }
    ]
}
```

### 2. 标签格式 (tag_based)
使用特殊标签来标记学习区域：

```python
{
    "text": """
    这是普通文本。
    <LEARN weight="2.0" type="concept">
    这是需要学习的重要概念。
    </LEARN>
    """,
    "default_weight": 0.0
}
```

### 3. 结构化格式 (structured)
将文本分割为结构化的段落：

```python
{
    "segments": [
        {"content": "介绍部分", "learn": False, "weight": 0.0},
        {"content": "重要概念", "learn": True, "weight": 2.0, "segment_type": "concept"}
    ]
}
```

## 配置文件

项目提供了三种预配置的配置文件：

- `multi_region_position_based.yaml` - 基于位置标记
- `multi_region_tag_based.yaml` - 基于标签标记
- `multi_region_structured.yaml` - 结构化格式

## 使用步骤

### 1. 准备数据
```bash
cd verl/utils/dataset
python prepare_multi_region_data.py
```

### 2. 训练模型
```bash
# 位置标记格式
bash examples/sft/multi_region/run_multi_region_position.sh 4 ./outputs

# 标签格式
bash examples/sft/multi_region/run_multi_region_tag.sh 4 ./outputs

# 结构化格式
bash examples/sft/multi_region/run_multi_region_structured.sh 4 ./outputs
```

### 3. 调试和可视化
```python
from verl.utils.debug.multi_region_visualizer import LossMaskVisualizer
from verl.utils.debug.training_monitor import MultiRegionTrainingMonitor

# 可视化 loss_mask
visualizer = LossMaskVisualizer(tokenizer)
visualizer.print_detailed_analysis(input_ids, loss_mask, region_info)

# 训练监控
monitor = MultiRegionTrainingMonitor()
stats = monitor.log_batch_stats(batch, step)
```

## 高级功能

### 动态权重调整
```python
from verl.utils.dataset.dynamic_weight_adjuster import DynamicWeightAdjuster

adjuster = DynamicWeightAdjuster({
    "difficulty_scaling": True,
    "epoch_weights": {"1": 1.0, "2": 1.2, "3": 1.5}
})
```

### 区域依赖处理
```python
from verl.utils.dataset.region_dependency_handler import RegionDependencyHandler

handler = RegionDependencyHandler({
    "cascade_learning": True,
    "region_dependencies": {
        "methodology": {"requires": ["definition"], "penalty_factor": 0.5}
    }
})
```

## 测试

运行单元测试：
```bash
python -m pytest tests/test_multi_region_dataset.py -v
```

## 性能建议

1. **内存优化**：启用梯度检查点，降低批次大小
2. **序列并行**：对于超长序列考虑启用 `ulysses_sequence_parallel_size`
3. **权重归一化**：避免梯度爆炸问题
4. **学习率调整**：长序列可能需要更小的学习率
