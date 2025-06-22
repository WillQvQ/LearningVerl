# prepare_multi_region_data.py
import pandas as pd
import json
from typing import List, Dict

def create_sample_multi_region_data():
    """创建多区域学习数据样例"""
    
    # 示例1：学术论文格式
    paper_sample = {
        "text": """
        摘要：本文提出了一种新的深度学习方法用于自然语言处理任务。
        
        1. 引言
        自然语言处理（NLP）是人工智能的重要分支。传统的方法包括基于规则的方法和统计方法。
        
        2. 相关工作
        近年来，深度学习在NLP领域取得了显著进展。Transformer架构的提出revolutionized了这个领域。
        
        3. 方法论
        我们提出的方法基于注意力机制。具体来说，我们使用多头自注意力来捕获长距离依赖关系。
        数学公式：Attention(Q,K,V) = softmax(QK^T/√d_k)V
        
        4. 实验结果
        在标准数据集上，我们的方法达到了state-of-the-art的性能。具体结果如下表所示。
        
        5. 结论
        本文提出的方法有效提升了NLP任务的性能，为未来研究提供了新的思路。
        """,
        "learning_regions": [
            # 学习关键概念定义
            {"start_char": 150, "end_char": 200, "weight": 2.0, "region_type": "definition"},
            # 学习方法论核心
            {"start_char": 350, "end_char": 450, "weight": 3.0, "region_type": "methodology"},
            # 学习数学公式
            {"start_char": 500, "end_char": 550, "weight": 2.5, "region_type": "formula"},
            # 学习结论
            {"start_char": 650, "end_char": 720, "weight": 1.5, "region_type": "conclusion"}
        ]
    }
    
    # 示例2：技术文档格式
    tech_doc_sample = {
        "text": """
        <LEARN weight="1.0" type="overview">
        Docker是一个开源的容器化平台，它允许开发者将应用程序及其依赖打包到轻量级的容器中。
        </LEARN>
        
        容器化技术的历史可以追溯到早期的虚拟化技术。
        
        <LEARN weight="2.5" type="core_concept">
        Docker的核心概念包括镜像(Image)、容器(Container)和仓库(Registry)。
        镜像是只读的模板，容器是镜像的运行实例。
        </LEARN>
        
        以下是一些使用示例：
        
        <LEARN weight="2.0" type="practical_usage">
        要创建一个Docker镜像，首先需要编写Dockerfile文件。
        典型的Dockerfile包含FROM、RUN、COPY等指令。
        </LEARN>
        
        更多详细信息请参考官方文档。
        
        <LEARN weight="1.5" type="best_practice">
        最佳实践包括：使用多阶段构建、最小化镜像层数、合理使用缓存等。
        </LEARN>
        """,
        "default_weight": 0.0
    }
    
    # 示例3：结构化教程格式
    tutorial_sample = {
        "segments": [
            {"content": "欢迎来到机器学习入门教程", "learn": False, "weight": 0.0},
            {
                "content": "第一节：什么是机器学习？机器学习是让计算机通过数据学习模式的技术。",
                "learn": True, "weight": 2.0, "segment_type": "concept"
            },
            {"content": "让我们通过一个例子来理解：", "learn": False, "weight": 0.0},
            {
                "content": "假设我们要预测房价。我们收集房屋面积、位置等特征作为输入，房价作为输出。",
                "learn": True, "weight": 1.5, "segment_type": "example"
            },
            {"content": "代码示例：\n```python\nimport sklearn\n```", "learn": False, "weight": 0.0},
            {
                "content": "机器学习的主要步骤包括：数据收集、特征工程、模型训练、模型评估。",
                "learn": True, "weight": 2.5, "segment_type": "process"
            },
            {"content": "接下来我们将详细介绍每个步骤...", "learn": False, "weight": 0.0},
            {
                "content": "总结：机器学习是一个从数据中学习规律并做出预测的过程。",
                "learn": True, "weight": 1.0, "segment_type": "summary"
            }
        ]
    }
    
    # 创建数据集
    samples = [
        {"format": "position_based", "data": paper_sample},
        {"format": "tag_based", "data": tech_doc_sample}, 
        {"format": "structured", "data": tutorial_sample}
    ]
    
    # 扩展数据集（生成更多样例）
    extended_samples = []
    for i in range(20):  # 生成20个样例
        for base_sample in samples:
            sample = base_sample.copy()
            sample["id"] = len(extended_samples)
            extended_samples.append(sample)
    
    # 保存为parquet文件
    df = pd.DataFrame(extended_samples)
    df.to_parquet("multi_region_data.parquet", index=False)
    print(f"已生成 {len(extended_samples)} 个多区域学习样例")
    
    return "multi_region_data.parquet"

def validate_multi_region_data(file_path):
    """验证多区域数据的格式和完整性"""
    df = pd.read_parquet(file_path)
    
    print(f"数据集包含 {len(df)} 个样例")
    
    format_counts = df["format"].value_counts()
    print("\\n格式分布：")
    for format_type, count in format_counts.items():
        print(f"  {format_type}: {count}")
    
    # 检查每种格式的样例
    for format_type in df["format"].unique():
        sample = df[df["format"] == format_type].iloc[0]
        print(f"\\n{format_type} 格式样例：")
        print(f"  数据键: {list(sample['data'].keys())}")
        
        if format_type == "position_based":
            regions = sample['data'].get('learning_regions', [])
            print(f"  学习区域数量: {len(regions)}")
            
        elif format_type == "tag_based":
            text = sample['data'].get('text', '')
            tag_count = text.count('<LEARN')
            print(f"  LEARN标签数量: {tag_count}")
            
        elif format_type == "structured":
            segments = sample['data'].get('segments', [])
            learn_segments = [s for s in segments if s.get('learn', False)]
            print(f"  总段落数: {len(segments)}, 学习段落数: {len(learn_segments)}")

if __name__ == "__main__":
    file_path = create_sample_multi_region_data()
    validate_multi_region_data(file_path)