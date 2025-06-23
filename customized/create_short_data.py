#!/usr/bin/env python3
"""
简化版简短中英文小说数据生成器（Qwen2.5格式）
生成1280个样本并提供详细的数据展示
"""

import os
import json
import pandas as pd
from typing import List, Dict, Any
from datetime import datetime


class SimpleShortDataGenerator:
    """简化的短数据生成器"""
    
    def __init__(self):
        # Qwen2.5配置
        self.system_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        self.user_query = "请你帮我写一个 4000 字的小说，并翻译成英文"
        
        # 模板
        self.template = {
            "system_start": "<|im_start|>system\n",
            "system_end": "<|im_end|>\n",
            "user_start": "<|im_start|>user\n", 
            "user_end": "<|im_end|>\n",
            "assistant_start": "<|im_start|>assistant\n",
            "assistant_end": "<|im_end|>"
        }
        
        # 简化的内容（只用一个主题，避免复杂性）
        self.base_content = {
            "topic": "友谊",
            "chinese": "小明和小红是最好的朋友。他们一起上学，一起玩耍，互相帮助。有一天，小明遇到了困难，小红毫不犹豫地伸出了援手。这就是真正的友谊。",
            "english": "Xiao Ming and Xiao Hong are best friends. They go to school together, play together, and help each other. One day, Xiao Ming encountered difficulties, and Xiao Hong reached out without hesitation. This is true friendship.",
            "summary": "这是一个关于友谊的温暖故事。"
        }
    
    def create_conversation(self) -> str:
        """创建完整对话"""
        assistant_response = (
            f"{self.base_content['chinese']}\n\n"
            f"<EN>\n\n"
            f"{self.base_content['english']}\n\n"
            f"</EN>\n\n"
            f"{self.base_content['summary']}"
        )
        
        return (
            f"{self.template['system_start']}{self.system_prompt}{self.template['system_end']}"
            f"{self.template['user_start']}{self.user_query}{self.template['user_end']}"
            f"{self.template['assistant_start']}{assistant_response}{self.template['assistant_end']}"
        )
    
    def create_segments(self) -> List[Dict[str, Any]]:
        """创建loss mask segments"""
        return [
            # System (不学习)
            {
                "content": f"{self.template['system_start']}{self.system_prompt}{self.template['system_end']}",
                "learn": False, "weight": 0.0, "segment_type": "system_prompt", "segment_id": 0
            },
            # User (不学习)
            {
                "content": f"{self.template['user_start']}{self.user_query}{self.template['user_end']}",
                "learn": False, "weight": 0.0, "segment_type": "user_query", "segment_id": 1
            },
            # Assistant开始 (不学习)
            {
                "content": self.template['assistant_start'],
                "learn": False, "weight": 0.0, "segment_type": "assistant_start_tag", "segment_id": 2
            },
            # 中文内容 (学习)
            {
                "content": self.base_content["chinese"],
                "learn": True, "weight": 1.0, "segment_type": "chinese_content", "segment_id": 3
            },
            # 换行
            {
                "content": "\n\n",
                "learn": True, "weight": 1.0, "segment_type": "separator", "segment_id": 4
            },
            # <EN> (高权重)
            {
                "content": "<EN>",
                "learn": True, "weight": 5.0, "segment_type": "special_token_start", "segment_id": 5
            },
            # 换行
            {
                "content": "\n\n",
                "learn": True, "weight": 1.0, "segment_type": "separator", "segment_id": 6
            },
            # 英文翻译 (学习)
            {
                "content": self.base_content["english"],
                "learn": True, "weight": 1.0, "segment_type": "english_translation", "segment_id": 7
            },
            # 换行
            {
                "content": "\n\n",
                "learn": True, "weight": 1.0, "segment_type": "separator", "segment_id": 8
            },
            # </EN> (高权重)
            {
                "content": "</EN>",
                "learn": True, "weight": 5.0, "segment_type": "special_token_end", "segment_id": 9
            },
            # 换行
            {
                "content": "\n\n",
                "learn": True, "weight": 1.0, "segment_type": "separator", "segment_id": 10
            },
            # 摘要 (中等权重)
            {
                "content": self.base_content["summary"],
                "learn": True, "weight": 2.0, "segment_type": "chinese_summary", "segment_id": 11
            },
            # Assistant结束 (学习)
            {
                "content": self.template['assistant_end'],
                "learn": True, "weight": 1.0, "segment_type": "assistant_end_tag", "segment_id": 12
            }
        ]
    
    def generate_samples(self, count: int = 1280) -> List[Dict[str, Any]]:
        """生成样本"""
        print(f"🔄 生成 {count} 个Qwen2.5格式样本...")
        
        samples = []
        conversation = self.create_conversation()
        segments = self.create_segments()
        
        for i in range(count):
            sample = {
                "format": "qwen25_structured",
                "conversation": conversation,
                "data": {"segments": segments},
                "id": f"qwen25_short_{i+1:04d}",
                "topic": self.base_content["topic"],
                "template": "qwen25",
                "total_segments": len(segments),
                "learning_segments": len([s for s in segments if s["learn"]]),
                "metadata": {
                    "generated_time": datetime.now().isoformat(),
                    "sample_type": "short_novel_qwen25",
                    "sample_index": i
                }
            }
            samples.append(sample)
            
            if (i + 1) % 80 == 0:
                print(f"  已生成: {i + 1}/{count} 个样本")
        
        print(f"✅ 生成完成: {len(samples)} 个样本")
        return samples
    
    def analyze_data(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """简化的数据分析"""
        if not samples:
            return {}
        
        # 基于第一个样本分析（因为所有样本都一样）
        sample = samples[0]
        segments = sample["data"]["segments"]
        
        analysis = {
            "total_samples": len(samples),
            "total_segments": len(segments),
            "learning_segments": len([s for s in segments if s["learn"]]),
            "learning_ratio": len([s for s in segments if s["learn"]]) / len(segments),
            "segment_types": {},
            "weight_distribution": {},
            "content_info": {
                "chinese_length": len(self.base_content["chinese"]),
                "english_length": len(self.base_content["english"]),
                "summary_length": len(self.base_content["summary"]),
                "conversation_length": len(sample["conversation"])
            }
        }
        
        # 统计段落类型和权重
        for seg in segments:
            seg_type = seg["segment_type"]
            weight = seg["weight"]
            
            analysis["segment_types"][seg_type] = analysis["segment_types"].get(seg_type, 0) + len(samples)
            weight_key = f"weight_{weight}"
            analysis["weight_distribution"][weight_key] = analysis["weight_distribution"].get(weight_key, 0) + len(samples)
        
        return analysis
    
    def display_sample_detail(self, sample: Dict[str, Any], sample_index: int = 0):
        """显示单个样本详情"""
        print(f"\n🔍 样本详情 #{sample_index}")
        print("=" * 80)
        print(f"ID: {sample['id']}")
        print(f"主题: {sample['topic']}")
        print(f"格式: {sample['format']}")
        print(f"总段落: {sample['total_segments']}")
        print(f"学习段落: {sample['learning_segments']}")
        
        print(f"\n📄 完整对话内容:")
        print("-" * 60)
        conversation = sample['conversation']
        print(conversation)
        
        print(f"\n🎯 Loss Mask详情:")
        print("-" * 60)
        segments = sample['data']['segments']
        
        for i, seg in enumerate(segments):
            learn_status = "✅学习" if seg['learn'] else "❌跳过"
            segment_type = seg['segment_type']
            weight = seg['weight']
            content = seg['content']
            
            print(f"\n段落 {i+1:2d}: [{segment_type}] {learn_status} (权重: {weight})")
            
            # 处理内容显示
            if '\n' in content:
                content_lines = content.split('\n')
                for j, line in enumerate(content_lines):
                    if line.strip():
                        print(f"     L{j+1}: {line}")
                    else:
                        print(f"     L{j+1}: (空行)")
            else:
                print(f"     内容: {content}")
        
        print("-" * 80)
    
    def display_analysis(self, analysis: Dict[str, Any]):
        """显示分析结果"""
        print(f"\n📊 数据分析报告")
        print("=" * 60)
        
        print(f"📈 基础统计:")
        print(f"  总样本数: {analysis['total_samples']}")
        print(f"  每样本段落数: {analysis['total_segments']}")
        print(f"  每样本学习段落: {analysis['learning_segments']}")
        print(f"  学习比例: {analysis['learning_ratio']:.2%}")
        
        print(f"\n📝 内容统计:")
        content = analysis['content_info']
        print(f"  中文长度: {content['chinese_length']} 字符")
        print(f"  英文长度: {content['english_length']} 字符")
        print(f"  摘要长度: {content['summary_length']} 字符")
        print(f"  完整对话长度: {content['conversation_length']} 字符")
        
        print(f"\n🔤 权重分布 (总计):")
        for weight_key, count in sorted(analysis['weight_distribution'].items()):
            print(f"  {weight_key}: {count} 个段落")
        
        print(f"\n📂 段落类型分布 (总计):")
        for seg_type, count in sorted(analysis['segment_types'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {seg_type}: {count} 个段落")
        
        print("=" * 60)
    
    def display_overview(self, samples: List[Dict[str, Any]]):
        """显示数据概览"""
        print(f"\n📖 数据概览")
        print("=" * 60)
        print(f"总样本数: {len(samples)}")
        
        if samples:
            sample = samples[0]
            print(f"数据格式: {sample['format']}")
            print(f"使用模板: {sample['template']}")
            print(f"主题: {sample['topic']}")
            print(f"每个样本段落数: {sample['total_segments']}")
            print(f"每个样本学习段落: {sample['learning_segments']}")
            
            # 显示内容预览
            segments = sample['data']['segments']
            chinese_seg = next((s for s in segments if s['segment_type'] == 'chinese_content'), None)
            english_seg = next((s for s in segments if s['segment_type'] == 'english_translation'), None)
            summary_seg = next((s for s in segments if s['segment_type'] == 'chinese_summary'), None)
            
            print(f"\n📝 内容预览:")
            if chinese_seg:
                print(f"  中文: {chinese_seg['content'][:50]}...")
            if english_seg:
                print(f"  英文: {english_seg['content'][:50]}...")
            if summary_seg:
                print(f"  摘要: {summary_seg['content']}")
        
        print("=" * 60)
    
    def save_data(self, samples: List[Dict[str, Any]], output_dir: str):
        """保存数据"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存parquet文件
        df = pd.DataFrame(samples)
        parquet_path = os.path.join(output_dir, "qwen25_short_novels_1280.parquet")
        df.to_parquet(parquet_path, index=False)
        
        # 保存分析
        analysis = self.analyze_data(samples)
        analysis_path = os.path.join(output_dir, "qwen25_short_analysis.json")
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)
        
        # 保存配置
        config = {
            "template": "qwen25",
            "system_prompt": self.system_prompt,
            "user_query": self.user_query,
            "special_tokens": ["<EN>", "</EN>", "<|im_start|>", "<|im_end|>"],
            "sample_count": len(samples),
            "content_type": "short_novels"
        }
        config_path = os.path.join(output_dir, "qwen25_short_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        return parquet_path, analysis_path, config_path


class DataDisplayer:
    """数据展示器"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.samples = self.load_data()
    
    def load_data(self) -> List[Dict[str, Any]]:
        """加载数据"""
        print(f"📖 加载数据: {self.data_path}")
        df = pd.read_parquet(self.data_path)
        samples = df.to_dict('records')
        print(f"✅ 加载完成: {len(samples)} 个样本")
        return samples
    
    def show_samples(self, start: int = 0, count: int = 3):
        """显示多个样本"""
        print(f"\n📖 样本展示 (显示 {start} 到 {start + count - 1})")
        print("=" * 80)
        
        for i in range(start, min(start + count, len(self.samples))):
            sample = self.samples[i]
            # print(f"\n🔍 样本 #{i}: {sample['id']}")
            # print(f"主题: {sample['topic']}")
            
            segments = sample['data']['segments']
            
            # 只显示重要段落
            important_types = ['chinese_content', 'special_token_start', 'english_translation', 
                             'special_token_end', 'chinese_summary']
            
            for seg in segments:
                if seg['segment_type'] in important_types:
                    learn_status = "✅" if seg['learn'] else "❌"
                    content = seg['content'].replace('\n', ' ').strip()
                    if len(content) > 60:
                        content = content[:60] + "..."
                    print(f"  [{seg['segment_type']}] {learn_status} W:{seg['weight']} - {content}")
            
            print("-" * 50)
    
    def search_content(self, keyword: str):
        """搜索内容"""
        print(f"\n🔍 搜索关键词: '{keyword}'")
        print("=" * 60)
        
        found = False
        for i, sample in enumerate(self.samples):
            segments = sample['data']['segments']
            for seg in segments:
                if keyword.lower() in seg['content'].lower():
                    if not found:
                        found = True
                    # print(f"样本 #{i} [{seg['segment_type']}]: {seg['content'][:100]}...")
                    break
        
        if not found:
            print(f"❌ 未找到包含 '{keyword}' 的内容")


def main():
    """主函数"""
    print("🚀 简化版Qwen2.5简短数据生成器")
    print("=" * 60)
    print("功能：生成1280个一致的Qwen2.5格式样本")
    print("特色：简化代码，详细展示")
    print("=" * 60)
    
    # 创建生成器
    generator = SimpleShortDataGenerator()
    
    # 生成数据
    samples = generator.generate_samples(1280)
    
    # 展示概览
    # generator.display_overview(samples)
    
    # 展示详细样本
    generator.display_sample_detail(samples[0], 0)
    
    # 分析数据
    analysis = generator.analyze_data(samples)
    generator.display_analysis(analysis)
    
    # 保存数据
    output_dir = "/home/jovyan/fdu_new/zyn/examples"
    parquet_path, analysis_path, config_path = generator.save_data(samples, output_dir)
    
    print(f"\n💾 数据保存完成")
    print("=" * 60)
    print(f"数据文件: {parquet_path}")
    print(f"分析报告: {analysis_path}")
    print(f"配置文件: {config_path}")
    
    # 演示数据展示器
    print(f"\n🔍 数据展示器演示")
    print("=" * 60)
    
    displayer = DataDisplayer(parquet_path)
    displayer.show_samples(0, 3)
    displayer.search_content("友谊")
    
    print(f"\n✅ 所有任务完成！")
    print("💡 数据结构: system -> user -> assistant(中文 -> <EN> -> 英文 -> </EN> -> 摘要)")
    print("🎯 所有1280个样本内容完全一致，适合快速测试")


if __name__ == "__main__":
    main()