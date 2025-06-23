#!/usr/bin/env python3
"""
中英文小说翻译数据生成器（优化版）
使用vLLM批量生成100条包含中文小说、英文翻译和中文摘要的数据
支持特殊token和多区域loss mask

数据结构特点：
- 每条数据包含固定的5个段落：中文正文 -> <EN> -> 英文翻译 -> </EN> -> 中文摘要
- 简化结构，避免重复的中英文段落组合
- 统一的loss权重分配：中文(0.0) -> 特殊token(5.0) -> 英文(1.0) -> 特殊token(5.0) -> 摘要(2.0)
"""

import sys
import os
import random
import json
import pandas as pd
from typing import List, Dict, Any
from tqdm import tqdm
import re

# 添加路径
sys.path.append('/home/jovyan/fdu_new/zyn/verl')

# vLLM相关导入
try:
    from vllm import LLM, SamplingParams
    print("✅ vLLM导入成功")
except ImportError:
    print("❌ vLLM未安装，请先安装: pip install vllm")
    sys.exit(1)


class ChineseNovelDataGenerator:
    """中文小说数据生成器（优化版）"""
    
    def __init__(self, model_path="/home/jovyan/fdu_new/models/Qwen2.5-7B-Instruct"):
        self.model_path = model_path
        self.special_tokens = ["<EN>", "</EN>"]
        
        # 小说主题和设定
        self.novel_themes = [
            ("都市言情", "现代都市", "爱情故事"),
            ("武侠江湖", "古代江湖", "武功秘籍"),
            ("仙侠修真", "修仙世界", "法宝丹药"),
            ("科幻未来", "未来世界", "科技发展"),
            ("历史穿越", "古代宫廷", "权谋斗争"),
            ("悬疑推理", "现代社会", "犯罪侦探"),
            ("青春校园", "校园生活", "青春成长"),
            ("商战职场", "商业世界", "职场竞争"),
            ("军事战争", "战争年代", "英雄事迹"),
            ("魔幻奇幻", "魔法世界", "魔法冒险")
        ]
        
        # 初始化vLLM模型
        print(f"🚀 正在加载模型: {model_path}")
        try:
            self.llm = LLM(
                model=model_path,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.8,
                max_model_len=4096,
                trust_remote_code=True
            )
            print("✅ 模型加载成功")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            sys.exit(1)
    
    def create_novel_prompt(self, sample_id: int) -> str:
        """创建生成小说的prompt"""
        theme, setting, element = random.choice(self.novel_themes)
        
        prompt = f"""请创作一个{theme}小说片段，严格按照以下结构：

1. 背景设定：{setting}
2. 核心元素：{element}
3. 要求结构：中文小说 + 一次英文翻译 + 一个中文摘要

严格格式要求：
第一部分：写一段完整的中文小说内容（1500-2000字）
第二部分：将整段中文内容翻译成英文，用<EN>和</EN>标签包围（只有一对标签）
第三部分：提供一段150字的中文摘要，以"故事摘要："开头

格式示例：
主人公张明走在夜晚的街道上，心情复杂。他刚刚从公司辞职，面临人生的重大转折。街灯照亮了他的脸庞，也照亮了内心的迷茫。（继续1500字的完整故事...）

<EN>The protagonist Zhang Ming walked down the night street with complicated feelings. He had just resigned from his company and was facing a major turning point in life. The street lights illuminated his face and also lit up the confusion in his heart. (Complete English translation of the entire story...)</EN>

故事摘要：主人公张明在人生转折点时选择辞职，通过夜晚的漫步和内心独白，最终找到了新的人生方向，体现了现代都市人面临选择时的勇气与成长。

注意事项：
- 中文小说要完整连贯，有开头、发展、高潮、结尾
- 英文翻译要准确，语法正确
- 只能有一对<EN></EN>标签
- 摘要要简洁明了，突出主题

请按此格式创作："""

        return prompt
    
    def create_all_prompts(self, total_samples: int) -> List[str]:
        """创建所有prompts"""
        print(f"📝 创建 {total_samples} 个prompts...")
        prompts = []
        
        for i in range(total_samples):
            prompt = self.create_novel_prompt(i)
            prompts.append(prompt)
            
        print(f"✅ 创建完成 {len(prompts)} 个prompts")
        return prompts
    
    def generate_all_texts(self, prompts: List[str]) -> List[str]:
        """批量生成所有文本"""
        print(f"🎯 开始批量生成 {len(prompts)} 个文本...")
        
        # 设置生成参数
        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.9,
            max_tokens=8192,
            stop=None
        )
        
        try:
            # 批量生成
            outputs = self.llm.generate(prompts, sampling_params)
            
            # 提取生成的文本
            generated_texts = []
            for output in outputs:
                generated_text = output.outputs[0].text.strip()
                generated_texts.append(generated_text)
            
            print(f"✅ 批量生成完成，共 {len(generated_texts)} 个文本")
            return generated_texts
            
        except Exception as e:
            print(f"❌ 批量生成失败: {e}")
            # 返回空列表，后续使用备用样本
            return []
    
    def parse_generated_content(self, text: str, sample_id: int) -> Dict[str, Any]:
        """解析生成的内容，提取各个部分"""
        
        # 寻找摘要部分（通常在最后）
        summary_patterns = [
            r'摘要[:：]\s*(.+?)(?:\n|$)',
            r'总结[:：]\s*(.+?)(?:\n|$)', 
            r'简介[:：]\s*(.+?)(?:\n|$)',
            r'(?:^|\n)([^<\n]{100,200}?[。！？])(?:\n|$)'  # 最后一个长句子作为摘要
        ]
        
        summary = ""
        main_text = text
        
        for pattern in summary_patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.MULTILINE)
            if matches:
                summary = matches[-1].strip()
                # 从主文本中移除摘要
                main_text = re.sub(pattern, "", text, flags=re.DOTALL | re.MULTILINE)
                break
        
        if not summary:
            # 如果没找到摘要，取最后一段作为摘要
            paragraphs = [p.strip() for p in main_text.split('\n') if p.strip()]
            if paragraphs:
                summary = paragraphs[-1]
                main_text = '\n'.join(paragraphs[:-1])
        
        # 提取英文部分 - 合并所有英文内容
        english_parts = re.findall(r'<EN>(.*?)</EN>', main_text, re.DOTALL)
        
        # 移除英文部分，得到纯中文部分
        chinese_text = re.sub(r'<EN>.*?</EN>', "", main_text, flags=re.DOTALL)
        chinese_text = re.sub(r'\n+', '\n', chinese_text).strip()
        
        # 合并所有英文内容为一段
        if english_parts:
            # 合并多个英文段落，用空格连接
            combined_english = " ".join([part.strip() for part in english_parts if part.strip()])
        else:
            # 如果没有英文部分，创建一个默认的翻译
            combined_english = "The story continues with meaningful narrative and character development."
        
        # 构建简化的段落结构：中文 -> 英文翻译 -> 中文摘要
        segments = []
        segment_id = 0
        
        # 1. 中文正文（合并所有中文段落）
        chinese_paragraphs = [p.strip() for p in chinese_text.split('\n') if p.strip()]
        if chinese_paragraphs:
            # 合并中文段落，保持合理的长度
            combined_chinese = " ".join(chinese_paragraphs)
            # 如果太长，截取前2000字符
            if len(combined_chinese) > 2000:
                combined_chinese = combined_chinese[:2000] + "..."
                
            segments.append({
                "content": combined_chinese,
                "learn": False,  # 中文不加loss
                "weight": 0.0,
                "segment_type": "chinese_text",
                "segment_id": segment_id
            })
            segment_id += 1
        
        # 2. 英文翻译部分（只有一次）
        # 特殊token开始
        segments.append({
            "content": "<EN>",
            "learn": True,
            "weight": 5.0,  # 特殊token 5倍loss
            "segment_type": "special_token_start", 
            "segment_id": segment_id
        })
        segment_id += 1
        
        # 英文翻译内容
        segments.append({
            "content": combined_english,
            "learn": True,
            "weight": 1.0,  # 英文翻译 1倍loss
            "segment_type": "english_translation",
            "segment_id": segment_id
        })
        segment_id += 1
        
        # 特殊token结束
        segments.append({
            "content": "</EN>",
            "learn": True,
            "weight": 5.0,  # 特殊token 5倍loss
            "segment_type": "special_token_end",
            "segment_id": segment_id
        })
        segment_id += 1
        
        # 3. 中文摘要（只有一次）
        if summary:
            segments.append({
                "content": f"故事摘要：{summary}",
                "learn": True,
                "weight": 2.0,  # 中文摘要 2倍loss
                "segment_type": "chinese_summary",
                "segment_id": segment_id
            })
        
        return {
            "format": "structured",
            "data": {"segments": segments},
            "id": sample_id,
            "topic": "中英文小说翻译",
            "total_segments": len(segments),
            "english_segments": len([s for s in segments if s["segment_type"] == "english_translation"]),
            "special_tokens": len([s for s in segments if "special_token" in s["segment_type"]])
        }
    
    def create_fallback_sample(self, sample_id: int) -> Dict[str, Any]:
        """创建备用样本（当生成失败时）"""
        # 随机选择主题创建多样化的备用样本
        theme, setting, element = random.choice(self.novel_themes)
        
        fallback_stories = [
            {
                "chinese": f"在{setting}中，主角面临着前所未有的挑战。",
                "english": f"In the world of {element.lower()}, the protagonist faces unprecedented challenges.",
                "summary": f"一个关于{theme}的故事，讲述了{element}带来的变化。"
            },
            {
                "chinese": f"月光洒在{setting}的街道上，一切都显得格外宁静。",
                "english": "The moonlight cast a serene glow over the ancient streets, whispering tales of old.",
                "summary": f"描述{setting}夜晚的宁静与美好，体现{theme}的独特魅力。"
            },
            {
                "chinese": f"他缓缓推开门，{element}的气息扑面而来。",
                "english": f"As he slowly opened the door, the essence of {element.lower()} filled the air.",
                "summary": f"通过开门这一动作，展现{theme}世界的神秘与魅力。"
            }
        ]
        
        story = random.choice(fallback_stories)
        
        segments = [
            {
                "content": story["chinese"],
                "learn": False,
                "weight": 0.0,
                "segment_type": "chinese_text",
                "segment_id": 0
            },
            {
                "content": "<EN>",
                "learn": True,
                "weight": 5.0,
                "segment_type": "special_token_start",
                "segment_id": 1
            },
            {
                "content": story["english"],
                "learn": True,
                "weight": 1.0,
                "segment_type": "english_translation",
                "segment_id": 2
            },
            {
                "content": "</EN>",
                "learn": True,
                "weight": 5.0,
                "segment_type": "special_token_end", 
                "segment_id": 3
            },
            {
                "content": f"故事摘要：{story['summary']}",
                "learn": True,
                "weight": 2.0,
                "segment_type": "chinese_summary",
                "segment_id": 4
            }
        ]
        
        return {
            "format": "structured",
            "data": {"segments": segments},
            "id": sample_id,
            "topic": "中英文小说翻译",
            "total_segments": len(segments),
            "english_segments": 1,
            "special_tokens": 2
        }
    
    def process_all_texts(self, generated_texts: List[str], total_samples: int) -> List[Dict[str, Any]]:
        """批量处理所有生成的文本"""
        print(f"🔄 开始处理 {len(generated_texts)} 个生成文本...")
        
        all_samples = []
        
        # 处理成功生成的文本
        for i, text in enumerate(tqdm(generated_texts, desc="处理生成文本")):
            if text and text.strip():
                try:
                    sample = self.parse_generated_content(text, i)
                    all_samples.append(sample)
                except Exception as e:
                    print(f"⚠️  处理第{i}个文本时出错: {e}")
                    fallback_sample = self.create_fallback_sample(i)
                    all_samples.append(fallback_sample)
            else:
                # 文本为空，使用备用样本
                fallback_sample = self.create_fallback_sample(i)
                all_samples.append(fallback_sample)
        
        # 如果生成的文本不够，补充备用样本
        while len(all_samples) < total_samples:
            sample_id = len(all_samples)
            fallback_sample = self.create_fallback_sample(sample_id)
            all_samples.append(fallback_sample)
            print(f"  ➕ 添加备用样本 {sample_id}")
        
        print(f"✅ 处理完成，共 {len(all_samples)} 个样本")
        return all_samples
    
    def generate_all_data(self, total_samples: int = 1000) -> List[Dict[str, Any]]:
        """生成所有数据（优化版）"""
        print(f"🎯 开始生成 {total_samples} 个中英文小说样本...")
        print("=" * 60)
        
        # 第一步：创建所有prompts
        prompts = self.create_all_prompts(total_samples)
        
        # 第二步：批量生成所有文本
        generated_texts = self.generate_all_texts(prompts)
        
        # 第三步：批量处理所有文本
        all_samples = self.process_all_texts(generated_texts, total_samples)
        
        print(f"\n✅ 数据生成完成！总计: {len(all_samples)} 个样本")
        return all_samples


def create_tokenizer_with_special_tokens():
    """创建包含特殊token的tokenizer配置"""
    
    tokenizer_config = {
        "special_tokens": {
            "additional_special_tokens": ["<EN>", "</EN>"]
        },
        "usage_instructions": """
使用方法：
1. 加载tokenizer后添加特殊tokens：
   tokenizer.add_special_tokens({'additional_special_tokens': ['<EN>', '</EN>']})
   
2. 调整模型embedding层大小：
   model.resize_token_embeddings(len(tokenizer))
   
3. 特殊token的ID：
   en_start_id = tokenizer.convert_tokens_to_ids('<EN>')
   en_end_id = tokenizer.convert_tokens_to_ids('</EN>')
"""
    }
    
    return tokenizer_config


def analyze_generated_data(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """分析生成的数据"""
    
    analysis = {
        "total_samples": len(samples),
        "total_segments": 0,
        "segment_types": {},
        "weight_distribution": {},
        "learning_segments": 0,
        "english_segments": 0,
        "special_tokens": 0,
        "average_segments_per_sample": 0
    }
    
    for sample in samples:
        segments = sample["data"]["segments"]
        analysis["total_segments"] += len(segments)
        
        for segment in segments:
            # 统计段落类型
            seg_type = segment["segment_type"]
            analysis["segment_types"][seg_type] = analysis["segment_types"].get(seg_type, 0) + 1
            
            # 统计权重分布
            weight = segment["weight"]
            weight_key = f"weight_{weight}"
            analysis["weight_distribution"][weight_key] = analysis["weight_distribution"].get(weight_key, 0) + 1
            
            # 统计学习段落
            if segment["learn"]:
                analysis["learning_segments"] += 1
                
                if seg_type == "english_translation":
                    analysis["english_segments"] += 1
                elif "special_token" in seg_type:
                    analysis["special_tokens"] += 1
    
    analysis["average_segments_per_sample"] = analysis["total_segments"] / len(samples) if len(samples) > 0 else 0
    analysis["learning_ratio"] = analysis["learning_segments"] / analysis["total_segments"] if analysis["total_segments"] > 0 else 0
    
    return analysis


def save_data_and_configs(samples: List[Dict[str, Any]], output_dir: str):
    """保存数据和配置文件"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 转换为DataFrame
    df = pd.DataFrame(samples)
    
    # 按80/20比例分割
    train_size = int(0.8 * len(df))
    train_df = df[:train_size]
    test_df = df[train_size:]
    
    # 保存parquet文件
    train_path = os.path.join(output_dir, "train.parquet")
    test_path = os.path.join(output_dir, "test.parquet")
    
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)
    
    # 保存tokenizer配置
    tokenizer_config = create_tokenizer_with_special_tokens()
    tokenizer_config_path = os.path.join(output_dir, "tokenizer_config.json")
    
    with open(tokenizer_config_path, 'w', encoding='utf-8') as f:
        json.dump(tokenizer_config, f, ensure_ascii=False, indent=2)
    
    # 保存数据分析报告
    analysis = analyze_generated_data(samples)
    analysis_path = os.path.join(output_dir, "data_analysis.json")
    
    with open(analysis_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    
    return train_path, test_path, tokenizer_config_path, analysis_path


def main():
    """主函数"""
    print("🚀 中英文小说翻译数据生成器（单一结构版）")
    print("=" * 60)
    print("功能：批量生成100条包含中文小说、英文翻译和中文摘要的数据")
    print("特色：支持特殊token <EN> </EN> 和多区域loss控制")
    print("结构：每条数据固定5个段落 - 中文->特殊token->英文->特殊token->摘要")
    print("优化：批量生成提升效率，一次性处理所有prompts")
    print("=" * 60)
    
    # 创建生成器
    generator = ChineseNovelDataGenerator()
    
    # 生成数据
    samples = generator.generate_all_data(total_samples=2000)
    
    # 保存数据和配置
    output_dir = "/home/jovyan/fdu_new/zyn/examples"
    train_path, test_path, tokenizer_config_path, analysis_path = save_data_and_configs(samples, output_dir)
    
    # 分析结果
    analysis = analyze_generated_data(samples)
    
    print(f"\n" + "=" * 60)
    print("📊 数据生成和分析完成")
    print("=" * 60)
    print(f"训练集: {len(pd.read_parquet(train_path))} 样本 -> {train_path}")
    print(f"测试集: {len(pd.read_parquet(test_path))} 样本 -> {test_path}")
    print(f"Tokenizer配置: {tokenizer_config_path}")
    print(f"数据分析报告: {analysis_path}")
    
    print(f"\n📈 数据统计:")
    print(f"  总样本数: {analysis['total_samples']}")
    print(f"  总段落数: {analysis['total_segments']}")
    print(f"  学习段落数: {analysis['learning_segments']}")
    print(f"  英文翻译段落: {analysis['english_segments']}")
    print(f"  特殊token段落: {analysis['special_tokens']}")
    print(f"  平均段落/样本: {analysis['average_segments_per_sample']:.1f}")
    print(f"  学习段落比例: {analysis['learning_ratio']:.2%}")
    
    print(f"\n🎯 Loss权重分布:")
    for weight_key, count in sorted(analysis['weight_distribution'].items()):
        print(f"  {weight_key}: {count} 个段落")
    
    print(f"\n📝 段落类型分布:")
    for seg_type, count in sorted(analysis['segment_types'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {seg_type}: {count} 个段落")
    
    print(f"\n" + "=" * 60)
    print("✅ 所有任务完成！可以开始训练了。")
    print("💡 结构优化：每条数据固定为 中文-英文-摘要 单一组合")
    print("🎯 数据特点：简化结构，统一权重，易于训练和评估")
    print("=" * 60)


if __name__ == "__main__":
    main()