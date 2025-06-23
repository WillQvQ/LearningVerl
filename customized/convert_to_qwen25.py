#!/usr/bin/env python3
"""
Qwen2.5 数据格式转换器（简化版）
将create_new_data.py生成的数据转换为Qwen2.5模板格式
修改：chinese_text 部分也参与 loss 计算
新增：special token 统计分析和数据过滤
"""

import os
import json
import pandas as pd
from typing import List, Dict, Any
from tqdm import tqdm
import re

class Qwen25DataConverter:
    """Qwen2.5数据格式转换器（简化版）"""
    
    def __init__(self):
        # Qwen2.5官方system prompt
        self.system_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        
        # 固定的用户query
        self.user_query = "请你帮我写一个 4000 字的小说，并翻译成英文"
        
        # Qwen2.5模板格式
        self.qwen_template = {
            "system_start": "<|im_start|>system\n",
            "system_end": "<|im_end|>\n",
            "user_start": "<|im_start|>user\n", 
            "user_end": "<|im_end|>\n",
            "assistant_start": "<|im_start|>assistant\n",
            "assistant_end": "<|im_end|>"
        }
        
        # 特殊token列表（用于统计和过滤）
        self.special_tokens = {
            "<|im_start|>": "qwen_template",
            "<|im_end|>": "qwen_template",
            "<EN>": "custom_special",
            "</EN>": "custom_special",
            "<ZH>": "custom_special",
            "</ZH>": "custom_special",
            "<NOVEL>": "custom_special", 
            "</NOVEL>": "custom_special",
            "<SUMMARY>": "custom_special",
            "</SUMMARY>": "custom_special"
        }
    
    def load_original_data(self, data_path: str) -> List[Dict[str, Any]]:
        """加载原始数据"""
        print(f"📖 加载原始数据: {data_path}")
        
        if data_path.endswith('.parquet'):
            df = pd.read_parquet(data_path)
            data = df.to_dict('records')
        elif data_path.endswith('.json'):
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            raise ValueError("支持的文件格式: .parquet 或 .json")
        
        print(f"✅ 加载完成，共 {len(data)} 个样本")
        return data
    
    def analyze_special_tokens_in_content(self, content: str) -> Dict[str, Any]:
        """分析内容中的特殊token"""
        token_stats = {}
        
        for token, token_type in self.special_tokens.items():
            count = content.count(token)
            if count > 0:
                if token_type not in token_stats:
                    token_stats[token_type] = {}
                token_stats[token_type][token] = count
        
        # 检查特殊token对是否平衡
        balance_check = {}
        token_pairs = [
            ("<EN>", "</EN>"),
            ("<ZH>", "</ZH>"),
            ("<NOVEL>", "</NOVEL>"),
            ("<SUMMARY>", "</SUMMARY>")
        ]
        
        for start_token, end_token in token_pairs:
            start_count = content.count(start_token)
            end_count = content.count(end_token)
            if start_count > 0 or end_count > 0:
                balance_check[f"{start_token}_{end_token}"] = {
                    "start_count": start_count,
                    "end_count": end_count,
                    "balanced": start_count == end_count,
                    "difference": abs(start_count - end_count)
                }
        
        return {
            "token_stats": token_stats,
            "balance_check": balance_check,
            "total_special_tokens": sum(
                sum(tokens.values()) for tokens in token_stats.values()
            )
        }
    
    def analyze_en_token_pairing(self, content: str, sample_id: str = "unknown") -> Dict[str, Any]:
        """详细分析<EN>和</EN>的成对情况"""
        en_start_count = content.count("<EN>")
        en_end_count = content.count("</EN>")
        
        # 基本统计
        pairing_analysis = {
            "sample_id": sample_id,
            "en_start_count": en_start_count,
            "en_end_count": en_end_count,
            "is_balanced": en_start_count == en_end_count,
            "difference": abs(en_start_count - en_end_count),
            "pairing_status": "unknown",
            "valid_pairs": 0,
            "orphaned_starts": 0,
            "orphaned_ends": 0
        }
        
        # 确定配对状态
        if en_start_count == 0 and en_end_count == 0:
            pairing_analysis["pairing_status"] = "no_tokens"
        elif en_start_count == 1 and en_end_count == 1:
            pairing_analysis["pairing_status"] = "perfect_pair"
            pairing_analysis["valid_pairs"] = 1
        elif en_start_count == en_end_count and en_start_count > 1:
            pairing_analysis["pairing_status"] = "multiple_balanced"
            pairing_analysis["valid_pairs"] = en_start_count
        elif en_start_count > en_end_count:
            pairing_analysis["pairing_status"] = "excess_starts"
            pairing_analysis["valid_pairs"] = en_end_count
            pairing_analysis["orphaned_starts"] = en_start_count - en_end_count
        elif en_end_count > en_start_count:
            pairing_analysis["pairing_status"] = "excess_ends"
            pairing_analysis["valid_pairs"] = en_start_count
            pairing_analysis["orphaned_ends"] = en_end_count - en_start_count
        else:
            pairing_analysis["pairing_status"] = "unknown_error"
        
        # 检查是否符合我们的筛选条件
        pairing_analysis["passes_filter"] = pairing_analysis["pairing_status"] in ["no_tokens", "perfect_pair"]
        
        return pairing_analysis
    
    def is_valid_sample(self, sample: Dict[str, Any]) -> bool:
        """检查样本是否有效（特殊token平衡且数量正确）"""
        # 提取助手回复
        assistant_response = self.extract_assistant_response(sample)
        
        # 分析<EN></EN>配对情况
        pairing_analysis = self.analyze_en_token_pairing(assistant_response, sample.get("id", "unknown"))
        
        # 只接受无token或完美配对的样本
        return pairing_analysis["passes_filter"]
    
    def extract_assistant_response(self, sample: Dict[str, Any]) -> str:
        """从原始样本中提取助手回复内容"""
        segments = sample["data"]["segments"]
        
        # 按顺序组合所有内容
        response_parts = []
        for segment in segments:
            content = segment["content"]
            response_parts.append(content)
        
        # 用换行符连接所有部分
        assistant_response = "\n\n".join(response_parts)
        return assistant_response
    
    def create_qwen25_conversation(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """创建Qwen2.5格式的对话"""
        
        # 提取助手回复
        assistant_response = self.extract_assistant_response(sample)
        
        # 分析助手回复中的特殊token
        special_token_analysis = self.analyze_special_tokens_in_content(assistant_response)
        
        # 分析<EN></EN>配对情况
        en_pairing_analysis = self.analyze_en_token_pairing(assistant_response, sample.get("id", "unknown"))
        
        # 构建完整的对话文本
        conversation = (
            f"{self.qwen_template['system_start']}"
            f"{self.system_prompt}"
            f"{self.qwen_template['system_end']}"
            f"{self.qwen_template['user_start']}"
            f"{self.user_query}"
            f"{self.qwen_template['user_end']}"
            f"{self.qwen_template['assistant_start']}"
            f"{assistant_response}"
            f"{self.qwen_template['assistant_end']}"
        )
        
        # 分析完整对话中的特殊token
        full_conversation_analysis = self.analyze_special_tokens_in_content(conversation)
        
        # 创建loss mask - 修改策略：chinese_text也计算loss
        segments = sample["data"]["segments"]
        loss_segments = []
        
        # 添加system部分 - 不计算loss
        system_text = (
            f"{self.qwen_template['system_start']}"
            f"{self.system_prompt}"
            f"{self.qwen_template['system_end']}"
        )
        loss_segments.append({
            "content": system_text,
            "learn": False,
            "weight": 0.0,
            "segment_type": "system_prompt",
            "segment_id": 0,
            "special_tokens": self.analyze_special_tokens_in_content(system_text)
        })
        
        # 添加user部分 - 不计算loss  
        user_text = (
            f"{self.qwen_template['user_start']}"
            f"{self.user_query}"
            f"{self.qwen_template['user_end']}"
        )
        loss_segments.append({
            "content": user_text,
            "learn": False,
            "weight": 0.0,
            "segment_type": "user_query",
            "segment_id": 1,
            "special_tokens": self.analyze_special_tokens_in_content(user_text)
        })
        
        # 添加assistant开始标签 - 不计算loss
        loss_segments.append({
            "content": self.qwen_template['assistant_start'],
            "learn": False,
            "weight": 0.0,
            "segment_type": "assistant_start_tag",
            "segment_id": 2,
            "special_tokens": self.analyze_special_tokens_in_content(self.qwen_template['assistant_start'])
        })
        
        # 添加assistant内容部分 - 修改：chinese_text也参与学习
        segment_id = 3
        for segment in segments:
            original_learn = segment["learn"]
            original_weight = segment["weight"]
            segment_type = segment["segment_type"]
            
            # 分析每个段落的特殊token
            segment_token_analysis = self.analyze_special_tokens_in_content(segment["content"])
            
            # 🔥 修改：如果是chinese_text类型，强制设置为学习状态
            if segment_type == "chinese_text":
                new_learn = True
                new_weight = 1.0 if original_weight == 0.0 else original_weight
            else:
                new_learn = original_learn
                new_weight = original_weight
            
            loss_segments.append({
                "content": segment["content"],
                "learn": new_learn,
                "weight": new_weight,
                "segment_type": segment_type,
                "segment_id": segment_id,
                "original_learn": original_learn,
                "original_weight": original_weight,
                "special_tokens": segment_token_analysis
            })
            segment_id += 1
        
        # 添加assistant结束标签 - 学习
        loss_segments.append({
            "content": self.qwen_template['assistant_end'],
            "learn": True,
            "weight": 1.0,
            "segment_type": "assistant_end_tag", 
            "segment_id": segment_id,
            "special_tokens": self.analyze_special_tokens_in_content(self.qwen_template['assistant_end'])
        })
        
        return {
            "format": "qwen25_structured",
            "conversation": conversation,
            "data": {"segments": loss_segments},
            "id": sample["id"],
            "topic": sample["topic"],
            "template": "qwen25",
            "total_segments": len(loss_segments),
            "learning_segments": len([s for s in loss_segments if s["learn"]]),
            "original_segments": sample["total_segments"],
            "chinese_text_modified": True,
            "special_token_analysis": {
                "assistant_response": special_token_analysis,
                "full_conversation": full_conversation_analysis
            },
            "en_pairing_analysis": en_pairing_analysis  # 新增：EN配对分析
        }
    
    def convert_batch(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """批量转换样本（包含数据过滤）"""
        print(f"🔄 开始转换 {len(samples)} 个样本为Qwen2.5格式...")
        print("💡 新策略：chinese_text 部分也参与 loss 计算")
        print("🔍 新功能：过滤不合格的特殊token样本")
        
        # 首先统计所有样本的<EN></EN>配对情况
        print("🔍 第一步：统计<EN></EN>配对情况...")
        pairing_statistics = {
            "no_tokens": 0,
            "perfect_pair": 0,
            "multiple_balanced": 0,
            "excess_starts": 0,
            "excess_ends": 0,
            "unknown_error": 0
        }
        
        all_pairing_analyses = []
        for sample in tqdm(samples, desc="配对分析"):
            assistant_response = self.extract_assistant_response(sample)
            pairing_analysis = self.analyze_en_token_pairing(assistant_response, sample.get("id", "unknown"))
            all_pairing_analyses.append(pairing_analysis)
            pairing_statistics[pairing_analysis["pairing_status"]] += 1
        
        # 显示配对统计
        print(f"\n📊 <EN></EN>配对统计:")
        print(f"  无token样本: {pairing_statistics['no_tokens']} ({pairing_statistics['no_tokens']/len(samples)*100:.1f}%)")
        print(f"  完美配对(1对): {pairing_statistics['perfect_pair']} ({pairing_statistics['perfect_pair']/len(samples)*100:.1f}%)")
        print(f"  多对平衡: {pairing_statistics['multiple_balanced']} ({pairing_statistics['multiple_balanced']/len(samples)*100:.1f}%)")
        print(f"  开始标签过多: {pairing_statistics['excess_starts']} ({pairing_statistics['excess_starts']/len(samples)*100:.1f}%)")
        print(f"  结束标签过多: {pairing_statistics['excess_ends']} ({pairing_statistics['excess_ends']/len(samples)*100:.1f}%)")
        print(f"  未知错误: {pairing_statistics['unknown_error']} ({pairing_statistics['unknown_error']/len(samples)*100:.1f}%)")
        
        # 计算通过筛选的样本数量
        valid_count = pairing_statistics['no_tokens'] + pairing_statistics['perfect_pair']
        invalid_count = len(samples) - valid_count
        print(f"\n✅ 筛选结果: {valid_count}/{len(samples)} 个样本通过筛选, 将删除 {invalid_count} 个样本")
        
        # 过滤有效样本
        valid_samples = []
        print("🔍 第二步：过滤无效样本...")
        for sample in tqdm(samples, desc="过滤进度"):
            if self.is_valid_sample(sample):
                valid_samples.append(sample)
        
        print(f"✅ 过滤完成: 有效样本 {len(valid_samples)}/{len(samples)}")
        
        # 转换有效样本
        converted_samples = []
        chinese_text_modifications = 0
        
        print("🔄 第三步：转换有效样本...")
        for sample in tqdm(valid_samples, desc="转换进度"):
            try:
                qwen_sample = self.create_qwen25_conversation(sample)
                converted_samples.append(qwen_sample)
                
                # 统计chinese_text修改次数
                if qwen_sample.get("chinese_text_modified", False):
                    chinese_text_modifications += 1
                    
            except Exception as e:
                print(f"⚠️  转换样本 {sample.get('id', 'unknown')} 时出错: {e}")
                continue
        
        print(f"✅ 转换完成，共 {len(converted_samples)} 个样本")
        print(f"🔄 chinese_text 修改次数: {chinese_text_modifications}")
        
        # 将配对统计信息附加到结果中
        return converted_samples, pairing_statistics
    
    def analyze_converted_data(self, samples: List[Dict[str, Any]], pairing_stats: Dict[str, int] = None) -> Dict[str, Any]:
        """分析转换后的数据"""
        
        analysis = {
            "total_samples": len(samples),
            "total_segments": 0,
            "learning_segments": 0,
            "segment_types": {},
            "weight_distribution": {},
            "template_overhead": 0,
            "content_segments": 0,
            "chinese_text_stats": {
                "total_chinese_text_segments": 0,
                "learning_chinese_text_segments": 0,
                "modified_chinese_text_segments": 0
            },
            "special_token_stats": {
                "samples_with_tokens": 0,
                "total_token_instances": 0,
                "token_type_distribution": {},
                "specific_token_counts": {},
                "balance_issues": {
                    "total_balance_issues": 0,
                    "issue_details": {}
                },
                "segments_with_tokens": 0
            },
            "en_pairing_stats": {  # 新增：EN配对统计
                "converted_samples_pairing": {
                    "no_tokens": 0,
                    "perfect_pair": 0,
                    "multiple_balanced": 0,
                    "excess_starts": 0,
                    "excess_ends": 0,
                    "unknown_error": 0
                },
                "original_samples_pairing": pairing_stats if pairing_stats else {},
                "filtering_efficiency": 0.0
            }
        }
        
        # 统计转换后样本的EN配对情况
        for sample in samples:
            segments = sample["data"]["segments"]
            analysis["total_segments"] += len(segments)
            
            # 统计转换后样本的EN配对状态
            en_pairing = sample.get("en_pairing_analysis", {})
            pairing_status = en_pairing.get("pairing_status", "unknown_error")
            if pairing_status in analysis["en_pairing_stats"]["converted_samples_pairing"]:
                analysis["en_pairing_stats"]["converted_samples_pairing"][pairing_status] += 1
            
            # 分析特殊token
            special_analysis = sample.get("special_token_analysis", {})
            assistant_analysis = special_analysis.get("assistant_response", {})
            
            # 统计包含特殊token的样本
            if assistant_analysis.get("total_special_tokens", 0) > 0:
                analysis["special_token_stats"]["samples_with_tokens"] += 1
                analysis["special_token_stats"]["total_token_instances"] += assistant_analysis["total_special_tokens"]
                
                # 统计token类型分布
                token_stats = assistant_analysis.get("token_stats", {})
                for token_type, tokens in token_stats.items():
                    if token_type not in analysis["special_token_stats"]["token_type_distribution"]:
                        analysis["special_token_stats"]["token_type_distribution"][token_type] = 0
                    
                    for token, count in tokens.items():
                        analysis["special_token_stats"]["token_type_distribution"][token_type] += count
                        
                        if token not in analysis["special_token_stats"]["specific_token_counts"]:
                            analysis["special_token_stats"]["specific_token_counts"][token] = 0
                        analysis["special_token_stats"]["specific_token_counts"][token] += count
                
                # 统计平衡问题
                balance_check = assistant_analysis.get("balance_check", {})
                for pair_name, pair_info in balance_check.items():
                    if not pair_info.get("balanced", True):
                        analysis["special_token_stats"]["balance_issues"]["total_balance_issues"] += 1
                        if pair_name not in analysis["special_token_stats"]["balance_issues"]["issue_details"]:
                            analysis["special_token_stats"]["balance_issues"]["issue_details"][pair_name] = 0
                        analysis["special_token_stats"]["balance_issues"]["issue_details"][pair_name] += 1
            
            for segment in segments:
                seg_type = segment["segment_type"]
                analysis["segment_types"][seg_type] = analysis["segment_types"].get(seg_type, 0) + 1
                
                weight = segment["weight"]
                weight_key = f"weight_{weight}"
                analysis["weight_distribution"][weight_key] = analysis["weight_distribution"].get(weight_key, 0) + 1
                
                if segment["learn"]:
                    analysis["learning_segments"] += 1
                    analysis["content_segments"] += 1
                else:
                    if seg_type in ["system_prompt", "user_query", "assistant_start_tag"]:
                        analysis["template_overhead"] += 1
                
                # 统计chinese_text
                if seg_type == "chinese_text":
                    analysis["chinese_text_stats"]["total_chinese_text_segments"] += 1
                    if segment["learn"]:
                        analysis["chinese_text_stats"]["learning_chinese_text_segments"] += 1
                    if 'original_learn' in segment and segment['original_learn'] != segment['learn']:
                        analysis["chinese_text_stats"]["modified_chinese_text_segments"] += 1
                
                # 统计段落级别的特殊token
                if 'special_tokens' in segment and segment['special_tokens'] is not None:
                    tokens = segment['special_tokens']
                    if isinstance(tokens, dict) and tokens.get('total_special_tokens', 0) > 0:
                        analysis["special_token_stats"]["segments_with_tokens"] += 1
        
        # 计算比例
        analysis["learning_ratio"] = analysis["learning_segments"] / analysis["total_segments"] if analysis["total_segments"] > 0 else 0
        analysis["template_ratio"] = analysis["template_overhead"] / analysis["total_segments"] if analysis["total_segments"] > 0 else 0
        
        # chinese_text统计比例
        chinese_stats = analysis["chinese_text_stats"]
        if chinese_stats["total_chinese_text_segments"] > 0:
            chinese_stats["learning_ratio"] = chinese_stats["learning_chinese_text_segments"] / chinese_stats["total_chinese_text_segments"]
            chinese_stats["modification_ratio"] = chinese_stats["modified_chinese_text_segments"] / chinese_stats["total_chinese_text_segments"]
        else:
            chinese_stats["learning_ratio"] = 0
            chinese_stats["modification_ratio"] = 0
        
        # 特殊token统计比例
        special_stats = analysis["special_token_stats"]
        if analysis["total_samples"] > 0:
            special_stats["token_coverage_ratio"] = special_stats["samples_with_tokens"] / analysis["total_samples"]
        else:
            special_stats["token_coverage_ratio"] = 0
            
        if analysis["total_segments"] > 0:
            special_stats["segment_token_ratio"] = special_stats["segments_with_tokens"] / analysis["total_segments"]
        else:
            special_stats["segment_token_ratio"] = 0
        
        # 计算过滤效率
        if pairing_stats:
            original_total = sum(pairing_stats.values())
            if original_total > 0:
                analysis["en_pairing_stats"]["filtering_efficiency"] = len(samples) / original_total
        
        return analysis
    
    def save_converted_data(self, samples: List[Dict[str, Any]], output_dir: str):
        """保存转换后的数据"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 转换为DataFrame
        df = pd.DataFrame(samples)
        
        # 按90/10比例分割
        train_size = int(0.8 * len(df))
        train_df = df[:train_size]
        test_df = df[train_size:]
        
        # 保存parquet文件
        train_path = os.path.join(output_dir, "qwen25_train.parquet")
        test_path = os.path.join(output_dir, "qwen25_test.parquet")
        
        train_df.to_parquet(train_path, index=False)
        test_df.to_parquet(test_path, index=False)
        
        # 保存转换配置
        config = {
            "template": "qwen25",
            "system_prompt": self.system_prompt,
            "user_query": self.user_query,
            "template_tokens": self.qwen_template,
            "special_tokens": list(self.special_tokens.keys()),
            "conversion_info": {
                "original_format": "structured",
                "target_format": "qwen25_structured",
                "loss_strategy": "modified_assistant_only",
                "description": "System prompt, user query和template tokens不计算loss，assistant内容部分(包括chinese_text)按权重计算loss",
                "chinese_text_modification": {
                    "enabled": True,
                    "description": "chinese_text段落强制设置为学习状态，权重设为1.0(如果原来是0.0)",
                    "reason": "确保中文内容也参与模型训练"
                },
                "special_token_analysis": {
                    "enabled": True,
                    "description": "分析并统计所有特殊token的使用情况",
                    "tracked_tokens": self.special_tokens
                },
                "data_filtering": {
                    "enabled": True,
                    "description": "过滤特殊token不平衡或数量异常的样本",
                    "rules": {
                        "<EN>_</EN>": "只接受恰好1对或0对的样本",
                        "other_pairs": "检查平衡性"
                    }
                },
                "en_pairing_analysis": {  # 新增：EN配对分析配置
                    "enabled": True,
                    "description": "详细分析每个样本中<EN>和</EN>的配对情况",
                    "filtering_criteria": "只保留无token或恰好1对EN token的样本"
                }
            }
        }
        
        config_path = os.path.join(output_dir, "qwen25_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        # 保存数据分析
        analysis = self.analyze_converted_data(samples)
        analysis_path = os.path.join(output_dir, "qwen25_analysis.json")
        
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)
        
        return train_path, test_path, config_path, analysis_path


def main():
    """主函数"""
    print("🚀 Qwen2.5 数据格式转换器（简化版）")
    print("=" * 60)
    print("功能：将原始小说数据转换为Qwen2.5模板格式")
    print("🔥 特性：chinese_text 部分参与 loss 计算")
    print("🎯 新功能：特殊Token统计分析和数据过滤")
    print("🔍 重点功能：<EN></EN>配对详细统计")
    print("=" * 60)
    
    # 输入输出路径
    input_dir = "/home/jovyan/fdu_new/zyn/examples/data/story"
    output_dir = "/home/jovyan/fdu_new/zyn/examples/data/qwen25_format"
    
    # 创建转换器
    converter = Qwen25DataConverter()
    
    # 存储所有转换后的样本用于最终分析
    all_converted_samples = []
    all_pairing_stats = {
        "no_tokens": 0,
        "perfect_pair": 0,
        "multiple_balanced": 0,
        "excess_starts": 0,
        "excess_ends": 0,
        "unknown_error": 0
    }
    
    # 处理训练集和测试集
    for dataset_type in ["train", "test"]:
        print(f"\n📂 处理 {dataset_type} 数据集")
        print("-" * 40)
        
        # 加载原始数据
        input_path = os.path.join(input_dir, f"{dataset_type}.parquet")
        if not os.path.exists(input_path):
            print(f"⚠️  文件不存在: {input_path}")
            continue
            
        original_samples = converter.load_original_data(input_path)
        
        # 转换数据（包含过滤和配对统计）
        converted_samples, pairing_stats = converter.convert_batch(original_samples)
        all_converted_samples.extend(converted_samples)
        
        # 累加配对统计
        for key in all_pairing_stats:
            all_pairing_stats[key] += pairing_stats[key]
        
        # 保存转换结果
        output_path = os.path.join(output_dir, f"qwen25_{dataset_type}.parquet")
        os.makedirs(output_dir, exist_ok=True)
        
        df = pd.DataFrame(converted_samples)
        df.to_parquet(output_path, index=False)
        print(f"✅ 保存完成: {output_path}")
    
    if all_converted_samples:
        # 保存配置和分析
        train_path, test_path, config_path, analysis_path = converter.save_converted_data(
            all_converted_samples, output_dir
        )
        
        # 显示简要统计（包含配对统计）
        analysis = converter.analyze_converted_data(all_converted_samples, all_pairing_stats)
        
        print(f"\n📊 转换完成统计:")
        print("=" * 60)
        print(f"总样本数: {analysis['total_samples']}")
        print(f"学习段落比例: {analysis['learning_ratio']:.2%}")
        print(f"chinese_text学习比例: {analysis['chinese_text_stats']['learning_ratio']:.2%}")
        print(f"特殊token覆盖率: {analysis['special_token_stats']['token_coverage_ratio']:.2%}")
        print(f"平衡问题数量: {analysis['special_token_stats']['balance_issues']['total_balance_issues']}")
        
        # 显示EN配对统计
        print(f"\n🔍 <EN></EN>配对最终统计:")
        print("-" * 40)
        original_pairing = analysis["en_pairing_stats"]["original_samples_pairing"]
        converted_pairing = analysis["en_pairing_stats"]["converted_samples_pairing"]
        
        if original_pairing:
            original_total = sum(original_pairing.values())
            print(f"📈 原始数据配对分布:")
            for status, count in original_pairing.items():
                percentage = count / original_total * 100 if original_total > 0 else 0
                print(f"  {status}: {count} ({percentage:.1f}%)")
            
            print(f"\n📊 转换后数据配对分布:")
            converted_total = sum(converted_pairing.values())
            for status, count in converted_pairing.items():
                percentage = count / converted_total * 100 if converted_total > 0 else 0
                print(f"  {status}: {count} ({percentage:.1f}%)")
            
            filtering_efficiency = analysis["en_pairing_stats"]["filtering_efficiency"]
            print(f"\n✅ 过滤效率: {filtering_efficiency:.2%} ({converted_total}/{original_total})")
        
        print(f"\n📁 输出文件:")
        print(f"训练集: {train_path}")
        print(f"测试集: {test_path}")
        print(f"配置文件: {config_path}")
        print(f"分析文件: {analysis_path}")
        print("=" * 60)


if __name__ == "__main__":
    main()