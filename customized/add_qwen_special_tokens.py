#!/usr/bin/env python3
"""
为 Qwen2.5-3B-Instruct 模型添加 special tokens '<EN>' 和 '</EN>'
基于 LearningVerl issue #9 的实施方案
"""

import json
import os
import shutil
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def check_model_status(model_path):
    """检查 Qwen 模型的词汇表状态"""
    print("🔍 检查模型当前状态...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cpu"  # 只加载到CPU检查
    )
    
    vocab_size = len(tokenizer)
    embedding_size = model.get_input_embeddings().weight.shape[0]
    lm_head_size = model.lm_head.weight.shape[0]
    available_slots = embedding_size - vocab_size
    
    print(f"📊 模型状态报告:")
    print(f"  Tokenizer 词汇表大小: {vocab_size}")
    print(f"  模型 embedding 层大小: {embedding_size}")
    print(f"  模型 lm_head 层大小: {lm_head_size}")
    print(f"  可用的预留 token 位置: {available_slots}")
    
    # 检查一些高ID的tokens
    print(f"\n🔢 高ID位置的tokens:")
    for i in range(max(0, vocab_size-5), min(vocab_size+5, embedding_size)):
        try:
            token = tokenizer.decode([i])
            print(f"  ID {i}: '{token}'")
        except:
            print(f"  ID {i}: <未定义>")
    
    return vocab_size, embedding_size, available_slots

def add_special_tokens_to_qwen(model_path, new_tokens, output_path=None):
    """
    为 Qwen 模型添加 special tokens
    
    Args:
        model_path: 原始模型路径
        new_tokens: 要添加的 special tokens 列表
        output_path: 输出路径，如果为None则就地修改
    """
    if output_path is None:
        output_path = model_path
    
    print(f"🔧 为模型添加特殊tokens: {new_tokens}")
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    original_vocab_size = len(tokenizer)
    print(f"  原始词汇表大小: {original_vocab_size}")
    
    # 添加新的 special tokens
    print(f"  添加 special tokens: {new_tokens}")
    
    # 使用 add_tokens 方法添加特殊token
    num_added = tokenizer.add_tokens(new_tokens, special_tokens=True)
    print(f"  成功添加 {num_added} 个 special tokens")
    
    new_vocab_size = len(tokenizer)
    print(f"  新词汇表大小: {new_vocab_size}")
    
    # 验证新 tokens
    print(f"\n✅ 新 tokens 的 ID:")
    token_ids = {}
    for token in new_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        token_ids[token] = token_id
        print(f"    {token}: {token_id}")
    
    # 保存修改后的 tokenizer
    print(f"💾 保存到: {output_path}")
    tokenizer.save_pretrained(output_path)
    
    return tokenizer, token_ids

def test_special_tokens(model_path, special_tokens):
    """测试特殊tokens是否正常工作"""
    print(f"🧪 测试特殊tokens...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 测试编码解码
    test_text = f"这是中文内容 {special_tokens[0]}This is English content{special_tokens[1]} 这是中文摘要"
    
    print(f"  测试文本: {test_text}")
    
    # 编码
    encoded = tokenizer.encode(test_text, add_special_tokens=False)
    print(f"  编码结果: {encoded}")
    
    # 解码
    decoded = tokenizer.decode(encoded)
    print(f"  解码结果: {decoded}")
    
    # 验证特殊tokens
    for token in special_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        decoded_token = tokenizer.decode([token_id])
        print(f"  {token} -> ID: {token_id} -> 解码: '{decoded_token}'")
    
    return encoded, decoded

def update_convert_script_for_special_tokens(examples_dir):
    """更新转换脚本以支持特殊tokens"""
    print(f"📝 更新数据转换脚本...")
    
    # 读取现有的转换脚本
    convert_script_path = os.path.join(examples_dir, "convert_to_qwen25.py")
    
    if os.path.exists(convert_script_path):
        print(f"  发现现有转换脚本: {convert_script_path}")
        # 这里可以添加更新逻辑
    else:
        print(f"  创建新的转换脚本: {convert_script_path}")
        # 创建支持特殊token的转换脚本
        create_enhanced_convert_script(convert_script_path)

def create_enhanced_convert_script(script_path):
    """创建增强的数据转换脚本"""
    script_content = '''#!/usr/bin/env python3
"""
Qwen2.5 数据格式转换器（支持特殊token '<EN>' 和 '</EN>'）
基于 LearningVerl 的实现方案
"""

import os
import json
import pandas as pd
from typing import List, Dict, Any
from tqdm import tqdm
from transformers import AutoTokenizer

class Qwen25DataConverterWithENTokens:
    """Qwen2.5数据格式转换器（支持 <EN> </EN> 特殊token）"""
    
    def __init__(self, model_path: str):
        # Qwen2.5官方system prompt
        self.system_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        
        # 固定的用户query
        self.user_query = "请你帮我写一个 4000 字的小说，并翻译成英文"
        
        # 特殊token定义（与原数据集保持一致）
        self.special_tokens = ["<EN>", "</EN>"]
        
        # Qwen2.5模板格式
        self.qwen_template = {
            "system_start": "<|im_start|>system\\n",
            "system_end": "<|im_end|>\\n",
            "user_start": "<|im_start|>user\\n", 
            "user_end": "<|im_end|>\\n",
            "assistant_start": "<|im_start|>assistant\\n",
            "assistant_end": "<|im_end|>"
        }
        
        # 加载已更新的tokenizer
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # 验证特殊tokens
        self._verify_special_tokens()
    
    def _verify_special_tokens(self):
        """验证特殊tokens是否已正确添加"""
        print("🔍 验证特殊tokens...")
        
        for token in self.special_tokens:
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if token_id == self.tokenizer.unk_token_id:
                raise ValueError(f"特殊token {token} 未找到！请先运行 add_qwen_special_tokens.py")
            print(f"  ✅ {token}: {token_id}")
    
    def load_original_data(self, data_path: str) -> List[Dict[str, Any]]:
        """加载原始数据"""
        print(f"📖 加载原始数据: {data_path}")
        
        if data_path.endswith('.parquet'):
            df = pd.read_parquet(data_path)
            data = df.to_dict('records')
        else:
            raise ValueError("支持的文件格式: .parquet")
        
        print(f"✅ 加载完成，共 {len(data)} 个样本")
        return data
    
    def extract_assistant_response(self, sample: Dict[str, Any]) -> str:
        """从原始样本中提取助手回复内容"""
        segments = sample["data"]["segments"]
        
        response_parts = []
        for segment in segments:
            content = segment["content"]
            response_parts.append(content)
        
        # 用换行符连接所有部分
        assistant_response = "\\n\\n".join(response_parts)
        return assistant_response
    
    def analyze_special_tokens(self, text: str) -> Dict[str, Any]:
        """分析文本中的特殊token使用情况"""
        analysis = {
            "has_en_tokens": False,
            "en_start_count": 0,
            "en_end_count": 0,
            "balanced": False,
            "en_segments": []
        }
        
        # 统计特殊token
        analysis["en_start_count"] = text.count("<EN>")
        analysis["en_end_count"] = text.count("</EN>")
        analysis["has_en_tokens"] = analysis["en_start_count"] > 0 or analysis["en_end_count"] > 0
        analysis["balanced"] = analysis["en_start_count"] == analysis["en_end_count"]
        
        # 提取英文段落
        if analysis["balanced"] and analysis["en_start_count"] > 0:
            import re
            en_pattern = r'<EN>(.*?)</EN>'
            matches = re.findall(en_pattern, text, re.DOTALL)
            analysis["en_segments"] = matches
        
        return analysis
    
    def create_qwen25_conversation(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """创建Qwen2.5格式的对话"""
        
        # 提取助手回复
        assistant_response = self.extract_assistant_response(sample)
        
        # 分析特殊token
        en_token_analysis = self.analyze_special_tokens(assistant_response)
        
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
        
        # 创建loss mask segments
        segments = sample["data"]["segments"]
        loss_segments = []
        
        # System部分 - 不计算loss
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
            "segment_id": 0
        })
        
        # User部分 - 不计算loss
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
            "segment_id": 1
        })
        
        # Assistant开始标签 - 不计算loss
        loss_segments.append({
            "content": self.qwen_template['assistant_start'],
            "learn": False,
            "weight": 0.0,
            "segment_type": "assistant_start_tag",
            "segment_id": 2
        })
        
        # Assistant内容部分 - 保持原有loss权重
        segment_id = 3
        for segment in segments:
            # 检查是否是特殊token
            is_special_token = segment["content"] in self.special_tokens
            
            loss_segments.append({
                "content": segment["content"],
                "learn": segment["learn"],
                "weight": segment["weight"],
                "segment_type": segment["segment_type"],
                "segment_id": segment_id,
                "is_special_token": is_special_token
            })
            segment_id += 1
        
        # Assistant结束标签 - 不计算loss
        loss_segments.append({
            "content": self.qwen_template['assistant_end'],
            "learn": False,
            "weight": 0.0,
            "segment_type": "assistant_end_tag",
            "segment_id": segment_id
        })
        
        return {
            "format": "qwen25_structured_with_en_tokens",
            "conversation": conversation,
            "data": {"segments": loss_segments},
            "id": sample["id"],
            "topic": sample["topic"],
            "template": "qwen25",
            "total_segments": len(loss_segments),
            "learning_segments": len([s for s in loss_segments if s["learn"]]),
            "original_segments": sample["total_segments"],
            "en_token_analysis": en_token_analysis
        }
    
    def convert_batch(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """批量转换样本"""
        print(f"🔄 开始转换 {len(samples)} 个样本为Qwen2.5格式...")
        
        converted_samples = []
        en_token_stats = {
            "samples_with_en_tokens": 0,
            "balanced_samples": 0,
            "total_en_segments": 0
        }
        
        for sample in tqdm(samples, desc="转换进度"):
            try:
                qwen_sample = self.create_qwen25_conversation(sample)
                converted_samples.append(qwen_sample)
                
                # 统计EN token使用情况
                en_analysis = qwen_sample["en_token_analysis"]
                if en_analysis["has_en_tokens"]:
                    en_token_stats["samples_with_en_tokens"] += 1
                    en_token_stats["total_en_segments"] += len(en_analysis["en_segments"])
                    if en_analysis["balanced"]:
                        en_token_stats["balanced_samples"] += 1
                        
            except Exception as e:
                print(f"⚠️  转换样本 {sample.get('id', 'unknown')} 时出错: {e}")
                continue
        
        print(f"✅ 转换完成，共 {len(converted_samples)} 个样本")
        print(f"📊 EN token统计:")
        print(f"  包含EN token的样本: {en_token_stats['samples_with_en_tokens']}")
        print(f"  平衡的样本数: {en_token_stats['balanced_samples']}")
        print(f"  英文段落总数: {en_token_stats['total_en_segments']}")
        
        return converted_samples
    
    def save_converted_data(self, samples: List[Dict[str, Any]], output_dir: str):
        """保存转换后的数据"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 转换为DataFrame
        df = pd.DataFrame(samples)
        
        # 按80/20比例分割
        train_size = int(0.8 * len(df))
        train_df = df[:train_size]
        test_df = df[train_size:]
        
        # 保存parquet文件
        train_path = os.path.join(output_dir, "qwen25_train.parquet")
        test_path = os.path.join(output_dir, "qwen25_test.parquet")
        
        train_df.to_parquet(train_path, index=False)
        test_df.to_parquet(test_path, index=False)
        
        # 保存配置
        config = {
            "template": "qwen25",
            "system_prompt": self.system_prompt,
            "user_query": self.user_query,
            "template_tokens": self.qwen_template,
            "special_tokens": self.special_tokens,
            "model_path": self.model_path,
            "conversion_info": {
                "format": "qwen25_structured_with_en_tokens",
                "loss_strategy": "template_excluded_content_weighted",
                "en_token_strategy": "preserved_with_original_weights"
            }
        }
        
        config_path = os.path.join(output_dir, "qwen25_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        return train_path, test_path, config_path

def main():
    print("🚀 Qwen2.5数据转换（支持EN特殊token）")
    print("=" * 60)
    
    # 路径配置
    model_path = "/home/jovyan/fdu_new/zyn/ckpts/Qwen2.5-3B-Instruct"
    input_dir = "/home/jovyan/fdu_new/zyn/examples"
    output_dir = "/home/jovyan/fdu_new/zyn/examples/qwen25_format"
    
    # 创建转换器
    converter = Qwen25DataConverterWithENTokens(model_path)
    
    # 处理数据
    all_samples = []
    for dataset_type in ["train", "test"]:
        input_path = os.path.join(input_dir, f"{dataset_type}.parquet")
        if os.path.exists(input_path):
            samples = converter.load_original_data(input_path)
            converted_samples = converter.convert_batch(samples)
            all_samples.extend(converted_samples)
    
    # 保存结果
    if all_samples:
        train_path, test_path, config_path = converter.save_converted_data(all_samples, output_dir)
        print(f"\\n✅ 转换完成！")
        print(f"  训练集: {train_path}")
        print(f"  测试集: {test_path}")
        print(f"  配置: {config_path}")

if __name__ == "__main__":
    main()
'''
    
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    os.chmod(script_path, 0o755)
    print(f"✅ 创建转换脚本: {script_path}")

def main():
    """主函数"""
    print("🚀 Qwen2.5-3B-Instruct 特殊Token添加工具")
    print("=" * 60)
    print("功能：为模型添加 '<EN>' 和 '</EN>' 特殊tokens")
    print("基于：LearningVerl issue #9 的实施方案")
    print("=" * 60)
    
    # 配置路径
    model_path = "/home/jovyan/fdu_new/zyn/ckpts/Qwen2.5-3B-Instruct"
    backup_path = f"{model_path}_backup_tokenizer"
    examples_dir = "/home/jovyan/fdu_new/zyn/examples"
    
    # 要添加的特殊tokens
    special_tokens = ["<EN>", "</EN>"]
    
    # 检查模型路径
    if not os.path.exists(model_path):
        print(f"❌ 模型路径不存在: {model_path}")
        return
    
    print(f"📂 模型路径: {model_path}")
    
    # 步骤1: 检查模型当前状态
    print(f"\n" + "="*50)
    print("步骤 1: 检查模型状态")
    print("="*50)
    
    vocab_size, embedding_size, available_slots = check_model_status(model_path)
    
    if available_slots < len(special_tokens):
        print(f"❌ 可用token位置不足！需要 {len(special_tokens)} 个，仅有 {available_slots} 个")
        return
    
    # 步骤2: 创建备份
    print(f"\n" + "="*50)
    print("步骤 2: 创建备份")
    print("="*50)
    
    if not os.path.exists(backup_path):
        print(f"📋 创建tokenizer备份: {backup_path}")
        os.makedirs(backup_path, exist_ok=True)
        
        # 备份tokenizer相关文件
        tokenizer_files = [
            "tokenizer.json", "tokenizer_config.json", 
            "vocab.json", "merges.txt", "special_tokens_map.json"
        ]
        
        for file in tokenizer_files:
            src = os.path.join(model_path, file)
            dst = os.path.join(backup_path, file)
            if os.path.exists(src):
                shutil.copy2(src, dst)
                print(f"  ✅ 备份: {file}")
    else:
        print(f"📋 备份已存在: {backup_path}")
    
    # 步骤3: 添加特殊tokens
    print(f"\n" + "="*50)
    print("步骤 3: 添加特殊tokens")
    print("="*50)
    
    try:
        tokenizer, token_ids = add_special_tokens_to_qwen(model_path, special_tokens)
        print(f"✅ 特殊tokens添加成功！")
    except Exception as e:
        print(f"❌ 添加特殊tokens失败: {e}")
        return
    
    # 步骤4: 测试验证
    print(f"\n" + "="*50)
    print("步骤 4: 测试验证")
    print("="*50)
    
    try:
        encoded, decoded = test_special_tokens(model_path, special_tokens)
        print(f"✅ 特殊tokens测试通过！")
    except Exception as e:
        print(f"❌ 特殊tokens测试失败: {e}")
        return
    
    # 步骤5: 更新相关脚本
    print(f"\n" + "="*50)
    print("步骤 5: 创建转换脚本")
    print("="*50)
    
    update_convert_script_for_special_tokens(examples_dir)
    
    # 总结
    print(f"\n" + "="*60)
    print("🎉 特殊Token添加完成！")
    print("="*60)
    print(f"✅ 已添加特殊tokens: {special_tokens}")
    print(f"📊 Token ID映射:")
    for token, token_id in token_ids.items():
        print(f"    {token}: {token_id}")
    
    print(f"\n📁 相关文件:")
    print(f"  模型路径: {model_path}")
    print(f"  备份路径: {backup_path}")
    
    print(f"\n📋 后续步骤:")
    print(f"  1. 运行数据转换脚本进行格式转换")
    print(f"  2. 使用更新后的模型进行训练")
    print(f"  3. 如需恢复，从备份路径复制文件")
    
    print(f"\n💡 集成到verl训练:")
    print(f"  使用模型路径: {model_path}")
    print(f"  特殊tokens已自动识别，无需额外配置")
    
    print("="*60)

if __name__ == "__main__":
    main()
