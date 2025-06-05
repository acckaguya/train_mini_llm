import json
import re
from tqdm import tqdm

input_path = "./dataset/sft_mini_512.jsonl"
output_path = "./dataset/tokenizer_training_data.jsonl"


def clean_text(text):
    """文本清洗函数"""
    # 1. 移除Markdown/HTML标签
    text = re.sub(r'[\*\_\#$$$$$$]', '', text)  # 移除*_#[]()等标记符号
    text = re.sub(r'\*\*.*?\*\*', lambda x: x.group()[2:-2], text)  # 移除加粗
    text = re.sub(r'`.*?`', lambda x: x.group()[1:-1], text)  # 移除代码块
    
    # 2. 处理换行和多余空格
    text = ' '.join(text.split())  # 合并连续空格/换行
    
    # 3. 中文特殊处理（可选）
    text = text.replace('“', '"').replace('”', '"')  # 统一引号
    
    # 4. 其他自定义规则
    text = text.replace('\u3000', ' ')  # 替换中文空格
    
    return text.strip()

def process_sft_to_tokenizer_data(input_path, output_path):
    """处理SFT数据为Tokenizer训练格式"""
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        for line in tqdm(fin, desc='Processing'):
            try:
                data = json.loads(line)
                conversations = data.get('conversations', [])
                
                # 合并所有对话内容
                full_text = ""
                for turn in conversations:
                    if isinstance(turn, dict) and 'content' in turn:
                        cleaned = clean_text(turn['content'])
                        full_text += f"{cleaned}\n"  # 每段对话用换行分隔
                
                if full_text.strip():
                    # 写入新的JSONL格式（每行一个包含清洗后text的对象）
                    json.dump({"text": full_text.strip()}, fout, ensure_ascii=False)
                    fout.write('\n')
                    
            except json.JSONDecodeError:
                print(f"忽略无效行: {line}")

# 使用示例
process_sft_to_tokenizer_data(input_path,output_path)