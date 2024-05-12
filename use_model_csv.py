import pandas as pd
import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from datetime import datetime, timedelta

# 定义生成摘要的函数
# def generate_summary(input_text, save_path, model_name="google/pegasus-xsum", max_length=512, device="cpu"):
#     # 加载保存的模型
#     loaded_model = PegasusForConditionalGeneration.from_pretrained(model_name)
#     loaded_model.load_state_dict(torch.load(save_path))
#     # 将模型移到指定设备上
#     loaded_model = loaded_model.to(device)
#     # 设置模型为评估模式
#     loaded_model.eval()
#     # 初始化分词器
#     tokenizer = PegasusTokenizer.from_pretrained(model_name)
#     # 使用加载的模型生成摘要
#     input_ids = tokenizer(input_text, return_tensors="pt", max_length=max_length, truncation=True, padding="max_length").input_ids
#     input_ids = input_ids.to(device)
#     output_ids = loaded_model.generate(input_ids, max_length=max_length, min_length=30, num_beams=4, length_penalty=2.0, early_stopping=True)
#     output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
#     return output_text


# 定义生成摘要的函数
def generate_summary(input_text, save_path, model_name="google/pegasus-xsum", max_length=512, device="cpu"):
    # 加载保存的模型
    loaded_model = PegasusForConditionalGeneration.from_pretrained(model_name)
    loaded_model.load_state_dict(torch.load(save_path, map_location=torch.device('cpu')))
    # 将模型移到指定设备上
    loaded_model = loaded_model.to(device)
    # 设置模型为评估模式
    loaded_model.eval()
    # 初始化分词器
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    # 使用加载的模型生成摘要
    input_ids = tokenizer(input_text, return_tensors="pt", max_length=max_length, truncation=True, padding="max_length").input_ids
    input_ids = input_ids.to(device)
    output_ids = loaded_model.generate(input_ids, max_length=max_length, min_length=30, num_beams=4, length_penalty=2.0, early_stopping=True)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text




# 获取前一天的日期
previous_date = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")

# 读取CSV文件
csv_file_path = f"DailyNews/{previous_date}_news.csv"  # 输入CSV文件路径
df = pd.read_csv(csv_file_path)

# 生成文本摘要
save_path = "summary_model.pth"  # 保存模型的路径

# 为每行文本生成摘要，并将其添加到新列中
df['Summary'] = df['Description'].apply(lambda x: generate_summary(x, save_path))

# 保存到新的CSV文件
output_csv_path = f"DailyNews/{previous_date}_summary.csv"  # 输出CSV文件路径
df.to_csv(output_csv_path, index=False)

print("Summary generation completed. Saved to:", output_csv_path)
