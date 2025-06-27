import pandas as pd

# 读取文件
df = pd.read_csv("deepseek_sentiment_result.csv")

# 标准化标签列
df["label_true"] = df["label_true"].astype(str).str.lower()

# 提取预测标签，0 为假新闻，1 为真新闻
def extract_binary_label(text):
    text = str(text).strip()
    if text.endswith("0"):
        return "fake"
    elif text.endswith("1"):
        return "real"
    else:
        return "unknown"

df["pred_label"] = df["predicted"].apply(extract_binary_label)

# 总体数据
total_news = len(df)
correct_total = (df["label_true"] == df["pred_label"]).sum()

# 假新闻部分
df_fake = df[df["label_true"] == "fake"]
correct_fake = (df_fake["pred_label"] == "fake").sum()

# 真新闻部分
df_real = df[df["label_true"] == "real"]
correct_real = (df_real["pred_label"] == "real").sum()

# 准确率计算
accuracy = correct_total / total_news if total_news else 0
accuracy_fake = correct_fake / len(df_fake) if len(df_fake) else 0
accuracy_real = correct_real / len(df_real) if len(df_real) else 0

# 打印输出
print("预测准确数量总计:", correct_total, f"/ {total_news}")
print("准确率 (Accuracy):", f"{accuracy:.2%}")
print("假新闻预测准确数量:", correct_fake, f"/ {len(df_fake)}")
print("假新闻准确率 (Accuracy_fake):", f"{accuracy_fake:.2%}")
print("真新闻预测准确数量:", correct_real, f"/ {len(df_real)}")
print("真新闻准确率 (Accuracy_true):", f"{accuracy_real:.2%}")
