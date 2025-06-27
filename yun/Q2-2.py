import pandas as pd
import requests

# 读取情绪分析后的 Excel 文件
df = pd.read_excel("sentiment_result.xlsx")

# 提取情绪字段（从sentiment列最后一句识别）
def extract_emotion(text):
    if pd.isna(text):
        return "未知"
    if "消极" in text:
        return "消极"
    elif "积极"in text:
        return "积极"
    elif "中性" in text:
        return "中性"
    else:
        return "未知"

df["emotion"] = df["sentiment"].astype(str).apply(extract_emotion)

# 存放结果
results = []

for i, row in df.iterrows():
    text = row["text"]
    label_true = row["label"]
    emotion = row["emotion"]

    # 加入情绪提示，并让模型输出 0 或 1
    prompt = f"""请判断以下新闻是否是假新闻（结合情绪倾向进行分析）：
- 情绪倾向为：{emotion}
- 如果是假新闻，请只输出：0
- 如果是真新闻，请只输出：1

{text}"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "deepseek-r1:1.5b",
                "prompt": prompt,
                "stream": False
            }
        )

        if response.status_code == 200:
            result = response.json()["response"].strip()
            results.append({
                "text": text,
                "label_true": label_true,
                "emotion": emotion,
                "predicted": result
            })
            print(f"✅ 第 {i+1} 条完成：{result}")
        else:
            print(f"❌ 第 {i+1} 条出错：{response.status_code}")
    except Exception as e:
        print(f"❌ 第 {i+1} 条异常：{str(e)}")

# 保存结果
df_result = pd.DataFrame(results)
df_result.to_csv("deepseek_sentiment_result.csv", index=False, encoding="utf-8")
print("🎉 全部完成，结果已保存到 deepseek_sentiment_result.csv")
