import pandas as pd
import requests

# 读取文档数据，需包含一列为 post_text（或 text）字段
df = pd.read_excel("posts.xlsx")

results = []

for i, row in df.iterrows():
    text = row["post_text"]
    prompt = f"请分析以下文本的情感倾向，只输出：积极、中性、消极 三个中的一个。\n\n{text}"

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",  # 本地部署地址
            json={
                "model": "deepseek-r1:1.5b",  # 使用你的模型名
                "prompt": prompt,
                "stream": False
            }
        )
        if response.status_code == 200:
            sentiment = response.json()["response"].strip()
            results.append({
                "text": text,
                "sentiment": sentiment
            })
            print(f"✅ 第 {i+1} 条完成：{sentiment}")
        else:
            print(f"❌ 第 {i+1} 条出错：{response.status_code}")
    except Exception as e:
        print(f"❌ 第 {i+1} 条异常：{str(e)}")

# 保存结果
pd.DataFrame(results).to_csv("sentiment_result.csv", index=False, encoding="utf-8")
print("🎉 情感分析完成，结果保存在 sentiment_result.csv")
