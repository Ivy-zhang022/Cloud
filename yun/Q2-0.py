import pandas as pd
import requests

# 读取数据（注意是 Excel）
df = pd.read_excel("posts.xlsx")

# 存放结果
results = []

for i, row in df.iterrows():
    text = row['post_text']
    label_true = row['label']

    prompt = f"""请判断以下新闻是否是假新闻:
    如果是假新闻，只输出:0
    如果是真新闻，只输出:1
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
                "predicted": result
            })
            print(f"✅ 第 {i+1} 条完成：{result}")
        else:
            print(f"❌ 第 {i+1} 条出错：{response.status_code}")
    except Exception as e:
        print(f"❌ 第 {i+1} 条异常：{str(e)}")

# 保存结果
pd.DataFrame(results).to_csv("deepseek_result.csv", index=False, encoding="utf-8")
print("🎉 全部完成，结果已保存到 deepseek_result.csv")