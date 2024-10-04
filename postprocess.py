import json
import pandas as pd


df = pd.read_excel("./data/test.xlsx")
# df_test = df[["裁判原文", "摘要"]]
with open(f"breeze_lora_model_v2_test_result.json", "r") as f:
    list_result = json.load(f)

summary_list = [l[-1] for l in list_result]
df["摘要"] = summary_list

df.to_excel("./data/submit.xlsx", index=False)
df.to_csv("./data/submit.csv", index=False)
