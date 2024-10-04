import pandas as pd
import uuid

df = pd.read_excel("./data/train.xlsx")
df = df[~df["裁判原文"].str.contains("點選下方附件連結")]
df = df[["裁判原文", "摘要"]]
df["id"] = [uuid.uuid4() for _ in range(len(df))]

df.to_excel("./data/train_preprocessed.xlsx", index=False)
