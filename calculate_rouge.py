import json
import pandas as pd
from recursive_character_text_splitter import RecursiveCharacterTextSplitter
from ignite.metrics import Rouge
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import jieba
from transformers import AutoTokenizer


def get_tokenizer_len_func(tokenizer=None):
    if tokenizer:

        def _huggingface_tokenizer_length(text: str) -> int:
            return len(tokenizer.encode(text))

        return _huggingface_tokenizer_length
    else:
        return len


def create_sliding_window_dict(df):
    sliding_window_data = []
    for index, row in df.iterrows():
        text = row["裁判原文"]
        summary = row["摘要"]
        chunks = text_splitter.split_text(text)
        sliding_window_data.append({"summary": summary, "chunk": chunks})
    return sliding_window_data


model_name = "breeze_lora_model_v2"
# model_name = "llama_lora_model_v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
chunk_size = 6000
chunk_overlap = 1024
text_splitter = RecursiveCharacterTextSplitter(
    separators=[
        "\n\n",
        "\n",
        " ",
        ".",
        ",",
        "\uff0e",  # Fullwidth full stop
        "\u3002",  # Ideographic full stop
        "\uff1f",  # Fullwidth question mark
        "\uff01",  # Fullwidth exclamation mark
        "\uff0c",  # Fullwidth comma
        "\u3001",  # Ideographic comma
        "\uff1b",  # Fullwidth semicolon
        "\uff1a",  # Fullwidth colon
    ],
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    keep_separator="end",
    length_function=get_tokenizer_len_func(tokenizer),
)
with open(f"{model_name}_test_result.json", "r") as f:
    list_result = json.load(f)


def create_sliding_window_dict(df):
    sliding_window_data = []
    for index, row in df.iterrows():
        text = row["裁判原文"]
        summary = row["摘要"]
        chunks = text_splitter.split_text(text)
        sliding_window_data.append({"summary": summary, "chunk": chunks})
    return sliding_window_data


df = pd.read_excel("./data/train_preprocessed.xlsx")
df_data = df[["裁判原文", "摘要"]]
_, df_test = train_test_split(df_data, test_size=0.1, random_state=66)
test_sliding_window = create_sliding_window_dict(df_test)

rouge = Rouge(variants=["L", 1, 2])
for window in test_sliding_window:
    summary = window["summary"]
unique = [window["summary"] for window in test_sliding_window]
for candidate, references in zip(list_result, unique):
    sentence = " ".join(jieba.cut(candidate[-1])).split()
    ground_truth = " ".join(jieba.cut(references)).split()
    rouge.update(([sentence], [[ground_truth]]))
print(rouge.compute())
