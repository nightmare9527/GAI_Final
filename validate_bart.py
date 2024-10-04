import pandas as pd
from recursive_character_text_splitter import RecursiveCharacterTextSplitter

from datasets import Dataset
import opencc
from ignite.metrics import Rouge
import jieba
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import (
    BartForConditionalGeneration,
    BartTokenizerFast,
    Text2TextGenerationPipeline,
)


def get_tokenizer_len_func(tokenizer=None):
    if tokenizer:

        def _huggingface_tokenizer_length(text: str) -> int:
            return len(tokenizer.encode(text))

        return _huggingface_tokenizer_length
    else:
        return len


s2t_converter = opencc.OpenCC("s2t.json")
t2s_converter = opencc.OpenCC("t2s.json")
rouge = Rouge(variants=["L", 1, 2])

model = BartForConditionalGeneration.from_pretrained("./model_bart")
tokenizer = BartTokenizerFast.from_pretrained("./model_bart")
prefix = "summary: "
text2text_generator = Text2TextGenerationPipeline(model, tokenizer, device="cuda")

if tokenizer.model_max_length < 10000:
    window_size = tokenizer.model_max_length
elif hasattr(model.config, "max_position_embeddings"):
    window_size = model.config.max_position_embeddings
else:
    window_size = 512

tokenizer_len_func = get_tokenizer_len_func(tokenizer)
chunk_size = window_size - tokenizer_len_func(prefix)
chunk_overlap = chunk_size // 4

print(f"chunk_size: {chunk_size}, chunk_overlap: {chunk_overlap}")

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
    length_function=tokenizer_len_func,
)

df = pd.read_excel("./data/train_preprocessed.xlsx")

df["裁判原文"] = df["裁判原文"].apply(lambda x: t2s_converter.convert(x))
df["摘要"] = df["摘要"].apply(lambda x: t2s_converter.convert(x))


def create_sliding_window(df):
    sliding_window_data = []
    for _, row in df.iterrows():
        text = row["裁判原文"]
        summary = row["摘要"]
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            sliding_window_data.append({"text": chunk, "summary": summary})
    return pd.DataFrame(sliding_window_data)


df_train, df_test = train_test_split(df, test_size=0.1, random_state=66)
df_train_sliding_window = create_sliding_window(df_train)
df_test_sliding_window = create_sliding_window(df_test)

train_dataset = Dataset.from_pandas(df_train_sliding_window)
test_dataset = Dataset.from_pandas(df_test_sliding_window)

prefix_test_dataset = test_dataset.map(
    lambda x: {"text": prefix + x["text"], "summary": x["summary"]}
)
gen = text2text_generator(
    prefix_test_dataset["text"],
    max_length=chunk_size // 2,
    batch_size=8,
    num_beams=4,
    length_penalty=1.5,
    no_repeat_ngram_size=2,
)
group = ""
list_group = []
last_summ = test_dataset[0]["summary"]
for g, s in zip(gen, test_dataset["summary"]):
    if s == last_summ:
        group += g["generated_text"]
    else:
        last_summ = s
        list_group.append(group)
        group = g["generated_text"]
list_group.append(group)


unique = list(dict.fromkeys(test_dataset["summary"]))
for candidate, references in zip(list_group, unique):
    sentence = " ".join(jieba.cut(candidate)).split()
    ground_truth = " ".join(jieba.cut(references)).split()
    rouge.update(([sentence], [[ground_truth]]))
print(rouge.compute())
