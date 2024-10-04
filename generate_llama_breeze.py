import json
from unsloth import FastLanguageModel
import pandas as pd
from recursive_character_text_splitter import RecursiveCharacterTextSplitter
from ignite.metrics import Rouge
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import jieba
from transformers import pipeline


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


max_seq_length = 8192
dtype = None
load_in_4bit = False

model_name = "breeze_lora_model_v2"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="breeze_lora_model_v2",  # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    device_map="auto",
)
FastLanguageModel.for_inference(model)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
)
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
# choose train or test
# Train
# df = pd.read_excel("./data/train_preprocessed.xlsx")

# df_data = df[["裁判原文", "摘要"]]
# _, df_test = train_test_split(df_data, test_size=0.1, random_state=66)

# Test
df = pd.read_excel("./data/test.xlsx")
df_test = df[["裁判原文", "摘要"]]

test_sliding_window = create_sliding_window_dict(df_test)
system_prompt = """你是一位書記官，你的任務是書寫判決書摘要"""
init_prompt = """請精簡並摘要以下文章：
<DOCUMENT>
{document}
</DOCUMENT>
"""
folllow_prompt = """請精簡並摘要以下文章：
<DOCUMENT>
{summary}

{document}
</DOCUMENT>
"""

list_result = []
for window in tqdm(test_sliding_window):
    chunks = window["chunk"]
    result = []
    for index, chunk in enumerate(tqdm(chunks)):
        if index == 0:
            chat = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": init_prompt.format(document=chunk),
                },
            ]
            prompt = tokenizer.apply_chat_template(chat, tokenize=False)
            x = pipe(prompt, max_new_tokens=1024, return_full_text=False, batch_size=2)
            result.append(x[0]["generated_text"])
        else:
            chat = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": folllow_prompt.format(
                        document=chunk, summary=result[-1]
                    ),
                },
            ]
            prompt = tokenizer.apply_chat_template(chat, tokenize=False)
            x = pipe(prompt, max_new_tokens=1024, return_full_text=False, batch_size=2)
            result.append(x[0]["generated_text"])
    list_result.append(result)

# with open("result_breeze_unsloth_v2.json", "w") as f:
# json.dump(list_result, f)
with open(f"{model_name}_test_result.json", "w") as f:
    json.dump(list_result, f, ensure_ascii=False)

# rouge = Rouge(variants=["L", 1, 2])
# for window in test_sliding_window:
#     summary = window["summary"]
# unique = [window["summary"] for window in test_sliding_window]
# for candidate, references in zip(list_result, unique):
#     sentence = " ".join(jieba.cut(candidate[-1])).split()
#     ground_truth = " ".join(jieba.cut(references)).split()
#     rouge.update(([sentence], [[ground_truth]]))
# print(rouge.compute())
