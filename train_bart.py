import pandas as pd
from recursive_character_text_splitter import RecursiveCharacterTextSplitter

from transformers import (
    BartForConditionalGeneration,
    BartTokenizerFast,
    MT5ForConditionalGeneration,
    T5Tokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

from datasets import Dataset
import opencc
from ignite.metrics import Rouge
import jieba
import numpy as np
from sklearn.model_selection import train_test_split


def get_tokenizer_len_func(tokenizer=None):
    if tokenizer:

        def _huggingface_tokenizer_length(text: str) -> int:
            return len(tokenizer.encode(text))

        return _huggingface_tokenizer_length
    else:
        return len


def get_model(model_type: str):
    if model_type == "IDEA-CCNL/Randeng-BART-139M-SUMMARY":
        model = BartForConditionalGeneration.from_pretrained(model_type)
        tokenizer = BartTokenizerFast.from_pretrained(model_type)
        prefix = "summary: "
    elif model_type == "heack/HeackMT5-ZhSum100k":
        model = MT5ForConditionalGeneration.from_pretrained(model_type)
        tokenizer = T5Tokenizer.from_pretrained(model_type)
        prefix = "summarize: "
    else:
        raise ValueError("Model type not supported")
    return model, tokenizer, prefix


s2t_converter = opencc.OpenCC("s2t.json")
t2s_converter = opencc.OpenCC("t2s.json")

model_type = "IDEA-CCNL/Randeng-BART-139M-SUMMARY"
model, tokenizer, prefix = get_model(model_type)

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

df = pd.read_excel("./data/train_with_id.xlsx")

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


def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=chunk_size, truncation=True)
    labels = tokenizer(examples["summary"], max_length=chunk_size, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)

rouge = Rouge(variants=["L", 1, 2])


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(
        predictions, skip_special_tokens=True, max_length=chunk_size, truncation=True
    )
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    for pred, l in zip(decoded_preds, decoded_labels):
        sentence = " ".join(jieba.cut(pred)).split()
        ground_truth = " ".join(jieba.cut(l)).split()
        rouge.update(([sentence], [[ground_truth]]))
    result = rouge.compute()
    return {k: round(v, 4) for k, v in result.items()}


data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
training_args = Seq2SeqTrainingArguments(
    run_name="train-4",
    output_dir="law_train",
    report_to="wandb",
    eval_strategy="epoch",
    # evaluation_strategy="epoch",
    learning_rate=1e-4,
    # per_device_train_batch_size=2,
    # per_device_eval_batch_size=2,
    auto_find_batch_size=True,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=10,
    predict_with_generate=True,
    fp16=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("./model_bart")
