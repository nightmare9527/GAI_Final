from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer, SFTConfig

# from transformers import TrainingArguments
from recursive_character_text_splitter import RecursiveCharacterTextSplitter
from sklearn.model_selection import train_test_split
import pandas as pd
from datasets import Dataset

max_seq_length = 8192  # Choose any! We auto support RoPE Scaling internally!
dtype = (
    None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
)
load_in_4bit = False  # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/mistral-7b-v0.3-bnb-4bit",  # New Mistral v3 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/llama-3-8b-bnb-4bit",  # Llama-3 15 trillion tokens model 2x faster!
    "unsloth/llama-3-8b-Instruct-bnb-4bit",
    "unsloth/llama-3-70b-bnb-4bit",
    "unsloth/Phi-3-mini-4k-instruct",  # Phi-3 2x faster!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/gemma-7b-bnb-4bit",  # Gemma 2.2x faster!
]  # More models at https://huggingface.co/unsloth

model_name = "MediaTek-Research/Breeze-7B-Instruct-v1_0"  # <- Choose one model
# model_name = "taide/Llama3-TAIDE-LX-8B-Chat-Alpha1"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",  # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=3407,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
)


def _huggingface_tokenizer_length(text: str) -> int:
    return len(tokenizer.encode(text))


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
    length_function=_huggingface_tokenizer_length,
)


def create_sliding_window(df):
    sliding_window_data = []
    for index, row in df.iterrows():
        text = row["裁判原文"]
        summary = row["摘要"]
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            sliding_window_data.append({"text": chunk, "summary": summary})
    return pd.DataFrame(sliding_window_data)


df = pd.read_excel("./data/train_preprocessed.xlsx")
df_data = df[["裁判原文", "摘要"]]

df_train, _ = train_test_split(df_data, test_size=0.1, random_state=66)
df_train_sliding_window = create_sliding_window(df_train)

train_dataset = Dataset.from_pandas(df_train_sliding_window)
system_prompt = """你是一位書記官，你的任務是書寫判決書摘要"""
init_prompt = """請精簡並摘要以下文章：
<DOCUMENT>
{document}
</DOCUMENT>
"""


def format_chat(document: str, summary: str):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": init_prompt.format(document=document)},
        {"role": "assistant", "content": summary},
    ]


def formatting_prompts_func(examples):
    texts = [
        tokenizer.apply_chat_template(format_chat(text, summary), tokenize=False)
        for text, summary in zip(examples["text"], examples["summary"])
    ]
    return {
        "text": texts,
    }


train_dataset = train_dataset.map(formatting_prompts_func, batched=True)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,  # Can make training 5x faster for short sequences.
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        # max_steps = 60,
        num_train_epochs=2,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
    ),
)
trainer_stats = trainer.train()
print(trainer_stats)
model.save_pretrained("breeze_lora_model_v2")  # Local saving
tokenizer.save_pretrained("breeze_lora_model_v2")
