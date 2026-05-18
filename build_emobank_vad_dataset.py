from datasets import load_from_disk, DatasetDict

ds = load_from_disk("integrated_vad_utterance_dataset/hf_dataset")

emobank = {}

for split in ["train", "validation", "test"]:
    emobank[split] = ds[split].filter(lambda x: x["source"] == "EmoBank")

emobank = DatasetDict(emobank)
emobank.save_to_disk("emobank_only_vad_dataset")

print(emobank)