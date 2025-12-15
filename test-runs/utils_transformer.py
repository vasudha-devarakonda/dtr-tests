from transformers import GPT2Tokenizer, GPT2ForSequenceClassification,AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from torch.utils.data.distributed import DistributedSampler
from typing import List, Dict
dataset_name_map = { "databricks-dolly-15k": "databricks/databricks-dolly-15k", "databricks-databricks-dolly-15k": "databricks/databricks-dolly-15k", "dolly-15k": "databricks/databricks-dolly-15k", }
class CustomClassificationDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512, device='cpu'):
        self.device = device
        self.labels = labels
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
            return_attention_mask=False
        )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: val[idx].to(self.device) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long, device=self.device)
        return item

    def __len__(self) -> int:
        return len(self.labels)


def prepare_data_classification(dataset_name):
    # Load dataset from Hugging Face
    dataset = load_dataset(dataset_name_map[dataset_name], split="train")
    dataset = dataset.train_test_split(test_size=0.1)

    # Convert instruction+response to a single text for classification
    train_texts = [
        f"Instruction: {item['instruction']} Response: {item['response']}"
        for item in dataset['train']
    ]
    eval_texts = [
        f"Instruction: {item['instruction']} Response: {item['response']}"
        for item in dataset['test']
    ]

    # Mock labels: e.g., binary classification based on presence of some word, just for example
    # Replace this with your real label extraction logic!
    train_labels = [0 if "no" in text.lower() else 1 for text in train_texts]
    eval_labels = [0 if "no" in text.lower() else 1 for text in eval_texts]

    return train_texts, train_labels, eval_texts, eval_labels


def fetch_dataloaders(dataset_name, model_name, batch_size, device='cpu', distributed=False, world_size=0, rank=0):
    print(f"the world_size is {world_size} and rank is {rank}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_texts, train_labels, eval_texts, eval_labels = prepare_data_classification(dataset_name)

    train_dataset = CustomClassificationDataset(train_texts, train_labels, tokenizer, device=device)
    eval_dataset = CustomClassificationDataset(eval_texts, eval_labels, tokenizer, device=device)
    if False:
        sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        train_loader = DataLoader(
                train_dataset, shuffle=False, batch_size=batch_size,sampler=sampler)
        sampler = DistributedSampler(eval_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        eval_loader = DataLoader(
                train_dataset, shuffle=False, batch_size=batch_size,sampler=sampler)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, drop_last=True)  
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, eval_loader, tokenizer