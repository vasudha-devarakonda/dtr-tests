import torch
torch.manual_seed(seed=42)
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    # TrainerCallback
)
from datasets import load_dataset
import os
from typing import Dict, List, Optional
import argparse
from tqdm import tqdm
import time

'''
Some small llm
microsoft/phi-2
openlm-research/open_llama_3b
EleutherAI/gpt-neo-125M
facebook/opt-350m
EleutherAI/pythia-160m
'''

torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)


class ForwardWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs):
        out = self.model(**inputs)
        return out.logits   # remove HF dict → required for stable dynamo export



# class EmptyCacheCallback(TrainerCallback):
#     def on_step_end(self, args, state, control, **kwargs):
#         torch.cuda.empty_cache()

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_model_size_bytes(model):
    """Calculate model parameter size in bytes"""
    total_params = 0
    for param in model.parameters():
        total_params += param.numel()
    # Assuming float32 parameters (4 bytes each)
    return total_params * 4

class CustomDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512, device='cpu'):
        self.device = device
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: val[idx].to(self.device) for key, val in self.encodings.items()}
        item['labels'] = item['input_ids'].clone()
        return item

    def __len__(self) -> int:
        return len(self.encodings.input_ids)

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_loss = float('inf')
        self.log_frequency = 200

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """Custom logging with additional metrics"""
        if self.state.global_step % self.log_frequency == 0:
            super().log(logs)

            # Calculate and log perplexity
            if 'loss' in logs:
                perplexity = torch.exp(torch.tensor(logs['loss'])).item()
                current_lr = self.optimizer.param_groups[0]['lr']

                print(f"\nStep: {self.state.global_step}")
                print(f"Loss: {logs['loss']:.4f}")
                print(f"Perplexity: {perplexity:.4f}")
                print(f"Learning Rate: {current_lr}")

                # Track best loss
                if logs['loss'] < self.best_loss:
                    self.best_loss = logs['loss']
                    print(f"New Best Loss: {self.best_loss:.4f}")

def train_loop(model, train_dataset, batch_size, epochs, use_dtr=True):
    # write my own training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    
    num_iters = min(len(dataloader), args.iter_limit) if args.iter_limit != -1 else len(dataloader)
    prev_base_compute_time = 0
    prev_remat_compute_time = 0
    prev_search_time = 0
    prev_cost_time = 0
    prev_remate_count = 0
    prev_remate_size = 0
    results = []
    i_no = 0
    model.train()
    model.zero_grad()
    model =  model._apply(lambda v: v.detach().checkpoint())
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        start = time.time()
        progress_bar = tqdm(total=num_iters, desc=f"Training Epoch {epoch}", leave=False)
        for step, batch in enumerate(dataloader):
            torch.cuda.reset_max_memory_allocated()
            if step >= num_iters - 1:
                break
            start_time = time.time()
            batch = {k: v.to(device) for k, v in batch.items()}
            tensor = batch['input_ids']
            labels = tensor
            tensor = tensor.checkpoint()
            labels = labels.checkpoint()
            outputs = model(tensor, labels=labels)
            logits = outputs.logits

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            print("\n\n=================\n\n\n")
            print(f"tensor shape: {tensor.shape}")
            print(f"labels shape: {labels.shape}")
            print(f"logits shape: {logits.shape}")
            print(f"shift_logits shape: {shift_logits.shape}")
            print(f"shift_labels shape: {shift_labels.shape}")
            print("\n\n==============\n\n\n")
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            optimizer.zero_grad()
            loss.backward()
            print("\n\n\n jjsjdff\n\n\n")
            tensor = tensor.decheckpoint()
            loss = loss.decheckpoint()
            labels = labels.decheckpoint()
            logits = logits.decheckpoint()
            print("\nupdating the gradient=========\n")
            optimizer.step()

            
            # progress_bar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
            progress_bar.update(1)
            end_time = time.time()
            total_mem = torch.cuda.max_memory_allocated()
            base_compute_time = torch.base_compute_time() - prev_base_compute_time
            remat_compute_time = torch.remat_compute_time() - prev_remat_compute_time
            search_time = torch.search_time() - prev_search_time
            cost_time = torch.cost_time() - prev_cost_time
            remate_count = torch.remat_compute_count() - prev_remate_count
            remate_size = torch.remat_compute_size() - prev_remate_size
            if step % 100 == 0:
                time_taken = time.time() - start
                print(f"Step {step} Time elapsed: {time_taken:.2f}s")
            result = {
                'time': end_time - start_time,
                'iter': i_no,
                'total_mem': total_mem,
                'base_compute_time': base_compute_time,
                'remat_compute_time': remat_compute_time,
                'search_time': search_time,
                'cost_time': cost_time, 
                'remat_count': remate_count, 
                'remat_size': remate_size,
                'epoch': epoch
            }
            prev_base_compute_time += base_compute_time
            prev_remat_compute_time += remat_compute_time
            prev_search_time += search_time
            prev_cost_time += cost_time
            prev_remate_count +=  remate_count
            prev_remate_size += remate_size
            i_no += 1
            results.append(result)
            del loss
            del labels 
            del tensor
        finish = time.time()
        time_taken = finish - start
        print('epoch {} training time consumed: {:.2f}s'.format(epoch, time_taken))
        progress_bar.close()
    del model
    return results

dataset_name_map = {
    "databricks-dolly-15k": "databricks/databricks-dolly-15k",
    "databricks-databricks-dolly-15k": "databricks/databricks-dolly-15k",
    "dolly-15k": "databricks/databricks-dolly-15k",
}
def prepare_data(dataset_name) -> List[str]:
    # Load dataset from Hugging Face
    dataset = load_dataset(dataset_name_map[dataset_name], split="train")

    dataset = dataset.train_test_split(test_size=0.1)  # Adjust test_size as needed

    # Extract instruction-response pairs for each split
    train_texts = [
        f"Instruction: {item['instruction']}\n\nResponse: {item['response']}"
        for item in dataset['train']
    ]
    eval_texts = [
        f"Instruction: {item['instruction']}\n\nResponse: {item['response']}"
        for item in dataset['test']
    ]

    return train_texts, eval_texts


def report_results(model_name, measurements, memory_budget, batch_size):
    if not measurements:
        return  # nothing to write8
    out_file_name = f"{model_name}_{batch_size}_{int(memory_budget/(1024*1024*1024))}.csv"
    with open(out_file_name, 'a', newline='') as csvfile:
        for j, data in enumerate(measurements):
            entry = {
                'model_name': model_name,
                'batch_size': batch_size,
                'rep': j,
                'epoch': data['epoch'],
                'time': data['time']*1e3,
                'total_mem': data['total_mem']*1e-6,
                'memory_budget': memory_budget,
                'base_compute_time': data['base_compute_time']*1e-6,
                'remat_compute_time': data['remat_compute_time']*1e-6,
                'search_time': data['search_time']*1e-6,
                'cost_time': data['cost_time']*1e-6,
                'remat_count': data['remat_count'],
                'remat_size': data['remat_size']*1e-6
            }

            # Create writer dynamically based on entry keys
            writer = csv.DictWriter(csvfile, fieldnames=entry.keys())

            # Write header only if file is empty
            if csvfile.tell() == 0:
                writer.writeheader()

            writer.writerow(entry)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    # parser.add_argument('-gpu-device', type=int, default=0, help='device id to use')
    parser.add_argument('-gpu-device', type=str, default="0", help='device id to use')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-e', type=int, default=2, help='number of epochs')
    parser.add_argument('-d', type=str, default='databricks-dolly-15k', help='dataset')
    parser.add_argument('-iter-limit', type=int, default=-1, help='limit the number of iterations per epoch, -1 to use full dataset')
    parser.add_argument('-c', action='store_true', default=False, help='use checkpoint')
    parser.add_argument('-empty-cache', action='store_true', default=False, help='use checkpoint')
    parser.add_argument('-no-eval', action='store_true', default=False, help='do evaluation after training')
    

    parser.add_argument('-rockmate', action='store_true', default=False, help='rockmate optimization')
    parser.add_argument('-hiremate', action='store_true', default=False, help='hiremate optimization')
    parser.add_argument('-offmate', action='store_true', default=False, help='offmate optimization')
    parser.add_argument('-checkmate', action='store_true', default=False, help='checkmate optimization')
    parser.add_argument('-segment-ilp', action='store_true', default=False, help='segment ilp optimization')
    parser.add_argument('-no-recompute', action='store_true', default=False, help='disable recomputation in rockmate')
    parser.add_argument('-dtr', action='store_true', default=False, help='dtr optimization')
    
    parser.add_argument('-budget', type=float, default=3, help='memory budget')
    parser.add_argument('-filename', type=str,default="results", help='file_to_write')
    parser.add_argument('-ilps', type=int, default=2, help='num,ber of ilps')
    # parser.add_argument('-print-memory', action='store_true', default=False, help='print memory usage after each epoch')
    # parser.add_argument('-no-iter-progress', action='store_true', default=False, help='show progress bar for iterations')
    # parser.add_argument('-no-eval', action='store_true', default=False, help='do evaluation after training')
    args = parser.parse_args()
    memory_budget = args.b *1024*1024*1024
    torch.set_memory_budget(memory_budget)

    device = torch.device("cpu")
    use_gpu = False
    if torch.cuda.is_available():
        device = torch.device('cuda:'+args.gpu_device)
        # torch.cuda.set_device(args.gpu_device)
        torch.cuda.set_device(device)
        print(f"Using GPU device {torch.cuda.current_device()}")
        device = torch.device("cuda")
        use_gpu = True
    else:
        print("No GPU available, using CPU")
    print(f"Using device: {device}")


    model_name = args.net  # You can easily change this to any model
    
    model = AutoModelForCausalLM.from_pretrained(model_name)


    model.to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    # Prepare dataset
    train_texts, eval_texts = prepare_data(args.d)
    train_dataset = CustomDataset(train_texts, tokenizer, device=device)
    eval_dataset = CustomDataset(eval_texts, tokenizer, device=device)
    
    train_loader = DataLoader(train_dataset, batch_size=args.b, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=args.b, shuffle=False)
    offline_time = 0
    time_start = time.time()

        
    batch_size = args.b 
    
    time_start = time.time()
    results = train_loop(model, train_dataset, args.b, args.e)
    use_dtr = True
    report_results(model_name,results, memory_budget, batch_size)
    time_train = time.time() - time_start
    
    
# python ./train_llm.py -net EleutherAI/pythia-160m -gpu -b 4 -e 1 -dtr -budget 10
# python ./train_llm.py -net EleutherAI/pythia-160m -gpu -b 4 -e 1 -no-recompute