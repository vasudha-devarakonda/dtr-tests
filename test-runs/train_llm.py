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
import csv 

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
identity = lambda x: x

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

def teardown(model):
    pass

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


import gc
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def prepare_llm(model_name, batch_size, use_dtr=True):
    """
    Returns model + tokenizer, but run_model expects data/target already as input_ids tensors.
    Uses LM loss only.
    """

    def prepare_llm(extra_params=None):
        model = AutoModelForCausalLM.from_pretrained(model_name) 
        model.cuda(0)
        model.train()
        model.zero_grad()
        model._apply(lambda v: v.detach().checkpoint())
        return [model]

    def run_model(model, data, target,
                  process_model=identity,
                  process_output=identity,
                  process_loss=identity,
                  optimizer=None):
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        process_model(model)
        input_ids = data.cuda(0)
        labels    = target.cuda(0)

        input_ids = input_ids.checkpoint()
        labels    = labels.checkpoint()

        output = model(input_ids=input_ids, labels=labels)
        logits = output.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
    
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        process_output(logits)
        process_loss(loss)

        optimizer.zero_grad()

        torch.annotate_log("BACKWARD")

        loss.backward()

        input_ids = input_ids.decheckpoint()
        labels    = labels.decheckpoint()
        loss      = loss.decheckpoint()
        logits = logits.decheckpoint()

        optimizer.step()

        del input_ids, labels, loss, output

    def teardown(model_list):
        pass

    return prepare_llm, run_model, teardown

def run_single_measurement(model_name, produce_model, run_model, teardown, inp,
                             use_dtr,  e, batch_size, memory_budget):
    use_dtr = True
    torch.cuda.reset_max_memory_allocated()
    remate_count_previous = torch.remat_compute_count() 
    remate_size_previous = torch.remat_compute_size()
    # resetting means the count should be reset to
    # only what's in scope, meaning only the input
    start_time_model = time.time()
    input_mem = torch.cuda.max_memory_allocated()
    model = produce_model(extra_params=[])
    params = []
    for m in model:
        if hasattr(m, 'parameters'):
            params.extend(m.parameters())

    model_mem = torch.cuda.max_memory_allocated()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)


    torch.cuda.synchronize()
    start_time = time.time()
    if use_dtr:
        torch.reset_profile()
    start.record()
    # with torch.autograd.profiler.profile(use_cuda=True) as prof:
    run_model(*model, *inp)
    end.record()
    start_sync = time.time()
    torch.cuda.synchronize()
    end_sync = time.time()
    end_time = time.time()
    # end timing

    if use_dtr:
        # operators-only time, tracked by DTR
        cuda_time = torch.compute_time()

    base_compute_time = torch.base_compute_time()
    remat_compute_time = torch.remat_compute_time()
    search_time = torch.search_time()
    cost_time = torch.cost_time()
    remate_count = torch.remat_compute_count() 
    remate_size = torch.remat_compute_size()
    total_mem = torch.cuda.max_memory_allocated()
    teardown(*model)
    torch.cuda.reset_max_memory_allocated()

    del model

    if use_dtr:
        torch.toggle_log(False)

    del params

    result = {
        'time': end_time - start_time,
        'time_with_model': end_time - start_time_model,
        'sync_time': end_sync - start_sync,
        'gpu_time': start.elapsed_time(end),
        'input_mem': input_mem,
        'model_mem': model_mem,
        'total_mem': total_mem,
        'base_compute_time': base_compute_time,
        'remat_compute_time': remat_compute_time,
        'search_time': search_time,
        'cost_time': cost_time, 
        'remat_count': remate_count - remate_count_previous, 
        'remat_size': remate_size - remate_size_previous,
        'epoch': e
    }
    if use_dtr:
        result['cuda_time'] = cuda_time
    else:
        result['cuda_time'] = -1.0
    return result

def train_loop(model_name, train_dataset, batch_size, epochs, use_dtr=True):
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    produce_model, run_model, teardown = prepare_llm(model_name,batch_size,use_dtr=True)
    num_iters = min(len(dataloader), args.iter_limit) if args.iter_limit != -1 else len(dataloader)
    results = []
    i_no = 0
    start_total_epoch = time.time()
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        progress_bar = tqdm(total=num_iters, desc=f"Training Epoch {epoch}", leave=False)
        for step, batch in enumerate(dataloader):
            gc.collect()
            if step >= num_iters - 1:
                break
            torch.cuda.reset_max_memory_allocated()
            start_time_with_model = time.time()
            remate_count_previous = torch.remat_compute_count() 
            remate_size_previous = torch.remat_compute_size()
            # params = []
            # for m in model:
            #     if hasattr(m, 'parameters'):
            #         params.extend(m.parameters())
            start_time = time.time()
            batch = {k: v.to(device) for k, v in batch.items()}
            tensor = batch['input_ids']
            labels = tensor
            tensor = tensor.cuda(0)
            labels = labels.cuda(0)
            inp = (tensor, labels)
            result = run_single_measurement(model_name, produce_model, run_model, teardown,
                                                inp, use_dtr,epoch, batch_size, memory_budget)
            progress_bar.update(1)
            results.append(result)
        progress_bar.close()
    finish_epoch = time.time()
    time_taken_epoch = finish_epoch - start_total_epoch
    file_name = f"llm_models_time.txt"

    # Open the file in append mode and save the time for this epoch
    with open(file_name, 'a') as file:
        file.write(f"Model:  {model_name}: {time_taken_epoch:.2f}s\n")
    torch.cuda.empty_cache()

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
    safe_model_name = model_name.replace("/", "_")
    out_file_name = f"{safe_model_name}_{batch_size}_{int(memory_budget/(1024**3))}.csv"
    with open(out_file_name, 'a', newline='') as csvfile:
        for j, data in enumerate(measurements):
            entry = {
                'model_name': model_name,
                'batch_size': batch_size,
                'rep': j,
                'epoch': data['epoch'],
                'time': data['time']*1e3,
                'time_with_model': data['time_with_model']*1e3,
                'total_mem': data['total_mem']/(1024*1024),
                'memory_budget': memory_budget/ (1024*1024),
                'base_compute_time': data['base_compute_time']*1e-6,
                'remat_compute_time': data['remat_compute_time']*1e-6,
                'search_time': data['search_time']*1e-6,
                'cost_time': data['cost_time']*1e-6,
                'remat_count': data['remat_count'],
                'remat_size': data['remat_size']/(1024*1024), 
                'model_mem': data['model_mem']/(1024*1024),
                'input_mem': data['input_mem']/(1024*1024)
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
    memory_budget = int(args.budget *1024*1024)
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
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    train_texts, eval_texts = prepare_data(args.d)
    train_dataset = CustomDataset(train_texts, tokenizer, device=device)
    eval_dataset = CustomDataset(eval_texts, tokenizer, device=device)
    
    train_loader = DataLoader(train_dataset, batch_size=args.b, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=args.b, shuffle=False)
    offline_time = 0
    time_start = time.time()

        
    batch_size = args.b 
    
    time_start = time.time()
    results = train_loop(model_name, train_dataset, args.b, args.e)
    use_dtr = True
    report_results(model_name,results, memory_budget, batch_size)
    time_train = time.time() - time_start
    
    
# python ./train_llm.py -net EleutherAI/pythia-160m -gpu -b 4 -e 1 -dtr -budget 10
# python ./train_llm.py -net EleutherAI/pythia-160m -gpu -b 4 -e 1 -no-recompute

# python ./train_llm.py -net facebook/opt-350m -gpu -b 4 -e 1 -dtr -budget 10
# python ./train_llm.py -net EleutherAI/pythia-160m -gpu -b 4 -e 1 -dtr -budget 10
# python ./train_llm.py -net gpt2 -gpu -b 4 -e 1 -dtr -budget 10