import glob
import os
import random
import math
import time
import argparse
from queue import Queue as LocalQueue # contrast with mp.Queue
import multiprocessing as mp
import csv
from common import invoke_main, read_json, write_json, prepare_out_file, check_file_exists
from utils_functions import get_network, get_training_dataloader
from conf import settings


def report_results(model_name, i, out_file, use_dtr, measurements, memory_budget, batch_size):
    if not measurements:
        return  # nothing to write
    out_file_name = f"{model_name}_{batch_size}_{int(memory_budget/(1024*1024))}.csv"
    with open(out_file_name, 'a', newline='') as csvfile:
        # https://github.com/uwsampl/dtr-prototype/blob/master/dtr_code/shared/run_torch_trial.py
        for j, data in enumerate(measurements):
            entry = {
                'model_name': model_name,
                'batch_size': batch_size,
                'rep': j,
                'epoch': data['epoch'],
                'time': data['time']*1e3,
                'time_with_model':  data['time_with_model']*1e3,
                'sync_time': data['sync_time']*1e3,
                'gpu_time': float(data['gpu_time']),
                'total_mem': data['total_mem']/ (1024*1024),
                'memory_budget': memory_budget,
                'base_compute_time': data['base_compute_time']*1e-6,
                'remat_compute_time': data['remat_compute_time']*1e-6,
                'search_time': data['search_time']*1e-6,
                'cost_time': data['cost_time']*1e-6, 
                'remat_count': data['remat_count'], 
                'remat_size': data['remat_size']/ (1024*1024),
                'input_mem': data['input_mem']/ (1024*1024),
                'model_mem': data['model_mem']/ (1024*1024)
            }

            # Create writer dynamically based on entry keys
            writer = csv.DictWriter(csvfile, fieldnames=entry.keys())

            # Write header only if file is empty
            if csvfile.tell() == 0:
                writer.writeheader()

            writer.writerow(entry)

# TODO: Since this will be a separate process, we don't need
# run_torch_trial as a whole to be a separate script anymore
def run_measurements( i, model_name,
                     n_reps, use_dtr,
                     results_queue, epochs, batch_size, memory_budget):
    """
    Sets up PyTorch and runs the specified number of measurements,
    placing them in the given queue.

    Handles all PT setup so it can be spun off into a separate process
    """
    import torch
    import gc

    import math
    import signal
    from tqdm import tqdm
    import model_util


    def run_single_measurement(model_name, produce_model, run_model, teardown, inp,
                               criterion,  use_dtr,  e, batch_size, memory_budget):
        """
        This function initializes a model and performs
        a single measurement of the model on the given input.

        While it might seem most reasonable to initialize
        the model outside of the loop, DTR's logs have shown
        that certain constants in the model persist between loop iterations;
        performing these actions in a separate *function scope* turned out to be the only
        way to prevent having those constants hang around.

        Returns a dict of measurements
        """
        torch.cuda.reset_max_memory_allocated()
        remate_count_previous = torch.remat_compute_count() 
        remate_size_previous = torch.remat_compute_size()
        # resetting means the count should be reset to
        # only what's in scope, meaning only the input
        start_time_model = time.time()
        input_mem = torch.cuda.max_memory_allocated()
'''
        This function initializes a model and performs
        a single measurement of the model on the given input.

        While it might seem most reasonable to initialize
        the model outside of the loop, DTR's logs have shown
        that certain constants in the model persist between loop iterations;
        performing these actions in a separate function scope turned out to be the only
        way to prevent having those constants hang around.
        https://github.com/uwsampl/dtr-prototype/blob/eff53cc4804cc7d6246a6e5086861ce2b846f62b/dtr_code/shared/run_torch_trial.py#L178C1-L185C59
'''
        model = produce_model(extra_params=[])
        params = []
        for m in model:
            if hasattr(m, 'parameters'):
                params.extend(m.parameters())

        model_mem = torch.cuda.max_memory_allocated()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        # start timing
        torch.cuda.synchronize()
        start_time = time.time()
        if use_dtr:
            torch.reset_profile()
        start.record()
        # with torch.autograd.profiler.profile(use_cuda=True) as prof:
        run_model(criterion, *model, *inp)
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

    def timing_loop(model_name, i,  n_reps,
                    use_dtr,
                    results_queue, batch_size):
        measurements = []

        if use_dtr:
            torch.toggle_log(False)

        print(f"training model {model_name} with batch size {batch_size}")
        produce_model, run_model, teardown = model_util.prepare_model(model_name,
                                                                                    batch_size,
                                                                                use_dtr=use_dtr)
        
        criterion = model_util.get_criterion(model_name)

        training_loader, num_classes_train = get_training_dataloader(
            settings.CIFAR100_TRAIN_MEAN,
            settings.CIFAR100_TRAIN_STD,
            num_workers=0,
            batch_size=batch_size,
            shuffle=True,
            name='cifar100'
        )
        start_total = time.time()
        for e in range(epochs):
            iter_limit = -1
            num_iters = min(len(training_loader), iter_limit) if iter_limit != -1 else len(training_loader)
            progress =  tqdm(total=num_iters, desc=f"Training Epoch {e}", leave=False)
            for j, (inputs, labels) in enumerate(training_loader):
                gc.collect()
                inputs = inputs.cuda(0)
                labels = labels.cuda(0)
                inp = (inputs, labels)
                res = run_single_measurement(model_name, produce_model, run_model, teardown,
                                                inp, criterion,  use_dtr,e, batch_size, memory_budget)
                
                results_queue.append(res)
                progress.update(1)
    
                if j >= num_iters-1: 
                    break
        finish_time_total = time.time() - start_total
        file_name = f"cnn_models_time.txt"

        # Open the file in append mode and save the time for this epoch
        with open(file_name, 'a') as file:
            file.write(f"Model:  {model_name}: {finish_time_total:.2f}s\n")
        torch.cuda.empty_cache()
        return results_queue


    if use_dtr:
        if memory_budget > 0:
            print(f'Setting budget to {memory_budget}')
            torch.set_memory_budget(memory_budget)

    measurements1 = timing_loop(model_name, i, n_reps,
                 use_dtr,
                results_queue, batch_size)
    return measurements1

def invoke_main(main_fn, *arg_names):
    parser = argparse.ArgumentParser()

    for name in arg_names:
        parser.add_argument(f"--{name}")

    args = parser.parse_args()

    values = [getattr(args, name) for name in arg_names]

    main_fn(*values)


def main(experiment_mode, model_name, input_idx, out_file, epochs, batch_size,memory_budget):
    print(model_name, out_file )
    epochs = int(epochs)
    use_dtr = (experiment_mode == 'dtr')
    print("memory budget is: ")
    print(float(memory_budget))
    i = 1
    memory_budget = int(float(memory_budget) *1024*1024)
    cwd = os.getcwd()
    batch_size = int(batch_size)
    training_loader, num_classes_train = get_training_dataloader(
    settings.CIFAR100_TRAIN_MEAN,
    settings.CIFAR100_TRAIN_STD,
    num_workers=0,
    batch_size=int(batch_size),
    shuffle=True,
    name='cifar100'
    )
    n_reps = len(training_loader)
    print(f"number of resulst {n_reps}")
    measurements = [] 
    remaining_reps = n_reps
    # def run_measurements( i, model_name,
    #                  n_reps, use_dtr,
    #                  results_queue, epochs):
    measurements1 = run_measurements(i, model_name, 
                     remaining_reps, use_dtr, measurements, epochs, batch_size, memory_budget)
    report_results(model_name, i,
                   out_file, use_dtr, measurements1, memory_budget, batch_size)

if __name__ == '__main__':
    invoke_main(main, 'experiment_mode',
                'model_name', 'input_idx',
                'out_file', 'epochs', 'batch_size', "memory_budget")
# python run_cnn.py --experiment_mode dtr --model_name resnet18  --input_idx 0 --out_file output.json --epochs 1 --batch_size 128 --memory_budget 500