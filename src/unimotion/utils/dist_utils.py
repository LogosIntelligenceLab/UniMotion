import os
import tempfile
import shutil
import pickle
import torch
from torch import distributed as dist


def collect_results_cpu(result_part):
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        return result_part
    # create a tmp dir if it is not specified
    MAX_LEN = 512
    # 32 is whitespace
    dir_tensor = torch.full((MAX_LEN, ),
                            32,
                            dtype=torch.uint8,
                            device='cuda')
    if rank == 0:
        os.makedirs('.dist_test', exist_ok=True)
        tmpdir = tempfile.mkdtemp(dir='.dist_test')
        tmpdir = torch.tensor(
            bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
        dir_tensor[:len(tmpdir)] = tmpdir
    dist.broadcast(dir_tensor, 0)
    tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    # dump the part result to the dir
    with open(os.path.join(tmpdir, f'part_{rank}.pkl'), 'wb') as f:
        pickle.dump(result_part, f)
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = os.path.join(tmpdir, f'part_{i}.pkl')
            with open(part_file, 'rb') as f:
                part_list.append(pickle.load(f))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results