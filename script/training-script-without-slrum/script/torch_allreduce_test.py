import torch
import os

def main():
    # 仅获取必要的进程信息
    rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
    world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
    local_rank = rank % 4
    torch.cuda.set_device(local_rank)

    # 直接用TCP初始化（不依赖env变量解析）
    master_addr = os.environ['MASTER_ADDR']
    master_port = os.environ['MASTER_PORT']
    torch.distributed.init_process_group(
        backend='nccl',
        init_method=f'tcp://{master_addr}:{master_port}',
        world_size=world_size,
        rank=rank
    )

    # 最基础的AllReduce（1个元素的张量，最小数据量）
    tensor = torch.tensor([1.0], device=f'cuda:{local_rank}')
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)

    # 只在主进程输出结果
    if rank == 0:
        print(f"AllReduce结果: {tensor.item()}，预期值: {world_size}")
        if abs(tensor.item() - world_size) < 1e-5:
            print("测试通过")
        else:
            print("测试失败")

    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()
