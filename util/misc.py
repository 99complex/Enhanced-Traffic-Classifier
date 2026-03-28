import builtins
import datetime
import os
import time
from collections import defaultdict, deque
from pathlib import Path

import torch
import torch.distributed as dist
from torch import inf


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    """
           log_every
           在迭代过程中记录进度和性能数据。

           参数:
           - iterable: 可迭代对象，如数据加载器。
           - print_freq: 打印频率，每print_freq次迭代打印一次。
           - header: 打印信息的头部字符串（默认为空）。
    """
    def log_every(self, iterable, print_freq, header=None):
        i = 0   # 初始化迭代计数器
        # 如果没有提供header，则header为空字符串
        if not header:
            header = ''
            # 记录开始时间
        start_time = time.time()
        # 记录当前时间，用于计算迭代时间
        end = time.time()
        # 初始化平滑值处理器，用于显示平均迭代时间
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        # 初始化平滑值处理器，用于显示平均数据加载时间
        data_time = SmoothedValue(fmt='{avg:.4f}')
        # 格式化进度信息中的迭代次数显示宽度
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        # 构建日志消息格式
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            '{meters}',
            'eta: {eta}',
            'time: {time}',
            'data: {data}'
        ]
        # 如果CUDA可用，添加显存使用量到日志消息中
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        # 将日志消息格式连接成字符串
        log_msg = self.delimiter.join(log_msg)
        # 定义MB的大小，用于转换显存大小
        MB = 1024.0 * 1024.0
        # 遍历可迭代对象
        for obj in iterable:
            # 更新数据加载时间
            data_time.update(time.time() - end)
            # 返回当前对象，支持迭代器协议
            yield obj
            # 更新迭代时间
            iter_time.update(time.time() - end)
            # 如果达到打印频率或最后一次迭代，打印日志
            if i % print_freq == 0 or i == len(iterable) - 1:
                # 计算剩余时间
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                # 根据CUDA是否可用，打印相应的日志信息
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            # 增加迭代计数器
            i += 1
            # 更新结束时间为当前时间，用于计算下一次迭代的时间
            end = time.time()
        # 计算总时间
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        # 打印总时间和平均迭代时间
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


class NativeScalerWithGradNormCount:
    # 定义一个状态字典键，用于保存AMP缩放器的状态
    state_dict_key = "amp_scaler"

    def __init__(self):
        # 初始化内部的GradScaler对象，用于自动混合精度训练中的梯度缩放
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        """
               调用缩放器以计算和更新梯度。

               参数:
               - loss: 当前批次的损失，将被缩放并反向传播。
               - optimizer: 优化器对象，用于梯度更新。
               - clip_grad: 梯度裁剪的阈值，如果为None，则不进行裁剪。
               - parameters: 需要更新的参数集合，用于计算梯度范数。
               - create_graph: 是否构建计算图，以便于梯度的二次求导。
               - update_grad: 是否执行梯度的更新。

               返回:
               - norm: 参数的梯度范数，如果没有更新梯度，则返回None。
               """
        # 缩放损失并反向传播
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            # 根据是否需要裁剪梯度，更新参数
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer) # 去除梯度的缩放，以便于真实梯度裁剪
                # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)# 更新权重
            self._scaler.update()# 根据梯度更新情况，调整缩放器
        else:
            norm = None
        return norm

    def state_dict(self):
        """
               返回缩放器的状态字典。

               返回:
               - state_dict: 缩放器的状态字典，用于保存和加载状态。
               """
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        """
               加载缩放器的状态字典。

               参数:
               - state_dict: 包含缩放器状态的字典。
               """
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm

# 保存模型
# args: 包含各种配置的参数对象。
# epoch: 当前训练的周期数。
# model: 当前训练的模型。
# model_without_ddp: 没有分布式数据并行包装的模型。
# optimizer: 用于优化模型的优化器。
# loss_scaler: 用于损失函数的缩放器，用于混合精度训练。
# name: 保存的检查点文件的名称，默认为None，如果未指定，则使用epoch的值。
def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, name=None):
    output_dir = Path(args.output_dir)
    # 如果name参数未指定，则将其设置为当前训练周期的字符串表示。
    if name is None:
        name = str(epoch)
    # 判断是否使用了损失缩放器
    if loss_scaler is not None:
        # 如果使用了损失缩放器，创建一个包含保存路径的列表，路径由输出目录和检查点文件名组成。
        checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % name)]
        # output_dir = "./saved_models" name = "epoch_10"  checkpoint_path=./saved_models/checkpoint-epoch_10.pth
        # 遍历所有检查点保存路径。
        for checkpoint_path in checkpoint_paths:
            # 创建一个字典，包含要保存的状态：模型状态、优化器状态、当前周期、缩放器状态和参数。
            to_save = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }
            # 调用save_on_master函数来保存状态字典到指定的检查点路径。这个函数可能是在分布式训练环境中使用的，确保只有主进程执行保存操作。
            save_on_master(to_save, checkpoint_path)
    else:
        client_state = {'epoch': epoch}
        # 调用模型的save_checkpoint方法来保存检查点，包括保存目录、检查点标签和客户端状态。
        # 检查点（Checkpoint）是深度学习训练过程中的一个快照，它记录了模型的参数、优化器的状态和训练周期等信息。当训练过程中断时，可以从最近的检查点恢复，继续训练而不必从头开始。
        model.save_checkpoint(save_dir=args.output_dir, tag="checkpoint-%s" % name, client_state=client_state)


def load_model(args, model_without_ddp, optimizer, loss_scaler):
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        print("Resume checkpoint %s" % args.resume)
        if 'optimizer' in checkpoint and 'epoch' in checkpoint and not (hasattr(args, 'eval') and args.eval):
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")


def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x