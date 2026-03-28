import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy

import util.misc as misc
import util.lr_sched as lr_sched

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

import matplotlib.pyplot as plt
import numpy as np

# 训练并保存模型
def pretrain_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, epochs: int,
                    log_writer=None,
                    model_without_ddp=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('steps', misc.SmoothedValue(window_size=1, fmt='{value:.0f}'))
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    steps_of_one_epoch = len(data_loader)

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        steps = steps_of_one_epoch * epoch + data_iter_step  #目前训练步数，即目前训练的总batch数
        metric_logger.update(steps=int(steps))

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate_pretrain(optimizer, data_iter_step / len(data_loader) + epoch, epochs, args)

        samples = samples.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            loss, _, _ ,_= model(samples, mask_ratio=args.mask_ratio)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
        # 检查当前的训练步数是否是save_steps_freq的倍数，这意味着模型将每隔save_steps_freq步保存一次
        # 例如，如果 save_steps_freq 设置为 100，那么模型将在每训练 100 个批次（steps）后保存一次。
        # steps 的值通常通过以下方式计算：steps = steps_of_one_epoch * epoch + data_iter_step，其中 steps_of_one_epoch 是一个训练周期中批次的总数，epoch 是当前的训练周期数，data_iter_step 是当前周期内处理的批次索引。
        if args.output_dir and steps % args.save_steps_freq == 0 and epoch > 0:
            # 保存模型
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, name='step'+str(steps))


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    # 将模型设置为训练模式
    model.train(True)

    # 初始化用于记录和报告训练状态的工具
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # 设置打印的头部信息
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    # 累积梯度的迭代次数
    accum_iter = args.accum_iter

    optimizer.zero_grad()

    # 如果提供了日志写入器，打印其目录
    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    # 初始化用于存储所有预测和目标的列表
    pred_all = []
    target_all = []

    # 遍历数据加载器中的每个batch,log_every函数在迭代过程中记录进度和性能数据。
    #data_loader 和 iterable 的关系是：data_loader 是传递给 log_every 函数的参数，它是一个可迭代对象，而 iterable 是 log_every 函数内部使用的变量，表示这个可迭代对象。
    #例如，如果 data_loader 有35个元素，这里的元素是指一个batch的数量。print_freq 设置为10，那么在第10、20、30次迭代时（以及最后一次迭代，即第35次），函数会打印出日志信息。在这些日志信息中，
    # 迭代次数会显示为 [10/35]、[20/35]、[30/35] 和 [34/35]（注意这里的索引是从0开始的，所以第35次迭代显示为 [34/35]）。
    #以IOTorIOP数据集为例子 batchsize为64  总数量为35479  则会分成35479%64=554  所以会显示[  0/554] 然后开始迭代到[553/554]
    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # 使用每迭代一次的学习率调度器
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # 将数据移动到指定的设备
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # 如果提供了mixup函数，对数据进行mixup处理
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        # 使用自动混合精度进行前向传播和计算损失
        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)

        # 获取预测结果，并将其添加到总预测列表中
        _, pred = outputs.topk(1, 1, True, True)
        pred = pred.t()
        pred_all.extend(pred[0].cpu())
        target_all.extend(targets.cpu())

        # 获取损失值，并检查其是否为有限数
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # 对损失进行平均，并使用梯度缩放进行反向传播
        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        # 同步所有GPU上的操作
        torch.cuda.synchronize()

        # 更新训练指标
        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        # 计算所有进程间的损失平均值
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # 在所有进程中同步和汇总统计信息
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


"""
    评估模型在给定数据集上的性能。

    参数：
    - data_loader: 数据加载器，用于加载评估数据。
    - model: 需要评估的模型。
    - device: 模型和数据所在的设备（如'cuda'或'cpu'）。

    返回：
    - test_state: 字典类型，包含评估结果的各种指标，如准确率、混淆矩阵等。
    """
@torch.no_grad()
def evaluate(data_loader, model, device):
    # 定义交叉熵损失函数
    criterion = torch.nn.CrossEntropyLoss()

    # 初始化用于记录和计算性能指标的工具
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    # 将模型切换到评估模式
    model.eval()

    pred_all = []
    target_all = []

    # 初始化用于存储所有预测和目标标签的列表
    for batch in metric_logger.log_every(data_loader, 10, header):
        # 从批次中获取图像和目标标签，并将它们移动到指定的设备上
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # 计算模型输出和损失
        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        # 获取预测结果，即最高得分的类别索引
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()

        # 将预测结果和目标标签添加到相应的列表中
        pred_all.extend(pred[0].cpu())
        target_all.extend(target.cpu())

        # 计算top-1和top-2准确率
        # topk=(1, 5)：这个参数指定了计算准确率时要考虑到的前 k 个最高得分或概率的类别。在这个例子中，(1, 5) 表示将分别计算 top-1 和 top-5 的准确率：
        # 在多分类问题中，尤其是在类别众多且某些类别样本较少的情况下，仅仅使用 top-1 准确率可能不足以全面评估模型的性能。因此，top-5 或 top-k 准确率提供了一个更宽松的性能度量，允许模型有更多的机会预测正确的类别。
        # acc1, acc5 = accuracy(output, target, topk=(1, 5))
        # 由于此次数据集是二分类 所以2<5 不能计算前五的acc 所以要设为 topk=(1, 2)
        acc1, acc5 = accuracy(output, target, topk=(1, 2))

        # 更新性能记录器中的损失和准确率
        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item()/100, n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item()/100, n=batch_size)

    # 计算加权平均的精确率、召回率和F1分数
    macro = precision_recall_fscore_support(target_all, pred_all, average='weighted')
    # 计算混淆矩阵
    cm = confusion_matrix(target_all, pred_all)

    # gather the stats from all processes
    # 在所有进程中同步性能统计信息
    metric_logger.synchronize_between_processes()
    # 打印总体评估结果
    print('* Acc@1 {top1.global_avg:.4f} Acc@5 {top5.global_avg:.4f} loss {losses.global_avg:.4f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    print(
        '* Pre {macro_pre:.4f} Rec {macro_rec:.4f} F1 {macro_f1:.4f}'
        .format(macro_pre=macro[0], macro_rec=macro[1],
                    macro_f1=macro[2]))

    test_state = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    test_state['macro_pre'] = macro[0]
    test_state['macro_rec'] = macro[1]
    test_state['macro_f1'] = macro[2]
    test_state['cm'] = cm

    return test_state