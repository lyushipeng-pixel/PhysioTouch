"""
main_stage1_static.py - 支持静态重建损失的主训练脚本

版本: v2.0
新增功能:
- 支持静态重建损失（单帧图像重建）
- 支持动态重建+预测损失（4帧视频）
- 通过 --static_ratio 参数控制静态/动态比例

参考: AnyTouch - Learning Unified Static-Dynamic Representation
"""

import os
import torch
from transformers import AutoConfig
from transformers.models.vit.configuration_vit import ViTConfig
from model.mae_model import TactileVideoMAE
from config import parse_args
import random
import numpy as np
from dataloader.stage1_dataset import PretrainDataset_Contact
import torch.backends.cudnn as cudnn
# from torch.utils.tensorboard import SummaryWriter  # 已删除TensorBoard
import timm.optim.optim_factory as optim_factory
from Stage1_engine import train_one_epoch  # ⭐ 使用新的训练引擎
import datetime
import json
import time
from pathlib import Path
import copy
import wandb  # ⭐ 添加 Wandb

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

torch.cuda.device_count.cache_clear()


def load_model_from_clip_video(ckpt, model):  
    """从 CLIP 预训练权重初始化触觉模型"""  
    new_ckpt = {}  
    for key, item in ckpt.items():  
        if "vision_model" in key and 'position_ids' not in key:  
            new_key = key.replace("vision_model", "touch_model")  
            new_ckpt[new_key] = item  
      
    msg = model.load_state_dict(new_ckpt, strict=False)  
    print(f"Loaded CLIP weights: {msg}")  
    return model 


def main(args):
    misc.init_distributed_mode(args)
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))  
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # 设置随机种子以确保可重复性
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)  
    cudnn.benchmark = True  

    # ⭐ 打印静态重建配置
    print(f"\n{'='*60}")
    print(f"Static-Dynamic Training Configuration:")
    print(f"  Static ratio: {args.static_ratio:.2%}")
    print(f"  Dynamic ratio: {1-args.static_ratio:.2%}")
    print(f"{'='*60}\n")

    # 创建数据集和数据加载器
    dataset_train = PretrainDataset_Contact(mode='train')
    print(f'Dataset size: {len(dataset_train)}')
    
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print(f"Sampler = {sampler_train}") 

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # 模型初始化
    config = AutoConfig.from_pretrained('CLIP-ViT-L-14-DataComp.XL-s13B-b90K/config.json')
    
    # 配置解码器
    decoder_config = ViTConfig()
    decoder_config.encoder_stride = 14
    decoder_config.hidden_size = 512
    decoder_config.intermediate_size = 2048
    decoder_config.num_attention_heads = 16
    decoder_config.num_hidden_layers = 8
    decoder_config.patch_size = 14

    # 创建模型
    model = TactileVideoMAE(args, config, decoder_config, 1, False, 1)
    model.initialize_decoder()
    
    # 加载预训练权重
    ckpt = torch.load('CLIP-ViT-L-14-DataComp.XL-s13B-b90K/pytorch_model.bin', map_location='cpu')
    model = load_model_from_clip_video(ckpt, model)
    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    # 计算有效批次大小和学习率
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    # 分布式训练包装
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    # 优化器设置
    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(
        param_groups, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.99)
    )
    print(optimizer)
    loss_scaler = NativeScaler()

    # 加载检查点（如果存在）
    misc.load_model(
        args=args, model_without_ddp=model_without_ddp, 
        optimizer=optimizer, loss_scaler=loss_scaler
    )

    print(f"Start training for {args.epochs} epochs")
    print(f"Static reconstruction ratio: {args.static_ratio}")

    # ⭐ 初始化 Wandb（仅主进程）
    if misc.is_main_process():
        # 生成基于时间的 run name
        run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 初始化 wandb
        wandb.init(
            project="PhysioTouch-Stage1",
            name=run_name,
            config={
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "learning_rate": args.lr,
                "warmup_epochs": args.warmup_epochs,
                "mask_ratio": args.mask_ratio,
                "static_ratio": args.static_ratio,
                "accum_iter": args.accum_iter,
                "weight_decay": args.weight_decay,
                "norm_pix_loss": args.norm_pix_loss,
                "use_video": args.use_video,
            }
        )
        print(f"✅ Wandb initialized: Project=PhysioTouch-Stage1, Run={run_name}")
    
    # ⭐ 不使用 TensorBoard
    log_writer = None

    # ⭐ 初始化最佳损失追踪
    best_loss = float('inf')
    best_epoch = -1

    # 训练循环
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        sampler_train.set_epoch(epoch)  
        
        # ⭐ 使用支持静态重建的训练函数
        train_stats = train_one_epoch(  
            model=model,  
            data_loader=data_loader_train,
            optimizer=optimizer,  
            device=device,  
            epoch=epoch,  
            loss_scaler=loss_scaler,  
            args=args  # 包含 static_ratio 参数
        ) 
        
        # ⭐ 新的保存策略：保存最佳(best)和最后(last)
        if args.output_dir and misc.is_main_process():
            # 获取当前epoch的训练损失
            current_loss = train_stats.get('loss', float('inf'))
            
            # 1. 始终保存/更新 last.pth（最后一次训练结果）
            last_checkpoint_path = os.path.join(args.output_dir, 'last.pth')
            torch.save({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
                'loss': current_loss,
            }, last_checkpoint_path)
            print(f'[Epoch {epoch}] Saved last.pth (loss: {current_loss:.4f})')
            
            # 2. 如果是最佳结果，保存/更新 best.pth
            if current_loss < best_loss:
                best_loss = current_loss
                best_epoch = epoch
                best_checkpoint_path = os.path.join(args.output_dir, 'best.pth')
                torch.save({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                    'loss': current_loss,
                }, best_checkpoint_path)
                print(f'⭐ [Epoch {epoch}] New best model! Saved best.pth (loss: {current_loss:.4f})')  

        # 记录日志
        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            'epoch': epoch,
        }

        if args.output_dir and misc.is_main_process():
            # 记录到文本日志
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
            
            # ⭐ 记录 epoch 级别的汇总信息到 Wandb
            wandb.log({
                "epoch": epoch,
                "epoch/train_loss": train_stats.get('loss', 0),
                "epoch/lr": train_stats.get('lr', 0),
                "epoch/best_loss_so_far": best_loss,
            })

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    
    # ⭐ 打印最佳模型信息
    if misc.is_main_process():
        print('\n' + '='*80)
        print('Training Summary:')
        print(f'  Best model at epoch {best_epoch} with loss: {best_loss:.4f}')
        print(f'  Saved models:')
        print(f'    - best.pth  (epoch {best_epoch}, loss {best_loss:.4f})')
        print(f'    - last.pth  (epoch {args.epochs-1})')
        print('='*80)
        
        # ⭐ 记录最终统计到 Wandb
        wandb.log({
            "summary/best_epoch": best_epoch,
            "summary/best_loss": best_loss,
            "summary/total_time_seconds": total_time,
        })
        
        # ⭐ 结束 Wandb 追踪
        wandb.finish()
        print("✅ Wandb logging finished")


if __name__ == "__main__":  
    parser = parse_args()  
    args = parser.parse_args()  
      
    if args.output_dir:  
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)  
      
    main(args)


