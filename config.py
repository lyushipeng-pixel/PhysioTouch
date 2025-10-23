import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--output_dir', default='output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)

    parser.add_argument('--mask_ratio', type=float, default=0.75, help='')
    parser.add_argument("--norm_pix_loss", action='store_true', help="")
    parser.set_defaults(norm_pix_loss=False)
    parser.add_argument("--use_video", action='store_true', help="")
    parser.add_argument("--use_sensor_token", action='store_true', help="")
    parser.add_argument("--use_same_patchemb", action='store_true', help="")
    parser.add_argument("--sensor_token_for_all", action='store_true', help="")
    parser.add_argument("--beta_start", type=float, default=0.0, help="")
    parser.add_argument("--beta_end", type=float, default=0.75, help="")
    parser.add_argument("--new_decoder_sensor_token", action='store_true', help="")
    parser.add_argument("--alpha_vl", type=float, default=0.2, help="")
    parser.add_argument("--alpha_vt", type=float, default=0.2, help="")
    parser.add_argument("--alpha_lt", type=float, default=1.0, help="")
    parser.add_argument("--TAG_times", type=int, default=1, help="")
    parser.add_argument("--cross_iter", type=int, default=6, help="")
    parser.add_argument("--cross_alpha", type=float, default=1.0, help="")
    parser.add_argument("--no_mae", action='store_true', help="")

    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--num_workers', type=int, default=32, metavar='N',
                        help='')
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--mae_dir', default=None,
                        help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--init_temp', type=float, default=0.07,
                        help='init_temp')
    parser.add_argument('--save_freq', default=1, type=int,  
                    help='save checkpoint every N epochs')
                        

    ############################
    # LoRA
    parser.add_argument("--convert_to_lora", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument('--lora_r', type=int, default=16, help='')
    parser.add_argument('--lora_alpha', type=int, default=16, help='')
    parser.add_argument('--lora_dropout', type=float, default=0.0, help='')

    parser.add_argument(
        "--seed", type=int, default=0, help="Default random seed."
    )

    # distributed training parameters
    parser.add_argument("--distributed", action='store_true', help="")
    parser.add_argument('--world_size', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local-rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')


    return parser