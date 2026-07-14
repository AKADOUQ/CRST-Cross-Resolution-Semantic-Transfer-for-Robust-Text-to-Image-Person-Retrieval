import logging
import time
import torch
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.comm import get_rank, synchronize
from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable
import torch.nn as nn

def do_train(start_epoch, args, model, train_loader, evaluator, optimizer,
             scheduler, checkpointer):

    log_period = args.log_period
    eval_period = args.eval_period
    device = "cuda"
    num_epoch = args.num_epoch
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0

    logger = logging.getLogger("CRST.train")
    logger.info('start training')

    meters = {
        "loss": AverageMeter(),
        "sdm_loss": AverageMeter(),
        "itc_loss": AverageMeter(),
        "id_loss": AverageMeter(),
        "mlm_loss": AverageMeter(),
        "rcr_loss": AverageMeter(),
        "feat_loss": AverageMeter(),
        "cr_rda_loss": AverageMeter(),
        # backward-compatible names from old logs
        "sr_loss": AverageMeter(),
        "cr_sdm_loss": AverageMeter(),
        "img_acc": AverageMeter(),
        "txt_acc": AverageMeter(),
        "mlm_acc": AverageMeter(),
        "rcr_gate_mean": AverageMeter(),
    }

    tb_writer = SummaryWriter(log_dir=args.output_dir)

    best_top1 = 0.0

    # train
    for epoch in range(start_epoch, num_epoch + 1):
        start_time = time.time()
        for meter in meters.values():
            meter.reset()
        model.train()

        for n_iter, batch in enumerate(train_loader):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

            ret = model(batch)
            total_loss = sum([v for k, v in ret.items() if "loss" in k])

            batch_size = batch['images'].shape[0]
            meters['loss'].update(total_loss.item(), batch_size)
            
            if 'sdm_loss' in ret: meters['sdm_loss'].update(ret['sdm_loss'].item(), batch_size)
            if 'itc_loss' in ret: meters['itc_loss'].update(ret['itc_loss'].item(), batch_size)
            if 'id_loss' in ret: meters['id_loss'].update(ret['id_loss'].item(), batch_size)
            if 'mlm_loss' in ret: meters['mlm_loss'].update(ret['mlm_loss'].item(), batch_size)
            if 'rcr_loss' in ret: meters['rcr_loss'].update(ret['rcr_loss'].item(), batch_size)
            if 'feat_loss' in ret: meters['feat_loss'].update(ret['feat_loss'].item(), batch_size)
            if 'cr_rda_loss' in ret: meters['cr_rda_loss'].update(ret['cr_rda_loss'].item(), batch_size)
            if 'sr_loss' in ret: meters['sr_loss'].update(ret['sr_loss'].item(), batch_size)
            if 'cr_sdm_loss' in ret: meters['cr_sdm_loss'].update(ret['cr_sdm_loss'].item(), batch_size)

            if 'img_acc' in ret: meters['img_acc'].update(ret['img_acc'].item(), batch_size)
            if 'txt_acc' in ret: meters['txt_acc'].update(ret['txt_acc'].item(), batch_size)
            if 'mlm_acc' in ret: meters['mlm_acc'].update(ret['mlm_acc'].item(), 1)
            if 'rcr_gate_mean' in ret: meters['rcr_gate_mean'].update(ret['rcr_gate_mean'].item(), batch_size)

            optimizer.zero_grad()
            total_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            
            optimizer.step()
            synchronize()

            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                for k, v in meters.items():
                    if v.avg > 0:
                        info_str += f", {k}: {v.avg:.4f}"
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)
        
        tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        tb_writer.add_scalar('temperature', ret.get('temperature', 0), epoch)
        for k, v in meters.items():
            if v.avg > 0:
                tb_writer.add_scalar(k, v.avg, epoch)


        scheduler.step()
        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch,
                        train_loader.batch_size / time_per_batch))
        if epoch % eval_period == 0:
            if get_rank() == 0:
                logger.info("Validation Results - Epoch: {}".format(epoch))
                if args.distributed:
                    top1 = evaluator.eval(model.module.eval())
                else:
                    top1 = evaluator.eval(model.eval())

                torch.cuda.empty_cache()
                if best_top1 < top1:
                    best_top1 = top1
                    arguments["epoch"] = epoch
                    checkpointer.save("best", **arguments)
    if get_rank() == 0:
        logger.info(f"best R1: {best_top1} at epoch {arguments['epoch']}")

import torch
import logging

def benchmark_time_ms(model, test_img_loader, test_txt_loader,
                      warmup=50, iters=200):
    logger = logging.getLogger("CRST.benchmark")
    device = next(model.parameters()).device
    model.eval()

    img_batch = next(iter(test_img_loader))
    txt_batch = next(iter(test_txt_loader))

    images = img_batch["images"].to(device)
    res_label = img_batch.get("res_label", None)
    if res_label is None:
        res_label = torch.zeros(images.size(0), dtype=torch.long, device=device)
    else:
        res_label = res_label.to(device)

    caption_ids = txt_batch["caption_ids"].to(device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model.encode_image(images, res_label)
            _ = model.encode_text(caption_ids)
        torch.cuda.synchronize()

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

        starter.record()
        for _ in range(iters):
            _ = model.encode_image(images, res_label)
            _ = model.encode_text(caption_ids)
        ender.record()

        torch.cuda.synchronize()
        total_ms = starter.elapsed_time(ender) / iters

    bs = images.size(0)
    per_sample_ms = total_ms / bs

    logger.info(f"[Time(ms)] batch={bs}, total={total_ms:.3f} ms/iter, per-sample={per_sample_ms:.3f} ms")
    return per_sample_ms


def do_inference(model, test_img_loader, test_txt_loader):
    logger = logging.getLogger("CRST.test")
    logger.info("Enter inferencing")

    benchmark_time_ms(model, test_img_loader, test_txt_loader)

    evaluator = Evaluator(test_img_loader, test_txt_loader)
    top1 = evaluator.eval(model.eval())
