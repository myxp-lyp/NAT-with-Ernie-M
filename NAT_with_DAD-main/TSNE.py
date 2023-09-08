#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import argparse
import logging
import math
import os
import sys
from typing import Dict, Optional, Any, List, Tuple, Callable

if "WORLD_SIZE" in os.environ.keys():
    del os.environ["WORLD_SIZE"]

import numpy as np
import torch
from fairseq import (
    checkpoint_utils,
    distributed_utils,
    options,
    quantization_utils,
    tasks,
    utils,
)
from fairseq.data import iterators
from fairseq.dataclass.configs import FairseqConfig
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.distributed_utils import is_master
from fairseq.logging import meters, metrics, progress_bar
from fairseq.model_parallel.megatron_trainer import MegatronTrainer
from fairseq.trainer import Trainer
from omegaconf import DictConfig, OmegaConf

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.train")


def main(cfg: FairseqConfig) -> None:
    if isinstance(cfg, argparse.Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    utils.import_user_module(cfg.common)

    if is_master(cfg.distributed_training) and "job_logging_cfg" in cfg:
        # make hydra logging work with ddp (see # see https://github.com/facebookresearch/hydra/issues/1126)
        logging.config.dictConfig(OmegaConf.to_container(cfg.job_logging_cfg))

    assert (
            cfg.dataset.max_tokens is not None or cfg.dataset.batch_size is not None
    ), "Must specify batch size either with --max-tokens or --batch-size"
    metrics.reset()

    np.random.seed(cfg.common.seed)
    utils.set_torch_seed(cfg.common.seed)

    if distributed_utils.is_master(cfg.distributed_training):
        checkpoint_utils.verify_checkpoint_directory(cfg.checkpoint.save_dir)

    
    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(cfg.task)
    # Load valid dataset (we load training data below, based on the latest checkpoint)
    for valid_sub_split in cfg.dataset.valid_subset.split(","):
        task.load_dataset(valid_sub_split, combine=False, epoch=1)
        
    criterion = task.build_criterion(cfg.criterion)
    #path = "/data/yl7622/NAT-with-Ernie-M/NAT_with_DAD-main/save_new_seed/wmt_dad_CAMLM_1-0/checkpoint_best.pt"
    path = "/data/yl7622/NAT-with-Ernie-M/NAT_with_DAD-main/save_new_seed/wmt_dad-2/checkpoint_best.pt"
    models,_ = checkpoint_utils.load_model_ensemble(utils.split_paths(path), task = task)
    model = models[0]
    encoder = model.encoder.embed_tokens
    #CAMLM validation pure only
   # encoder = model.encoder
   # encoder = checkpoint_utils.load_pretrained_component_from_model(component = encoder, checkpoint = "/data/yl7622/NAT-with-Ernie-M/NAT_with_DAD-main/save_new/CAMLM_wmt/checkpoint_best.pt")
   # encoder = encoder.embed_tokens
    #models1,_ = checkpoint_utils.load_model_ensemble(utils.split_paths(path1), task = task)
    #model1 = models1[0]
    #encoder1 = model1.encoder.embed_tokens
    token_id_src = []
    token_id_tgt = []
    token_src = []
    token_tgt = []
    for i in range(3):
        for j in task.datasets['valid'].src.__getitem__(i):
            if j in token_id_src or task.tgt_dict[j][-1]=='@':
                continue
            token_id_src.append(j)
            token_src.append(task.tgt_dict[j])
    for i in range(3):
        for j in task.datasets['valid'].tgt.__getitem__(i):
            if j in token_id_tgt or task.tgt_dict[j][-1]=='@':
                continue
            token_id_tgt.append(j)
            token_tgt.append(task.tgt_dict[j])    
    token_id_src = torch.LongTensor(token_id_src)
    token_id_tgt = torch.LongTensor(token_id_tgt)
    embedding_src = encoder(token_id_src)
    embedding_tgt = encoder(token_id_tgt)
    #embedding1 = encoder1(token_id)
    tsne = TSNE(n_components=2, random_state=6)
    #token.index('disempowered')
    # (optionally) Configure quantization
    embedding_src = embedding_src.detach().numpy()
    embedding_tgt = embedding_tgt.detach().numpy()
    #embedding1 = embedding1.detach().numpy()
    Y = tsne.fit_transform(embedding_src)
    Y2 = tsne.fit_transform(embedding_tgt)
    
    
    
    for xx in range(len(token_id_src)):
        plt.scatter(Y[xx, 0], Y[xx, 1], color = 'blue')
        #plt.scatter(Y2[xx, 0], Y2[xx, 1], color = 'red')
        plt.annotate(token_src[xx], xy=(Y[xx, 0],Y[xx, 1]), xytext=(0,0), textcoords='offset points')
        #plt.annotate(token[xx], xy=(Y2[xx, 0],Y2[xx, 1]), xytext=(0,0), textcoords='offset points')
    for xx in range(len(token_id_tgt)):
        #plt.scatter(Y[xx, 0], Y[xx, 1], color = 'blue')
        plt.scatter(Y2[xx, 0], Y2[xx, 1], color = 'red')
        #plt.annotate(token_src[xx], xy=(Y[xx, 0],Y[xx, 1]), xytext=(0,0), textcoords='offset points')
        plt.annotate(token_tgt[xx], xy=(Y2[xx, 0],Y2[xx, 1]), xytext=(0,0), textcoords='offset points')
    #plt.savefig('DAD.png')
    #plt.savefig('CAMLM1.png')
    plt.savefig('CAMLM_pure.png')
    plt.show()
    #plt.clf()
    #for xx in token_id:
        #plt.scatter(Y[xx, 0], Y[xx, 1], color = 'blue')
      #  plt.scatter(Y2[xx, 0], Y2[xx, 1], color = 'red')
        #plt.annotate(token[xx], xy=(Y[xx, 0],Y[xx, 1]), xytext=(0,0), textcoords='offset points')
       # plt.annotate(token[xx], xy=(Y2[xx, 0],Y2[xx, 1]), xytext=(0,0), textcoords='offset points')
    #plt.savefig('CAMLM.png')
    #plt.show()
    #plt.xlim((0,4))
    #plt.ylim((0,4))
    #plt.legend()#title = ('DAD','CAMLM'))
    #plt.savefig('Comparison.png')
    #plt.show()
    exit()
    if cfg.common.quantization_config_path is not None:
        quantizer = quantization_utils.Quantizer(
            config_path=cfg.common.quantization_config_path,
            max_epoch=cfg.optimization.max_epoch,
            max_update=cfg.optimization.max_update,
        )
    else:
        quantizer = None

    # Build trainer
    if cfg.common.model_parallel_size == 1:
        trainer = Trainer(cfg, task, model, criterion, quantizer)
    else:
        trainer = MegatronTrainer(cfg, task, model, criterion)

    logger.info(
        "training on {} devices (GPUs/TPUs)".format(
            cfg.distributed_training.distributed_world_size
        )
    )
    logger.info(
        "max tokens per GPU = {} and batch size per GPU = {}".format(
            cfg.dataset.max_tokens,
            cfg.dataset.batch_size,
        )
    )

    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(
        cfg.checkpoint,
        trainer,
        # don't cache epoch iterators for sharded datasets
        disable_iterator_cache=task.has_sharded_data("train"),
    )

    max_epoch = cfg.optimization.max_epoch or math.inf
    lr = trainer.get_lr()
    train_meter = meters.StopwatchMeter()
    train_meter.start()
    while epoch_itr.next_epoch_idx <= max_epoch:
        if lr <= cfg.optimization.stop_min_lr:
            logger.info(
                f"stopping training because current learning rate ({lr}) is smaller "
                "than or equal to minimum learning rate "
                f"(--stop-min-lr={cfg.optimization.stop_min_lr})"
            )
            break
        # print("cfg.checkpoint.best_checkpoint_metric:", cfg.checkpoint.best_checkpoint_metric)
        # exit()
        # train for one epoch

        # 改动
        # valid_subsets = cfg.dataset.valid_subset.split(",")
        # valid_losses = validate(cfg, trainer, task, epoch_itr, valid_subsets)
        # exit()

        valid_losses, should_stop = train(cfg, trainer, task, epoch_itr)
        if should_stop:
            break

        # only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])

        epoch_itr = trainer.get_train_iterator(
            epoch_itr.next_epoch_idx,
            # sharded data: get train iterator for next epoch
            load_dataset=task.has_sharded_data("train"),
            # don't cache epoch iterators for sharded datasets
            disable_iterator_cache=task.has_sharded_data("train"),
        )
    train_meter.stop()
    logger.info("done training in {:.1f} seconds".format(train_meter.sum))


def should_stop_early(cfg: DictConfig, valid_loss: float) -> bool:
    # skip check if no validation was done in the current epoch
    if valid_loss is None:
        return False
    if cfg.checkpoint.patience <= 0:
        return False

    def is_better(a, b):
        return a > b if cfg.checkpoint.maximize_best_checkpoint_metric else a < b

    prev_best = getattr(should_stop_early, "best", None)
    if prev_best is None or is_better(valid_loss, prev_best):
        should_stop_early.best = valid_loss
        should_stop_early.num_runs = 0
        return False
    else:
        should_stop_early.num_runs += 1
        if should_stop_early.num_runs >= cfg.checkpoint.patience:
            logger.info(
                "early stop since valid performance hasn't improved for last {} runs".format(
                    cfg.checkpoint.patience
                )
            )
            return True
        else:
            return False


@metrics.aggregate("train")
def train(
        cfg: DictConfig, trainer: Trainer, task: tasks.FairseqTask, epoch_itr
) -> Tuple[List[Optional[float]], bool]:
    """Train the model for one epoch and return validation losses."""
    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=cfg.distributed_training.fix_batches_to_gpus,
        shuffle=(epoch_itr.next_epoch_idx > cfg.dataset.curriculum),
    )
    update_freq = (
        cfg.optimization.update_freq[epoch_itr.epoch - 1]
        if epoch_itr.epoch <= len(cfg.optimization.update_freq)
        else cfg.optimization.update_freq[-1]
    )
    itr = iterators.GroupedIterator(itr, update_freq)
    if cfg.common.tpu:
        itr = utils.tpu_data_loader(itr)
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        epoch=epoch_itr.epoch,
        tensorboard_logdir=(
            cfg.common.tensorboard_logdir
            if distributed_utils.is_master(cfg.distributed_training)
            else None
        ),
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
        wandb_project=(
            cfg.common.wandb_project
            if distributed_utils.is_master(cfg.distributed_training)
            else None
        ),
        wandb_run_name=os.environ.get(
            "WANDB_NAME", os.path.basename(cfg.checkpoint.save_dir)
        ),
        azureml_logging=(
            cfg.common.azureml_logging
            if distributed_utils.is_master(cfg.distributed_training)
            else False
        ),
    )
    progress.update_config(_flatten_config(cfg))

    trainer.begin_epoch(epoch_itr.epoch)

    valid_subsets = cfg.dataset.valid_subset.split(",")
    should_stop = False
    num_updates = trainer.get_num_updates()
    logger.info("Start iterating over samples")
    for i, samples in enumerate(progress):
        with metrics.aggregate("train_inner"), torch.autograd.profiler.record_function(
                "train_step-%d" % i
        ):
            log_output = trainer.train_step(samples)

        if log_output is not None:  # not OOM, overflow, ...
            # log mid-epoch stats
            num_updates = trainer.get_num_updates()
            if num_updates % cfg.common.log_interval == 0:
                stats = get_training_stats(metrics.get_smoothed_values("train_inner"))
                progress.log(stats, tag="train_inner", step=num_updates)

                # reset mid-epoch stats after each log interval
                # the end-of-epoch stats will still be preserved
                metrics.reset_meters("train_inner")

        end_of_epoch = not itr.has_next()      
        valid_losses, should_stop = validate_and_save(
            cfg, trainer, task, epoch_itr, valid_subsets, end_of_epoch
        )

        if should_stop:
            break

    # log end-of-epoch stats
    logger.info("end of epoch {} (average epoch stats below)".format(epoch_itr.epoch))
    stats = get_training_stats(metrics.get_smoothed_values("train"))
    progress.print(stats, tag="train", step=num_updates)

    # reset epoch-level meters
    metrics.reset_meters("train")
    return valid_losses, should_stop


def _flatten_config(cfg: DictConfig):
    config = OmegaConf.to_container(cfg)
    # remove any legacy Namespaces and replace with a single "args"
    namespace = None
    for k, v in list(config.items()):
        if isinstance(v, argparse.Namespace):
            namespace = v
            del config[k]
    if namespace is not None:
        config["args"] = vars(namespace)
    return config


def validate_and_save(
        cfg: DictConfig,
        trainer: Trainer,
        task: tasks.FairseqTask,
        epoch_itr,
        valid_subsets: List[str],
        end_of_epoch: bool,
) -> Tuple[List[Optional[float]], bool]:
    num_updates = trainer.get_num_updates()
    max_update = cfg.optimization.max_update or math.inf

    # Stopping conditions (and an additional one based on validation loss later
    # on)
    should_stop = False
    if num_updates >= max_update:
        should_stop = True
        logger.info(
            f"Stopping training due to "
            f"num_updates: {num_updates} >= max_update: {max_update}"
        )

    training_time_hours = trainer.cumulative_training_time() / (60 * 60)
    if (
            cfg.optimization.stop_time_hours > 0
            and training_time_hours > cfg.optimization.stop_time_hours
    ):
        should_stop = True
        logger.info(
            f"Stopping training due to "
            f"cumulative_training_time: {training_time_hours} > "
            f"stop_time_hours: {cfg.optimization.stop_time_hours} hour(s)"
        )

    do_save = (
            (end_of_epoch and epoch_itr.epoch % cfg.checkpoint.save_interval == 0)
            or should_stop
            or (
                    cfg.checkpoint.save_interval_updates > 0
                    and num_updates > 0
                    and num_updates % cfg.checkpoint.save_interval_updates == 0
                    and num_updates >= cfg.dataset.validate_after_updates
            )
    )
    do_validate = (
                          (not end_of_epoch and do_save)  # validate during mid-epoch saves
                          or (end_of_epoch and epoch_itr.epoch % cfg.dataset.validate_interval == 0)
                          or should_stop
                          or (
                                  cfg.dataset.validate_interval_updates > 0
                                  and num_updates > 0
                                  and num_updates % cfg.dataset.validate_interval_updates == 0
                          )
                  ) and not cfg.dataset.disable_validation

    # Validate
    valid_losses = [None]
    if do_validate:
        valid_losses = validate(cfg, trainer, task, epoch_itr, valid_subsets)

    should_stop |= should_stop_early(cfg, valid_losses[0])

    # Save checkpoint
    if do_save or should_stop:
        checkpoint_utils.save_checkpoint(
            cfg.checkpoint, trainer, epoch_itr, valid_losses[0]
        )

    return valid_losses, should_stop


def get_training_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    stats["wall"] = round(metrics.get_meter("default", "wall").elapsed_time, 0)
    return stats


def validate(
        cfg: DictConfig,
        trainer: Trainer,
        task: tasks.FairseqTask,
        epoch_itr,
        subsets: List[str],
) -> List[Optional[float]]:
    """Evaluate the model on the validation set(s) and return the losses."""

    if cfg.dataset.fixed_validation_seed is not None:
        # set fixed seed for every validation
        utils.set_torch_seed(cfg.dataset.fixed_validation_seed)

    trainer.begin_valid_epoch(epoch_itr.epoch)
    valid_losses = []
    for subset in subsets:
        logger.info('begin validation on "{}" subset'.format(subset))

        # Initialize data iterator
        itr = trainer.get_valid_iterator(subset).next_epoch_itr(
            shuffle=False, set_dataset_epoch=False  # use a fixed valid set
        )
        if cfg.common.tpu:
            itr = utils.tpu_data_loader(itr)
        progress = progress_bar.progress_bar(
            itr,
            log_format=cfg.common.log_format,
            log_interval=cfg.common.log_interval,
            epoch=epoch_itr.epoch,
            prefix=f"valid on '{subset}' subset",
            tensorboard_logdir=(
                cfg.common.tensorboard_logdir
                if distributed_utils.is_master(cfg.distributed_training)
                else None
            ),
            default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
            wandb_project=(
                cfg.common.wandb_project
                if distributed_utils.is_master(cfg.distributed_training)
                else None
            ),
            wandb_run_name=os.environ.get(
                "WANDB_NAME", os.path.basename(cfg.checkpoint.save_dir)
            ),
        )

        # create a new root metrics aggregator so validation metrics
        # don't pollute other aggregators (e.g., train meters)
        with metrics.aggregate(new_root=True) as agg:
            for sample in progress:
                trainer.valid_step(sample)

        # log validation stats
        stats = get_valid_stats(cfg, trainer, agg.get_smoothed_values())
        # print("stats:", stats)
        # print("valid_losses:", valid_losses)
        # print("cfg.checkpoint.best_checkpoint_metric:", cfg.checkpoint.best_checkpoint_metric)
        progress.print(stats, tag=subset, step=trainer.get_num_updates())

        valid_losses.append(stats[cfg.checkpoint.best_checkpoint_metric])
    return valid_losses


def get_valid_stats(
        cfg: DictConfig, trainer: Trainer, stats: Dict[str, Any]
) -> Dict[str, Any]:
    stats["num_updates"] = trainer.get_num_updates()
    # print("states:", stats)
    if hasattr(checkpoint_utils.save_checkpoint, "best"):
        key = "best_{0}".format(cfg.checkpoint.best_checkpoint_metric)
        best_function = max if cfg.checkpoint.maximize_best_checkpoint_metric else min
        stats[key] = best_function(
            checkpoint_utils.save_checkpoint.best,
            stats[cfg.checkpoint.best_checkpoint_metric],
        )
    return stats


def cli_main(
        modify_parser: Optional[Callable[[argparse.ArgumentParser], None]] = None
) -> None:
    parser = options.get_training_parser()
    parser.add_argument("--generatedict", default=False, help="whether to generate filtered dict")
    args = options.parse_args_and_arch(parser, modify_parser=modify_parser)

    cfg = convert_namespace_to_omegaconf(args)

   
    
    #create filtered dict
    if args.generatedict == "True":
        task = tasks.setup_task(cfg.task)
        for train_sub_split in cfg.dataset.train_subset.split(","):
            task.load_dataset(train_sub_split, combine=False, epoch=1)
        
        filtered_dict = set()
        for i in range(len(task.datasets['train'].tgt)):
            if i % 5000 == 0:
                print("have add "+str(i)+ " samples")
            for j in task.datasets['train'].tgt.__getitem__(i):
                # lbpe filtered dict
                if '_@1@_' in task.tgt_dict.symbols[j]:
                    filtered_dict.add(int(j))
                
                # normal filtered dict
                #filtered_dict.add(int(j))
        
        with open ("/data/yl7622/NAT-with-Ernie-M/NAT_with_DAD-main/filtered_dict_new/train_deen_mul_correct_lbpe_1.dict", 'w') as f:
            f.write('[')
            judge =  0
            for i in filtered_dict:
                if judge == 0:
                    f.write(str(i))
                    judge = 1
                f.write(', ' + str(i))
            f.write(']')
        exit()
        
    if args.profile:
        with torch.cuda.profiler.profile():
            with torch.autograd.profiler.emit_nvtx():
                distributed_utils.call_main(cfg, main)
    else:
        distributed_utils.call_main(cfg, main)


if __name__ == "__main__":
    cli_main()
