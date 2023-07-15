# -*- coding:utf-8 -*-
# create: 2023/6/2
import os
import re
import time

import munch
import torch
from PIL import Image
from transformers import AutoTokenizer, DonutProcessor, VisionEncoderDecoderModel, \
    VisionEncoderDecoderConfig, DonutImageProcessor, AutoConfig, AutoModel

from base.common_util import get_absolute_file_path
from base.driver import logger
from base.meter import AverageMeter
from base.torch_utils.dl_util import get_optimizer
from models.configuration_docparser import DocParserConfig
from models.modeling_docparser import DocParserModel
from mydatasets import get_dataset
from .base_experiment import BaseExperiment


class DocParserExperiment(BaseExperiment):

    def __init__(self, config):
        config = self._init_config(config)
        self.experiment_name = config["name"]
        self.args = munch.munchify(config)
        self.init_device(config)
        self.init_random_seed(config)
        self.init_model(config)
        self.init_dataset(config)
        self.init_trainer_args(config)
        self.init_predictor_args(config)
        self.prepare_accelerator()

    """
        Main Block
    """

    def predict(self, **kwargs):
        for img_path in self.args.predictor.img_paths:
            image = Image.open(img_path)
            if not image.mode == "RGB":
                image = image.convert('RGB')

            pixel_values = self.processor(image, return_tensors="pt").pixel_values
            # prepare decoder inputs
            task_prompt = self.args.datasets.train.dataset.task_start_token
            decoder_input_ids = self.processor.tokenizer(task_prompt, add_special_tokens=False,
                                                         return_tensors="pt").input_ids
            start = time.time()
            with torch.no_grad():
                outputs = self.model.generate(
                    pixel_values.to(self.args.device.device_id),
                    decoder_input_ids=decoder_input_ids.to(self.args.device.device_id),
                    max_length=self.model.decoder.config.max_length,
                    early_stopping=True,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=True,
                    num_beams=1,
                    bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                    return_dict_in_generate=True,
                )
            sequence = self.processor.batch_decode(outputs.sequences)[0]
            batch_time = time.time() - start
            logger.info("batch inference time:{} s".format(batch_time))
            sequence = sequence.replace(self.processor.tokenizer.eos_token, "").replace(
                self.processor.tokenizer.pad_token, "")
            sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
            print(self.processor.token2json(sequence))

    def train(self, **kwargs):
        batch_time = AverageMeter()
        loss_meter = AverageMeter()
        norm_meter = AverageMeter()
        global_step = self.args.trainer.start_epoch * len(self.train_data_loader)
        global_eval_step = 0
        ni = 0
        for epoch in range(self.args.trainer.start_epoch, self.args.trainer.epochs):
            self.optimizer.zero_grad()
            for i, batch in enumerate(self.train_data_loader):
                if global_step < self.args.trainer.start_global_step:
                    global_step += 1
                    continue
                start = time.time()
                self.model.train()
                ni = i + len(self.train_data_loader) * epoch  # number integrated batches (since train start)
                with self.gradient_accumulate_scope(self.model):
                    result = self._step_forward(batch)
                    self._step_backward(result.loss)
                    if self.accelerator is not None or ((i + 1) % self.args.trainer.grad_accumulate
                                                        == 0) or ((i + 1) == len(self.train_data_loader)):
                        grad_norm = self._step_optimizer()
                        norm_meter.update(grad_norm)
                        if not self.args.trainer.scheduler_by_epoch:
                            self._step_scheduler(global_step)
                loss_meter.update(result['loss'].item(), self.args.datasets.train.batch_size)
                batch_time.update(time.time() - start)
                global_step += 1
                global_eval_step = self._print_step_log(epoch, global_step, global_eval_step, loss_meter, norm_meter,
                                                        batch_time, ni)
            if self.args.trainer.scheduler_by_epoch:
                self._step_scheduler(global_step)
            global_eval_step = self._print_epoch_log(epoch, global_step, global_eval_step, loss_meter, ni)
        model_config_path = self._train_post_process()
        if self.args.device.is_master:
            self.writer.close()
        return {
            'acc': self.args.trainer.best_eval_result,
            'best_model_path': self.args.trainer.best_model_path,
            'model_config_path': model_config_path,
        }

    def _step_forward(self, batch, is_train=True, eval_model=None, **kwargs):
        input_args_list = ['pixel_values', 'labels', 'decoder_input_ids']
        batch = {k: v.to(self.args.device.device_id) for k, v in batch.items() if k in input_args_list}
        # Runs the forward pass with auto-casting.
        with self.precision_scope:
            output = self.model(**batch)
        return output

    """
        Initialization Functions
    """

    def init_model(self, config):
        model_args = config["model"]
        tokenizer_args = model_args["tokenizer_args"]
        # we can borrow donut tokenizer & processor for docparser
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=tokenizer_args['pretrained_model_name_or_path']
        )
        image_processor = DonutImageProcessor(
            size={"height": model_args['image_size'][0], "width": model_args['image_size'][1]})
        self.processor = DonutProcessor(image_processor=image_processor,
                                        tokenizer=tokenizer)

        # model initialization
        AutoConfig.register("docparser-swin", DocParserConfig)
        AutoModel.register(DocParserConfig, DocParserModel)
        config = VisionEncoderDecoderConfig.from_pretrained(model_args["pretrained_model_name_or_path"])
        config.encoder.image_size = model_args['image_size']
        # during pre-training, a larger image size was used; for fine-tuning,
        # we update max_length of the decoder (for generation)
        config.decoder.max_length = model_args['max_length']
        model = VisionEncoderDecoderModel(config=config)
        logger.info("init weight from pretrained model:{}".format(model_args["pretrained_model_name_or_path"]))
        model.decoder.resize_token_embeddings(len(self.processor.tokenizer))
        self.model = model
        self.model.to(self.args.device.device_id)
        if "model_path" in model_args and model_args['model_path'] is not None:
            model_path = get_absolute_file_path(model_args['model_path'])
            self.load_model(model_path, strict=model_args.get('load_strict', True))
        total = sum([param.nelement() for param in self.model.parameters()])
        logger.info("Number of parameter: %.2fM" % (total / 1e6))

    def _init_optimizer(self, trainer_args, **kwargs):
        optimizer_args = trainer_args.get("optimizer")
        if optimizer_args.get("scale_lr"):
            num_process = 1 if self.accelerator is None else self.accelerator.num_processes
            optimizer_args['lr'] = float(optimizer_args['lr']) * self.grad_accumulate * \
                                   self.train_data_loader.batch_size * num_process
            optimizer_args['img_lr'] = float(optimizer_args['img_lr']) * self.grad_accumulate * \
                                       self.train_data_loader.batch_size * num_process
        self.optimizer = get_optimizer(self.model, **optimizer_args)

    def init_dataset(self, config):
        if 'datasets' in config and config.get('phase', 'train') != 'predict':
            dataset_args = config.get("datasets")
            train_data_loader_args = dataset_args.get("train")
            if config.get('phase', 'train') == 'train':
                train_data_loader_args['dataset'].update({
                    "donut_model": self.model,
                    "processor": self.processor,
                    "max_length": config['model']['max_length'],
                    "phase": 'train',
                })
                if "cache_dir" not in train_data_loader_args['dataset']:
                    train_data_loader_args['dataset'].update({
                        "cache_dir": config['trainer']['save_dir']})
                self.train_dataset = get_dataset(train_data_loader_args['dataset'])
                self.train_data_loader = self._get_data_loader_from_dataset(self.train_dataset,
                                                                            train_data_loader_args,
                                                                            phase='train')
                logger.info("success init train data loader len:{} ".format(len(self.train_data_loader)))

            # set task start token & pad token for bart decoder;
            # Do NOT change it since you can only set the start_token after dataset initialization where special tokens
            # are added into vocab
            self.model.config.decoder_start_token_id = self.processor.tokenizer.convert_tokens_to_ids(
                train_data_loader_args['dataset']['task_start_token'])
            self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id

    """
        Tool Functions
    """

    def _print_step_log(self, epoch, global_step, global_eval_step, loss_meter, norm_meter, batch_time, ni, **kwargs):
        current_lr = self._get_current_lr(ni, global_step)
        if self.args.device.is_master and self.args.trainer.print_freq > 0 and global_step % self.args.trainer.print_freq == 0:
            message = "experiment:{}; train, (epoch: {}, steps: {}, lr:{:e}, step_mean_loss:{}," \
                      " average_loss:{}), time, (train_step_time: {:.5f}s, train_average_time: {:.5f}s);" \
                      "(grad_norm_mean: {:.5f}, grad_norm_step: {:.5f})". \
                format(self.experiment_name, epoch, global_step, current_lr,
                       loss_meter.val, loss_meter.avg, batch_time.val, batch_time.avg, norm_meter.avg,
                       norm_meter.val)
            logger.info(message)
            if self.writer is not None:
                self.writer.add_scalar("{}_train/lr".format(self.experiment_name), current_lr, global_step)
                self.writer.add_scalar("{}_train/step_loss".format(self.experiment_name), loss_meter.val, global_step)
                self.writer.add_scalar("{}_train/average_loss".format(self.experiment_name), loss_meter.avg,
                                       global_step)
        if global_step > 0 and self.args.trainer.save_step_freq > 0 and self.args.device.is_master and global_step % self.args.trainer.save_step_freq == 0:
            message = "experiment:{}; eval, (epoch: {}, steps: {});".format(self.experiment_name, epoch, global_step)
            logger.info(message)
            # result = self.evaluate(global_eval_step=global_eval_step)
            checkpoint_name = "{}_epoch{}_step{}_lr{:e}_average_loss{:.5f}.pth".format(
                self.experiment_name, epoch, global_step, current_lr, loss_meter.avg)
            checkpoint_path = os.path.join(self.args.trainer.save_dir, checkpoint_name)
            tokenizer_path = os.path.join(self.args.trainer.save_dir, "tokenizer")
            os.makedirs(tokenizer_path, exist_ok=True)
            self.processor.tokenizer.save_pretrained(tokenizer_path)
            self.save_model(checkpoint_path, epoch=epoch, global_step=global_step, loss=loss_meter.val)
        return global_eval_step

    def _print_epoch_log(self, epoch, global_step, global_eval_step, loss_meter, ni, **kwargs):
        current_lr = self._get_current_lr(ni, global_step)
        if self.args.trainer.save_epoch_freq > 0 and self.args.device.is_master and epoch % self.args.trainer.save_epoch_freq == 0:
            message = "experiment:{}; eval, (epoch: {}, steps: {});".format(self.experiment_name, epoch, global_step)
            logger.info(message)
            checkpoint_name = "{}_epoch{}_step{}_lr{:e}_average_loss{:.5f}.pth".format(
                self.experiment_name, epoch, global_step, current_lr, loss_meter.avg)
            checkpoint_path = os.path.join(self.args.trainer.save_dir, checkpoint_name)
            tokenizer_path = os.path.join(self.args.trainer.save_dir, "tokenizer")
            os.makedirs(tokenizer_path, exist_ok=True)
            self.save_model(checkpoint_path, epoch=epoch, global_step=global_step, loss=loss_meter.val)
            self.processor.tokenizer.save_pretrained(tokenizer_path)
        return global_eval_step
