import json
import math
import os
import sys
import time

import numpy as np
import torch
import transformers
import wandb
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from loguru import logger
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, AutoTokenizer, AutoModel, AutoModelForCausalLM, get_linear_schedule_with_warmup

from dataset_process import PreprocessedDataset, PreprocessedRecDataCollator # Use preprocessed datasets
from evaluate_rec import RecEvaluator
from config import main_tokenizer_special_tokens, prompt_special_tokens_dict
from model_prompt import KGPrompt
from model_crs import PromptCRSModel
from args import parse_args

import warnings
warnings.filterwarnings("ignore", message=".*Profiler function.*")


if __name__ == '__main__':
    args = parse_args() 
    config = vars(args)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        device_placement=False,
        mixed_precision='bf16' if args.fp16 else 'no',
        kwargs_handlers=[ddp_kwargs]
    )
    device = accelerator.device

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    local_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    logger.remove()
    logger.add(sys.stderr, level='DEBUG' if accelerator.is_local_main_process else 'ERROR')
    log_file_path = os.path.join(args.output_dir, f"{local_time}.log")
    logger.add(log_file_path, level='DEBUG' if accelerator.is_local_main_process else 'ERROR')
    logger.info(config)
    logger.info(accelerator.state)
    # Log experiment description if provided via environment variable DES
    des = os.getenv('DES', '')
    if des:
        logger.info(f"DES: {des}")

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    run = None
    if args.use_wandb and accelerator.is_local_main_process:
        name = args.name if args.name else local_time
        group = args.name if args.name else 'DDP_' + local_time
        run = wandb.init(entity=args.entity, project=args.project, group=group, config=config, name=f"{name}_{accelerator.process_index}")

    if args.seed is not None:
        set_seed(args.seed)


    accelerator.print("Initializing Tokenizers and Sub-Models...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens(main_tokenizer_special_tokens)

    text_tokenizer = AutoTokenizer.from_pretrained(args.text_tokenizer)
    text_tokenizer.add_special_tokens(prompt_special_tokens_dict)

    language_model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True)
    language_model.resize_token_embeddings(len(tokenizer))
    language_model.config.pad_token_id = tokenizer.pad_token_id
    
    text_encoder = AutoModel.from_pretrained(args.text_encoder)
    text_encoder.resize_token_embeddings(len(text_tokenizer))


    entity2id_file = os.path.join('data', args.dataset,'entity2id.json')
    entity2id = json.load(open(entity2id_file, 'r', encoding='utf-8'))
    


    NUM_VIRTUAL_TOKENS = 20
    prompt_encoder = KGPrompt(
        model_hidden_size=language_model.config.hidden_size,
        token_hidden_size=text_encoder.config.hidden_size,
        n_entity=len(entity2id),
        num_virtual_tokens=NUM_VIRTUAL_TOKENS,
        args=args
    )
    
    crs_model = PromptCRSModel(language_model, prompt_encoder, text_encoder)

    trainable_params = filter(lambda p: p.requires_grad, crs_model.parameters())
    optimizer = AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)

    if args.prompt_encoder is not None:
        pretrained_prompt_path = os.path.join(args.prompt_encoder, 'prompt_encoder.pt')
        try:
            accelerator.print(f"Loading pre-trained prompt_encoder from {pretrained_prompt_path}")
            state_dict = torch.load(pretrained_prompt_path, map_location='cpu')
            crs_model.prompt_encoder.load_state_dict(state_dict)
            accelerator.print("Pre-trained prompt_encoder loaded successfully.")
        except FileNotFoundError:
            accelerator.print(f"Warning: No pre-trained prompt_encoder found at {pretrained_prompt_path}, starting from scratch.")
        except Exception as e:
            accelerator.print(f"An error occurred while loading the prompt_encoder: {e}")


    accelerator.print("Loading Datasets for Rec Task...")
    # data
    candidates_dataset = CandidatesDataset(dataset=args.dataset)
    train_dataset = PreprocessedDataset(args.dataset, 'train', 'rec', args.debug)
    valid_dataset = PreprocessedDataset(args.dataset, 'valid', 'rec', args.debug)
    test_dataset = PreprocessedDataset(args.dataset, 'test', 'rec', args.debug)
    
    data_collator = PreprocessedRecDataCollator(
        tokenizer=tokenizer,
        device=device, 
        pad_entity_id=len(entity2id),
        prompt_tokenizer=text_tokenizer,
        debug=args.debug,
        context_max_length=args.context_max_length,
        entity_max_length=args.entity_max_length,
        prompt_max_length=args.prompt_max_length
    )
    train_dataloader = DataLoader(train_dataset, batch_size=args.per_device_train_batch_size, collate_fn=data_collator, shuffle=True, num_workers=args.num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.per_device_eval_batch_size, collate_fn=data_collator, num_workers=args.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=args.per_device_eval_batch_size, collate_fn=data_collator, num_workers=args.num_workers)
    evaluator = RecEvaluator(device=accelerator.device)

    crs_model.to(device)

    accelerator.print("Preparing components with Accelerator for DDP...")
    crs_model, optimizer, train_dataloader, valid_dataloader, test_dataloader = accelerator.prepare(
        crs_model, optimizer, train_dataloader, valid_dataloader, test_dataloader
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    
    lr_scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
        power=1.0, 
        lr_end=args.learning_rate * 0.1 
    )

    lr_scheduler = accelerator.prepare(lr_scheduler)
    
    logger.info("***** Running training for Rec Task*****")
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    
    metric, mode = 'recall@5', 1
    best_metric = 0 if mode == 1 else float('inf')
    best_metric_dir = os.path.join(args.output_dir, 'best')
    os.makedirs(best_metric_dir, exist_ok=True)
    best_model_path = os.path.join(best_metric_dir, 'prompt_encoder.pt')


    for epoch in range(args.num_train_epochs):
        crs_model.train()
        train_loss = []
        train_evaluator = RecEvaluator(device=accelerator.device)
        logger.info(f"Epoch {epoch+1}/{args.num_train_epochs} - Training...")
        progress_bar.set_description(f"Epoch {epoch+1}")
        train_iter = tqdm(train_dataloader, disable=not accelerator.is_local_main_process, leave=False)
        for step, batch in enumerate(train_iter):
            outputs = crs_model(context=batch['context'], prompt=batch['prompt'], address=batch['address'], lonlat=batch['lonlat'], metadata=batch['metadata'],
                                cndidates_address=candidates_dataset()['address'], candidates_lonlat=candidates_dataset()['lonlat'], candidates_metadata=candidates_dataset()['metadata'])
            loss = outputs['loss']
            loss_for_log = loss.item()
            progress_bar.set_postfix({'lr': f"{optimizer.param_groups[0]['lr']:.2e}", 'loss': f"{loss_for_log:.4f}"})
            loss = loss / args.gradient_accumulation_steps
            
            accelerator.backward(loss)
            train_loss.append(loss_for_log)

            rec_logits = outputs['rec_logits']
            ranks = torch.topk(rec_logits, k=50, dim=-1).indices
            train_evaluator.evaluate(ranks, batch['context']['rec_labels'])

            if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1
            
            if completed_steps >= args.max_train_steps:
                break
        
        train_report = train_evaluator.report()
        gathered_train_report = {k: accelerator.gather(v).sum().item() for k, v in train_report.items()}
        train_metrics = {f'train/{k}': v / gathered_train_report['count'] for k, v in gathered_train_report.items() if k != 'count'}
        train_metrics['train/loss'] = np.mean(train_loss)
        logger.info(f"Epoch {epoch+1} Train Report: {train_metrics}")
        if run: run.log(train_metrics, step=epoch)

        crs_model.eval()
        valid_loss = []
        valid_evaluator = RecEvaluator(device=accelerator.device)
        logger.info(f"Epoch {epoch+1}/{args.num_train_epochs} - Validation...")
        for batch in tqdm(valid_dataloader, disable=not accelerator.is_local_main_process):
            with torch.no_grad():
                outputs = crs_model(context=batch['context'], prompt=batch['prompt'], address=batch['address'], lonlat=batch['lonlat'], metadata=batch['metadata'],
                                cndidates_address=candidates_dataset()['address'], candidates_lonlat=candidates_dataset()['lonlat'], candidates_metadata=candidates_dataset()['metadata'])
            valid_loss.append(accelerator.gather(outputs['loss']).mean().item())
            rec_logits = outputs['rec_logits']
            ranks = torch.topk(rec_logits, k=50, dim=-1).indices
            valid_evaluator.evaluate(ranks, batch['context']['rec_labels'])

        valid_report = valid_evaluator.report()
        gathered_valid_report = {k: accelerator.gather(v).sum().item() for k, v in valid_report.items()}
        valid_metrics = {f'valid/{k}': v / gathered_valid_report['count'] for k, v in gathered_valid_report.items() if k != 'count'}
        valid_metrics['valid/loss'] = np.mean(valid_loss)
        logger.info(f"Epoch {epoch+1} Validation Report: {valid_metrics}")
        if run: run.log(valid_metrics, step=epoch)

        test_loss = []
        test_evaluator = RecEvaluator(device=accelerator.device)
        logger.info(f"Epoch {epoch+1}/{args.num_train_epochs} - Testing...")
        for batch in tqdm(test_dataloader, disable=not accelerator.is_local_main_process):
            with torch.no_grad():
                outputs = crs_model(context=batch['context'], prompt=batch['prompt'], address=batch['address'], lonlat=batch['lonlat'], metadata=batch['metadata'],
                                cndidates_address=candidates_dataset()['address'], candidates_lonlat=candidates_dataset()['lonlat'], candidates_metadata=candidates_dataset()['metadata'])
            test_loss.append(accelerator.gather(outputs['loss']).mean().item())
            rec_logits = outputs['rec_logits']
            ranks = torch.topk(rec_logits, k=50, dim=-1).indices
            test_evaluator.evaluate(ranks, batch['context']['rec_labels'])

        test_report = test_evaluator.report()
        gathered_test_report = {k: accelerator.gather(v).sum().item() for k, v in test_report.items()}
        test_metrics = {f'test/{k}': v / gathered_test_report['count'] for k, v in gathered_test_report.items() if k != 'count'}
        test_metrics['test/loss'] = np.mean(test_loss)
        logger.info(f"Epoch {epoch+1} Test Report: {test_metrics}")
        if run: run.log(test_metrics, step=epoch)

        current_metric = valid_metrics[f'valid/{metric}']
        if (current_metric * mode) > (best_metric * mode):
            best_metric = current_metric
            logger.info(f'New best model found with {metric}: {best_metric:.4f}, saving state to {best_metric_dir}...')
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(crs_model)
                torch.save(unwrapped_model.prompt_encoder.state_dict(), best_model_path)
            accelerator.wait_for_everyone()
            logger.info('State saved.')

        if completed_steps >= args.max_train_steps:
            break

    logger.info("***** Training Finished *****")
    logger.info('Saving final model state...')
    final_dir = os.path.join(args.output_dir, 'final')
    os.makedirs(final_dir, exist_ok=True)
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(crs_model)
        torch.save(unwrapped_model.prompt_encoder.state_dict(), os.path.join(final_dir, 'prompt_encoder.pt'))
    logger.info(f'Final model state saved to {final_dir}')