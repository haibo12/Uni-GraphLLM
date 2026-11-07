import copy
import gc
import json
import os
import time
from datetime import timedelta

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import InitProcessGroupKwargs
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Batch
from tqdm import tqdm
from transformers import LlamaTokenizer, AutoTokenizer, AutoConfig

from config import parse_args, apply_backbone_preset
from utils import (
    TrainerBase,
    eval_binary_auc_from_logits_and_labels,
    eval_classification,
    output_decode,
    print_trainable_params,
    seed_everything,
)
from utils.get_dataloader import get_dataloader

# Disable Weights & Biases logging by default so no account/offline-run dirs are
# needed. Override at runtime with `export WANDB_MODE=online` to re-enable.
os.environ.setdefault("WANDB_MODE", "disabled")

# torch>=2.6 flipped torch.load's default to weights_only=True, which rejects the
# pickled PyG Data objects this project loads (its own trusted local .pt files).
# Restore the pre-2.6 default globally so every torch.load in the data pipeline
# (get_dataloader, gft_data_utils, checkpoints, ...) works unchanged.
_orig_torch_load = torch.load
def _torch_load_weights_only_false(*a, **kw):
    kw.setdefault("weights_only", False)
    return _orig_torch_load(*a, **kw)
torch.load = _torch_load_weights_only_false


def group_batch_by_domain(instances):
    domain_to_indices = {}
    for idx, ins in enumerate(instances):
        domain = ins.dataname
        domain_to_indices.setdefault(domain, []).append(idx)
    return domain_to_indices


def grad_norm_on_params(loss, params, retain_graph=False):
    grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=True,
    )
    grad_norm_sq = None
    for grad in grads:
        if grad is None:
            continue
        cur = (grad.detach().float() ** 2).sum()
        grad_norm_sq = cur if grad_norm_sq is None else grad_norm_sq + cur
    if grad_norm_sq is None:
        return loss.detach().new_tensor(0.0)
    return torch.sqrt(grad_norm_sq + 1e-12)

def main(args, SEED):
    run_time = time.strftime("%Y%m%d%H%M", time.localtime())
    wandb_name = f"{args.prefix}_EXP{SEED}_{run_time}"
    if args.inference:
        group = f"{args.dataset}_{args.test_dataset}"
    else:
        group = f"{args.dataset}"
    accelerator.init_trackers(project_name=f"{args.project}",
                              init_kwargs={"wandb":
                                               {"tags": [args.dataset, args.backbone],
                                                "group": group,
                                                "name": wandb_name,
                                                "config": args}
                                           },
                              )

    seed_everything(seed=SEED)
    accelerator.print(args)

    with accelerator.main_process_first():

        # Backbone-aware tokenizer. Llama (vicuna) keeps the original slow
        # LlamaTokenizer + unk_token padding; Qwen3 / Qwen3.5 use AutoTokenizer.
        _model_type = (getattr(AutoConfig.from_pretrained(args.backbone), 'model_type', 'llama') or 'llama').lower()
        accelerator.print(f'[preset] {_model_type}: {apply_backbone_preset(args, _model_type)}')
        if _model_type == 'llama':
            tokenizer = LlamaTokenizer.from_pretrained(args.backbone)
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.backbone)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        special={'additional_special_tokens': ['<Node {}>'.format(i) for i in range(1, 20)]}   # Add a new special token as place holder
        tokenizer.add_special_tokens(special)
        # Record where the <Node i> placeholder ids start so downstream code can
        # detect graph-token positions independent of the backbone vocab size
        # (Llama=32000, Qwen>=150k). These tokens are always appended last, so
        # they occupy the highest, contiguous id range.
        _node_ids = tokenizer.convert_tokens_to_ids(['<Node {}>'.format(i) for i in range(1, 20)])
        tokenizer.node_token_start = min(_node_ids)
        # For Qwen3/3.5 build prompts with the model's *native* chat template and
        # enable_thinking=False (splices a closed empty <think></think> block so
        # the model answers directly). Llama/vicuna keeps its own conversation
        # template. Requires the tokenizer to actually ship a chat_template.
        tokenizer.use_chat_template = (
            _model_type != 'llama' and getattr(tokenizer, 'chat_template', None) is not None
        )

    accelerator.wait_for_everyone()

    cur_device = torch.cuda.current_device()
    trainer = TrainerBase(args, cur_device)

    # accelerator.print('Building DataLoader')
    # if not args.inference:
    #     train_loader = trainer.create_dataloader(tokenizer)
    # else:
    #     test_loader = trainer.create_dataloader(tokenizer)
    #     train_loader = test_loader
    args.gnn_input = 768
    args.edge_dim = None
    
    accelerator.print('Building Model')
    first_model, model = trainer.create_model()
    torch.set_default_dtype(torch.float32)

    accelerator.print('Building Optimizer')
    trainable_params, all_param = print_trainable_params(first_model, model)
    accelerator.print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")

    os.makedirs('./saved_model/first_model', exist_ok=True)
    if not args.freeze_llama:
        os.makedirs('./saved_model/model', exist_ok=True)

    first_model_path = './saved_model/first_model/{}' + str(args.num_token) + '_fm_{}_epoch{}_{}.pth'
    model_path = './saved_model/model/{}' + str(args.num_token) + '_m_{}_epoch{}_{}.pth'


    if not args.inference:
        model.config.use_cache = False
        train_loader = get_dataloader(args.dataset, args.batch_size, tokenizer, 'train', args=args)
        optimizer, warmup_scheduler = trainer.create_optimizer_and_scheduler(first_model, model, train_loader)

        domain_names = [name.strip() for name in args.dataset.split('+') if name.strip()]
        domain_difficulty_ema = {name: 0.0 for name in domain_names}
        domain_difficulty_sum = {name: 0.0 for name in domain_names}
        domain_difficulty_count = {name: 0 for name in domain_names}
        ema_momentum = args.curriculum_ema_momentum
        reweight_temperature = args.curriculum_reweight_temperature

        first_model, model, train_loader, optimizer, warmup_scheduler = accelerator.prepare(first_model, model, train_loader, optimizer, warmup_scheduler)

        projector_params = [
            p for n, p in first_model.named_parameters()
            if p.requires_grad and 'graph_projector' in n
        ]
        if len(projector_params) == 0:
            projector_params = [p for p in first_model.parameters() if p.requires_grad]

        accelerator.print('Training')
        num_training_steps = args.epoch * len(train_loader)
        curriculum_mean_warmup_ratio = min(max(args.curriculum_mean_warmup_ratio, 0.0), 1.0)
        curriculum_mean_warmup_steps = int(num_training_steps * curriculum_mean_warmup_ratio)
        progress_bar = tqdm(range(num_training_steps))
        global_step = 0

        # ---- profiling: alignment-training wall-clock & peak GPU memory ----
        # Timer starts here, i.e. AFTER dataset build / optimizer / prepare, so
        # the initial data load is not counted. Per-epoch weight-save time is
        # measured and subtracted below so it is excluded from training time.
        accelerator.wait_for_everyone()
        torch.cuda.synchronize(cur_device)
        torch.cuda.reset_peak_memory_stats(cur_device)
        train_start_time = time.time()
        save_time_total = 0.0
        cumulative_train_time = 0.0
        epoch_time_records = []

        for epoch in range(args.epoch):

            model.train()
            first_model.train()
            epoch_loss, accum_loss = 0., 0.
            last_monitor = None
            epoch_start_time = time.time()

            for step, raw_batch in enumerate(train_loader):
                with accelerator.accumulate(model):
                    optimizer.zero_grad()
                    batch = [copy.deepcopy(b) for b in raw_batch]
                    
                    input_ids = [b.input_ids.squeeze(0) for b in batch]
                    attention_mask = [b.attention_mask.squeeze(0) for b in batch]
                    target_ids = [b.target_ids.squeeze(0) for b in batch]
                    is_node = [b.is_node.squeeze(0) for b in batch]

                    for b in batch:
                        del b.input_ids, b.attention_mask, b.target_ids, b.is_node

                    if isinstance(batch, list):
                        batch = Batch.from_data_list(batch)
 
                    batch.input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
                    batch.attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
                    batch.target_ids = pad_sequence(target_ids, batch_first=True, padding_value=-100)
                    batch.is_node = pad_sequence(is_node, batch_first=True, padding_value=0)
                    batch.to(device = batch.x.device)

                    domain_to_indices = group_batch_by_domain(raw_batch)
                    active_domains = list(domain_to_indices.keys())

                    embeds = first_model(batch)
                    # --- smoke diagnostic: catch degenerate LLM inputs before the
                    #     forward turns a bad tensor into an opaque MKL SIGFPE ---
                    if (embeds.shape[0] == 0 or embeds.shape[1] == 0
                            or not torch.isfinite(embeds).all()):
                        accelerator.print(
                            f"[DIAG] step={step} bad embeds: shape={tuple(embeds.shape)} "
                            f"nan={torch.isnan(embeds).any().item()} inf={torch.isinf(embeds).any().item()} "
                            f"seqlens={batch.attention_mask.sum(1).tolist()} "
                            f"is_node_per_sample={batch.is_node.sum(1).tolist()} "
                            f"num_nodes={batch.x.shape[0]} num_graphs={int(batch.batch.max())+1 if batch.batch.numel() else 0} "
                            f"datanames={[b.dataname for b in raw_batch]}"
                        )
                        raise RuntimeError("Degenerate embeds detected (see [DIAG] above); aborting before SIGFPE.")
                    output=model(inputs_embeds=embeds, attention_mask=batch.attention_mask, labels=batch.target_ids)
                    base_loss = output['loss']

                    # Build per-sample mean token loss for domain-aware reweighting.
                    logits = output['logits']
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = batch.target_ids[..., 1:].contiguous()
                    per_token_loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100,
                        reduction='none',
                    ).view(shift_labels.size(0), shift_labels.size(1))
                    valid_token_mask = (shift_labels != -100).float()
                    valid_token_count = valid_token_mask.sum(dim=1).clamp(min=1.0)
                    per_sample_loss = (per_token_loss * valid_token_mask).sum(dim=1) / valid_token_count

                    domain_mean_loss = {}
                    for d in active_domains:
                        idx_tensor = torch.tensor(domain_to_indices[d], device=per_sample_loss.device, dtype=torch.long)
                        domain_mean_loss[d] = per_sample_loss.index_select(0, idx_tensor).mean()

                    # Difficulty estimation with grad norm on shared projector params.
                    domain_difficulty_now = {}
                    for d in active_domains:
                        # Keep graph for the later accelerator.backward(loss) pass.
                        g_d = grad_norm_on_params(domain_mean_loss[d], projector_params, retain_graph=True)
                        g_item = g_d.item()
                        domain_difficulty_sum[d] = domain_difficulty_sum.get(d, 0.0) + g_item
                        domain_difficulty_count[d] = domain_difficulty_count.get(d, 0) + 1
                        mean_now = domain_difficulty_sum[d] / max(domain_difficulty_count[d], 1)

                        if global_step < curriculum_mean_warmup_steps:
                            difficulty_value = mean_now
                            domain_difficulty_ema[d] = difficulty_value
                        else:
                            if domain_difficulty_count[d] <= 1 and domain_difficulty_ema.get(d, 0.0) == 0.0:
                                domain_difficulty_ema[d] = mean_now
                            ema_old = domain_difficulty_ema.get(d, mean_now)
                            difficulty_value = ema_momentum * ema_old + (1.0 - ema_momentum) * g_item
                            domain_difficulty_ema[d] = difficulty_value

                        domain_difficulty_now[d] = difficulty_value

                    difficulty_tensor = torch.tensor(
                        [domain_difficulty_now[d] for d in active_domains],
                        device=per_sample_loss.device,
                        dtype=per_sample_loss.dtype,
                    )
                    domain_weights = torch.softmax(difficulty_tensor / reweight_temperature, dim=0)

                    loss = per_sample_loss.new_tensor(0.0)
                    for i, d in enumerate(active_domains):
                        loss = loss + domain_weights[i] * domain_mean_loss[d]

                    # Keep lightweight monitor stats for periodic logging.
                    last_monitor = {
                        'base_loss': base_loss.detach().item(),
                        'weighted_loss': loss.detach().item(),
                        'active_domains': active_domains,
                        'domain_weights': {
                            d: domain_weights[i].detach().item()
                            for i, d in enumerate(active_domains)
                        },
                        'domain_difficulty': {
                            d: float(domain_difficulty_now[d])
                            for d in active_domains
                        },
                    }

                    accelerator.backward(loss)

                    accelerator.clip_grad_norm_(optimizer.param_groups[0]['params'], 0.1)
                    accelerator.clip_grad_norm_(optimizer.param_groups[1]['params'], 0.1)

                    optimizer.step()
                    warmup_scheduler.step()
                    epoch_loss, accum_loss = epoch_loss + loss.item(), accum_loss + loss.item()
                    global_step += 1


                if (step + 1) % args.grad_steps == 0:
                    graph_lr = optimizer.param_groups[0]["lr"]
                    lora_lr = optimizer.param_groups[1]["lr"]

                    accelerator.print({'Graph Lr': graph_lr, 'Lora Lr': lora_lr})
                    accelerator.print({'Accum Loss': accum_loss / args.grad_steps})
                    accelerator.log({'Graph Lr': graph_lr, 'Lora Lr': lora_lr})
                    accelerator.log({'Accum Loss': accum_loss / args.grad_steps})

                    if last_monitor is not None:
                        accelerator.print({
                            'Active Domains': last_monitor['active_domains'],
                            'Domain Weights': last_monitor['domain_weights'],
                            'Domain Difficulty EMA': last_monitor['domain_difficulty'],
                            'Base Loss': last_monitor['base_loss'],
                            'Weighted Loss': last_monitor['weighted_loss'],
                        })
                        monitor_log = {
                            'Base Loss': last_monitor['base_loss'],
                            'Weighted Loss': last_monitor['weighted_loss'],
                            'Active Domain Count': float(len(last_monitor['active_domains'])),
                        }
                        for d, w in last_monitor['domain_weights'].items():
                            monitor_log[f'Domain Weight/{d}'] = w
                        for d, diff in last_monitor['domain_difficulty'].items():
                            monitor_log[f'Domain Difficulty/{d}'] = diff
                        accelerator.log(monitor_log)
                    
                    accum_loss = 0.

                progress_bar.update(1)

            accelerator.print(f"Epoch: {epoch}|{args.epoch}: Train Loss (Epoch Mean): {epoch_loss / len(train_loader)}")
            accelerator.log({
                'Epoch': epoch,
                'Loss': epoch_loss / len(train_loader)
                })

            # train only one epoch and save model
            # Sync so the epoch's GPU work is complete before we timestamp, then
            # time the save separately so it is excluded from training time.
            accelerator.wait_for_everyone()
            torch.cuda.synchronize(cur_device)
            _save_t0 = time.time()
            if accelerator.is_main_process:
                accelerator.save(accelerator.unwrap_model(first_model).state_dict(), first_model_path.format(args.prefix, args.dataset, epoch, 'end'))
                if not args.freeze_llama:
                    accelerator.save(accelerator.unwrap_model(model).state_dict(), model_path.format(args.prefix, args.dataset, epoch, 'end'))
            accelerator.wait_for_everyone()
            epoch_save_time = time.time() - _save_t0
            save_time_total += epoch_save_time
            # best_epoch = args.best_epoch

            epoch_time = (time.time() - epoch_start_time) - epoch_save_time
            cumulative_train_time += epoch_time
            epoch_time_records.append({
                'epoch': epoch,
                'epoch_time_sec': epoch_time,
                'cumulative_train_time_sec': cumulative_train_time,
                'weight_save_time_sec': epoch_save_time,
            })
            accelerator.print(
                f"[Profiling][Epoch {epoch}] train_time={epoch_time:.2f}s "
                f"cumulative={cumulative_train_time:.2f}s save={epoch_save_time:.2f}s"
            )
     
        # ---- profiling: capture alignment-training cost before releasing memory ----
        accelerator.wait_for_everyone()
        torch.cuda.synchronize(cur_device)
        # Training time = wall-clock since timer start, minus all weight-save time
        # (initial data load already excluded: timer started after get_dataloader).
        train_elapsed = (time.time() - train_start_time) - save_time_total
        train_steps = max(int(global_step), 1)
        train_profile = {
            'phase': 'align_train',
            'backbone': args.backbone,
            'dataset': args.dataset,
            'prefix': args.prefix,
            'freeze_llama': bool(args.freeze_llama),
            'batch_size': args.batch_size,
            'grad_steps': args.grad_steps,
            'epochs': args.epoch,
            'num_processes': accelerator.num_processes,
            'total_train_time_sec': train_elapsed,
            'weight_save_time_sec': save_time_total,
            'per_epoch': epoch_time_records,
            'optimizer_steps': int(global_step),
            'avg_sec_per_step': train_elapsed / train_steps,
            'peak_gpu_mem_alloc_gb': torch.cuda.max_memory_allocated(cur_device) / (1024 ** 3),
            'peak_gpu_mem_reserved_gb': torch.cuda.max_memory_reserved(cur_device) / (1024 ** 3),
            'gpu_rank': accelerator.process_index,
            'gpu_name': torch.cuda.get_device_name(cur_device),
        }
        accelerator.print('[Profiling][Train]', train_profile)
        if accelerator.is_main_process:
            os.makedirs('./results/profiling', exist_ok=True)
            with open(f'./results/profiling/{args.prefix}{args.dataset}_train_profile.json', 'w') as f:
                json.dump(train_profile, f, ensure_ascii=False, indent=2, default=float)

        accelerator.wait_for_everyone()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        accelerator.wait_for_everyone()
    else:
        model.config.use_cache = True
        test_loader = get_dataloader(args.test_dataset, args.batch_size, tokenizer, 'test', args=args)
        first_model, model, test_loader = accelerator.prepare(first_model, model, test_loader)
        best_epoch = args.best_epoch

        # Step 5. Evaluating
        accelerator.print('Evaluating')
        with accelerator.main_process_first():
            first_model = accelerator.unwrap_model(first_model)
            # first_model.load_state_dict(torch.load(first_model_path.format(args.prefix, args.dataset, best_epoch, 'end')))
            first_model.load_state_dict(torch.load(first_model_path.format(args.prefix, args.dataset, best_epoch, 'end'), weights_only=False))
            model = model.cuda() # transformers bug
            model = accelerator.unwrap_model(model)
            if not args.freeze_llama:
                model.load_state_dict(torch.load(model_path.format(args.prefix, args.dataset, best_epoch, 'end'), weights_only=False))

        first_model.eval()
        model.eval()
        samples_seen = 0
        eval_output = []
        eval_label = []
        eval_binary_logits = []
        compute_binary_auc = args.test_dataset in ["chemhiv", "chempcba"]
        yes_token_id = None
        no_token_id = None
        if compute_binary_auc:
            yes_ids = tokenizer.encode(" Yes", add_special_tokens=False)
            no_ids = tokenizer.encode(" No", add_special_tokens=False)
            if len(yes_ids) != 1 or len(no_ids) != 1:
                yes_ids = tokenizer.encode("Yes", add_special_tokens=False)
                no_ids = tokenizer.encode("No", add_special_tokens=False)
            if len(yes_ids) != 1 or len(no_ids) != 1:
                raise ValueError("Tokenizer must map Yes/No to a single token for AUC.")
            yes_token_id, no_token_id = yes_ids[0], no_ids[0]

        progress_bar_test = tqdm(range(len(test_loader)))

        # ---- profiling: inference wall-clock, generation-only time & peak GPU memory ----
        accelerator.wait_for_everyone()
        torch.cuda.synchronize(cur_device)
        torch.cuda.reset_peak_memory_stats(cur_device)
        infer_start_time = time.time()
        gen_time_total = 0.0

        for step, raw_batch in enumerate(test_loader):
            batch = [copy.deepcopy(b) for b in raw_batch]
            input_ids = [b.input_ids.squeeze(0) for b in batch]
            attention_mask = [b.attention_mask.squeeze(0) for b in batch]
            target_ids = [b.target_ids.squeeze(0) for b in batch]
            is_node = [b.is_node.squeeze(0) for b in batch]

            for b in batch:
                del b.input_ids, b.attention_mask, b.target_ids, b.is_node
            if isinstance(batch, list):
                batch = Batch.from_data_list(batch)
            def left_pad(seqs, pad_val):
                return pad_sequence([s.flip(0) for s in seqs], batch_first=True, padding_value=pad_val).flip(1)
            
            batch.input_ids = left_pad(input_ids, tokenizer.pad_token_id)
            batch.attention_mask = left_pad(attention_mask, 0)
            batch.target_ids = left_pad(target_ids, -100)
            batch.is_node = left_pad(is_node, 0)
            batch.to(device = batch.x.device)
            with torch.no_grad():
            
                embeds = first_model(batch)

                torch.cuda.synchronize(cur_device)
                _gen_t0 = time.time()
                gen_opts = dict(greedy=args.greedy, top_p=args.top_p)
                if compute_binary_auc:
                    results, step_logits = model.g_step(
                        in_embeds=embeds,
                        attention_mask=batch.attention_mask,
                        return_step_logits=True,
                        yes_token_id=yes_token_id,
                        no_token_id=no_token_id,
                        auc_source=args.auc_source,
                        **gen_opts,
                    )
                else:
                    results = model.g_step(in_embeds=embeds, attention_mask=batch.attention_mask, **gen_opts)
                    step_logits = None
                torch.cuda.synchronize(cur_device)
                gen_time_total += time.time() - _gen_t0
                results = accelerator.pad_across_processes(results, dim=1, pad_index=tokenizer.pad_token_id)
                results_gathered = accelerator.gather(results).cpu().numpy()
                gathered_binary_logits = None
                if step_logits is not None:
                    gathered_binary_logits = accelerator.gather(step_logits).float().cpu()

                labels = accelerator.pad_across_processes(
                    batch.target_ids,
                    dim=1,
                    pad_index=tokenizer.pad_token_id)
                labels_gathered = accelerator.gather(labels).cpu().numpy()

                if accelerator.num_processes > 1:
                    if step == len(test_loader) - 1:
                        results_gathered = results_gathered[
                                                    : len(test_loader.dataset) - samples_seen]
                        labels_gathered = labels_gathered[
                                                    : len(test_loader.dataset) - samples_seen]
                        if gathered_binary_logits is not None:
                            gathered_binary_logits = gathered_binary_logits[
                                                    : len(test_loader.dataset) - samples_seen]
                    else:
                        samples_seen += len(results_gathered)
                labels_gathered = np.where(labels_gathered != -100, labels_gathered, tokenizer.pad_token_id)
                accelerator.print(tokenizer.batch_decode(results_gathered, skip_special_tokens=True))
                # accelerator.print(tokenizer.batch_decode(labels_gathered, skip_special_tokens=True))

                eval_output.append(results_gathered)
                eval_label.append(labels_gathered)
                if gathered_binary_logits is not None:
                    eval_binary_logits.append(gathered_binary_logits)
            progress_bar_test.update(1)

        # ---- profiling: capture inference cost ----
        accelerator.wait_for_everyone()
        torch.cuda.synchronize(cur_device)
        infer_elapsed = time.time() - infer_start_time
        total_samples = len(test_loader.dataset)
        infer_profile = {
            'phase': 'inference',
            'backbone': args.backbone,
            'test_dataset': args.test_dataset,
            'prefix': args.prefix,
            'batch_size': args.batch_size,
            'num_processes': accelerator.num_processes,
            'total_samples': int(total_samples),
            'total_inference_time_sec': infer_elapsed,
            'generation_only_time_sec': gen_time_total,
            'samples_per_sec': (total_samples / infer_elapsed) if infer_elapsed > 0 else None,
            'avg_ms_per_sample': (infer_elapsed / total_samples * 1000.0) if total_samples else None,
            'peak_gpu_mem_alloc_gb': torch.cuda.max_memory_allocated(cur_device) / (1024 ** 3),
            'peak_gpu_mem_reserved_gb': torch.cuda.max_memory_reserved(cur_device) / (1024 ** 3),
            'gpu_rank': accelerator.process_index,
            'gpu_name': torch.cuda.get_device_name(cur_device),
        }
        accelerator.print('[Profiling][Inference]', infer_profile)
        # Results are grouped under results/<model prefix>/<test_dataset>/ so each
        # model gets its own directory holding all per-dataset outputs.
        result_dir = f'./results/{args.prefix}{args.result_tag}/{args.test_dataset}'
        if accelerator.is_local_main_process:
            os.makedirs(result_dir, exist_ok=True)
            with open(f'{result_dir}/inference_profile.json', 'w') as f:
                json.dump(infer_profile, f, ensure_ascii=False, indent=2, default=float)

        # Step 6. Post-processing & Evaluating
        res_path = f'{result_dir}/model_results.txt'
        label_path = f'{result_dir}/model_labels.txt'
        metric_path = f'{result_dir}/metric.txt'

        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        if accelerator.is_local_main_process:
            eval_pred, eval_decode_label = output_decode(eval_output, eval_label, tokenizer)
            if compute_binary_auc:
                all_binary_logits = torch.cat(eval_binary_logits, dim=0) if len(eval_binary_logits) > 0 else None
                all_labels = [1 if x == "Yes" else 0 for x in eval_decode_label]
                all_labels = torch.from_numpy(np.array(all_labels)).long()
                metric = eval_binary_auc_from_logits_and_labels(
                    all_binary_logits[:, :2],
                    all_labels
                )
                metric['auc_source'] = args.auc_source
                if all_binary_logits.shape[1] == 4:
                    scores = all_binary_logits[:, :2]
                    metric['frac_one_of_yes_no_filtered'] = float((torch.isinf(scores).sum(1) == 1).float().mean())
                    metric['frac_both_filtered'] = float((torch.isinf(scores).sum(1) == 2).float().mean())
                    metric['auc_raw_logits'] = eval_binary_auc_from_logits_and_labels(
                        all_binary_logits[:, 2:], all_labels)['auc']
            else:
                metric = eval_classification(eval_pred, eval_decode_label, dataset_name=args.test_dataset)
            print(f"Evaluation results: {metric}")
            with open(res_path, 'w') as f:
                json.dump(eval_pred, f)
            with open(label_path, 'w') as f:
                json.dump(eval_decode_label, f)
            with open(metric_path, 'w') as f:
                json.dump(metric, f, ensure_ascii=False, indent=2, default=float)
if __name__ == "__main__":

    args = parse_args()
    for exp, SEED in enumerate(range(args.exp_num)):
        init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        transformers.logging.set_verbosity_error()
        accelerator = Accelerator(log_with="wandb", kwargs_handlers=[ddp_kwargs, init_kwargs],
                                  gradient_accumulation_steps=args.grad_steps)
        if args.seed != -1:
            SEED = args.seed
        main(args, SEED)
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        gc.collect()
