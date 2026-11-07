import torch
from peft import LoraConfig, get_peft_model
from transformers import get_scheduler, AutoConfig

from model import GraphEncoder, InstructGLM, Qwen3GLM, Qwen3_5GLM
from .utils import get_optimizer


class TrainerBase(object):
    def __init__(self, args, device):
        self.args = args
        self.cur_device = device

    def create_config(self):
        # AutoConfig lets a single code path support Llama (vicuna), Qwen3 and
        # Qwen3.5 backbones. For a Llama backbone this returns a LlamaConfig,
        # identical to the previous behaviour.
        config = AutoConfig.from_pretrained(self.args.backbone)

        for attr in ('dropout_rate', 'dropout', 'attention_dropout', 'activation_dropout'):
            try:
                setattr(config, attr, self.args.dropout)
            except Exception:
                pass
        # Qwen3.5 nests the transformer hyper-params under ``text_config``.
        if hasattr(config, 'text_config') and config.text_config is not None:
            for attr in ('dropout', 'attention_dropout'):
                try:
                    setattr(config.text_config, attr, self.args.dropout)
                except Exception:
                    pass

        return config

    @staticmethod
    def _select_glm_class(model_type):
        mt = (model_type or 'llama').lower()
        if mt == 'llama':
            return InstructGLM
        if mt == 'qwen3':
            if Qwen3GLM is None:
                raise ImportError("Qwen3 backbone requested but transformers has no Qwen3ForCausalLM.")
            return Qwen3GLM
        if mt in ('qwen3_5', 'qwen3_5_moe'):
            if Qwen3_5GLM is None:
                raise ImportError("Qwen3.5 backbone requested but transformers has no Qwen3_5ForConditionalGeneration.")
            return Qwen3_5GLM
        raise ValueError(
            f"Unsupported backbone model_type '{model_type}'. Supported: llama, qwen3, qwen3_5."
        )

    def create_model(self):
        config = self.create_config()

        model_type = (getattr(config, 'model_type', 'llama') or 'llama').lower()
        model_cls = self._select_glm_class(model_type)

        load_kwargs = dict(
            config=config,
            torch_dtype=torch.bfloat16,
            device_map={"": self.cur_device},
        )
        # vicuna ships as pytorch .bin; preserve the original loading behaviour.
        # Qwen ships as safetensors where this kwarg is irrelevant.
        if model_type == 'llama':
            load_kwargs['weights_only'] = False

        model = model_cls.from_pretrained(
            self.args.backbone,
            **load_kwargs,
        )

        llama_embeds = model.get_input_embeddings().weight.data
        
        if self.args.freeze_llama:
            for n, p in model.named_parameters():
                p.requires_grad_(False)
        else:
            lora_r = 8
            lora_alpha = 16
            lora_target_modules=['q_proj','k_proj','v_proj', 'o_proj','lm_head']  # Select LoRA tuning modules.
            lora_dropout = 0.05
            LORA_config = LoraConfig(r=lora_r,lora_alpha=lora_alpha,target_modules=lora_target_modules,lora_dropout=lora_dropout,bias="none",task_type="CAUSAL_LM")
            model = get_peft_model(model, LORA_config)


        first_model = GraphEncoder(self.args, llama_embed=llama_embeds).to(self.cur_device, dtype=torch.bfloat16)
        if not self.args.inference:
            first_model.GT.load_state_dict(torch.load(self.args.pretrain_gnn, weights_only=False))
        for n, p in first_model.named_parameters():
            if n.split('.')[0] == 'GT':
                p.requires_grad_(self.args.train_gt)

        return first_model, model


    def create_optimizer_and_scheduler(self, first_model, model, train_loader):
        #no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in first_model.named_parameters() if p.requires_grad],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if p.requires_grad],
                "weight_decay": self.args.weight_decay,
            }
        ]

        optimizer_class = get_optimizer(self.args.optim)
        # beta2 is adam_beta1, not adam_beta2: all released checkpoints were trained this way.
        optimizer = optimizer_class(
            optimizer_grouped_parameters,
            lr=self.args.lr,
            eps=self.args.adam_eps,
            betas=(self.args.adam_beta1, self.args.adam_beta1))

        # 根据warmup配置，设置warmup
        total_steps = len(train_loader) * self.args.epoch
        num_warmup_steps = int(total_steps * self.args.warmup_ratio)
        assert num_warmup_steps <= total_steps, \
            'num_warmup_steps {} is too large, more than total_steps {}'.format(num_warmup_steps, total_steps)
        warmup_scheduler = get_scheduler(
            name=self.args.lr_scheduler_type,
            optimizer=optimizer, 
            num_warmup_steps=num_warmup_steps, 
            num_training_steps=total_steps
            )

        return optimizer, warmup_scheduler
