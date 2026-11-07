import argparse
import pprint
import yaml


def parse_args(parse=True, **optional_kwargs):
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument("--dataset", type=str, default='Cora+Pubmed+wikics+arxiv+WN18RR+FB15K237+chempcba+chemhiv')
    parser.add_argument("--test_dataset", type=str, default='Cora')
    parser.add_argument("--project", type=str, default="project_GraphLLM")
    parser.add_argument("--exp_num", type=int, default=1)

    # Model Config
    parser.add_argument('--backbone', type=str, default='../llms/llms/lmsys/vicuna-7b-v1.5')
    parser.add_argument('--pretrain_gnn', type=str, default='')
    parser.add_argument('--prefix', type=str, default='trainable_llama_gnn')
    parser.add_argument('--gt_layers', type=int, default=2)
    parser.add_argument('--num_token', type=int, default=5)
    parser.add_argument('--att_d_model', type=int, default=2048)
    parser.add_argument('--gnn_output', type=int, default=384)
    parser.add_argument('--max_text_length', type=int, default=2215)

    # Training
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--freeze_llama', action='store_true')
    parser.add_argument('--train_gt', action='store_true',
                        help='whether to train GT in GraphEncoder (default: freeze GT)')
    parser.add_argument('--optim', default='adamw')
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--warmup_ratio', type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", type=str, default='cosine')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0,
                        help='currently unused: train_glm.py clips each param group to 0.1')
    parser.add_argument("--grad_steps", type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--adam_eps', type=float, default=1e-8)
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.999,
                        help='currently unused: the optimizer uses adam_beta1 for both betas')
    parser.add_argument('--epoch', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--best_epoch', type=int, default=0)
    parser.add_argument(
        '--curriculum_ema_momentum',
        type=float,
        default=0.9,
        help='EMA momentum for domain difficulty in curriculum alignment.',
    )
    parser.add_argument(
        '--curriculum_reweight_temperature',
        type=float,
        default=1.0,
        help='Softmax temperature for domain reweighting in curriculum alignment.',
    )
    parser.add_argument(
        '--curriculum_mean_warmup_ratio',
        type=float,
        default=0.3,
        help='Ratio of total training steps using running mean before switching to EMA.',
    )
    parser.add_argument(
        '--dataset_repeat',
        type=str,
        default='Cora:20,Pubmed:20,chemhiv:10,*:1',
        help='per-dataset repeat spec for train split, e.g. "Cora:5,Pubmed:4,*:1"',
    )

    parser.add_argument('--chem_feat_ids', type=str, default=None, choices=['processed', 'jh_summary'],
                        help="where chempcba/chemhiv atom/bond feature ids come from (default: backbone preset). "
                             "'processed': the GFT geometric_data_processed.pt that owns node_embs/edge_embs. "
                             "'jh_summary': the ids stored in summary/<name>.json; for chemhiv these follow a "
                             "different vocab order, only use it for checkpoints that were trained that way.")

    # Inference
    parser.add_argument('--gen_max_length', type=int, default=64,
                        help='currently unused: generation uses max_new_tokens=80')
    parser.add_argument('--greedy', action='store_true',
                        help='greedy decoding; default samples with temperature=0.7')
    parser.add_argument('--top_p', type=float, default=None,
                        help="sampling top_p (default: backbone preset; if the preset has none, the "
                             "backbone's generation_config is used)")
    parser.add_argument('--auc_source', type=str, default=None, choices=['logits', 'gen_scores'],
                        help="Yes/No score for chemhiv/chempcba AUC (default: backbone preset). "
                             "'logits': raw last-token logits. 'gen_scores': first-step generate() scores "
                             "after temperature/top-k/top-p processing (filtered tokens become -inf).")
    parser.add_argument('--result_tag', type=str, default='',
                        help='appended to the prefix for the results dir: results/<prefix><tag>/<test_dataset>')

    # Parse the arguments.
    if parse:
        args = parser.parse_args()
    # For interative engironmnet (ex. jupyter)
    else:
        args = parser.parse_known_args()[0]

    # Namespace => Dictionary
    kwargs = vars(args)
    kwargs.update(optional_kwargs)

    args = Config(**kwargs)

    return args


# Defaults for flags left unset, keyed by the backbone's HF model_type. Each preset
# reproduces how that backbone's released checkpoints were trained and evaluated:
# - llama (Vicuna): generation_config top_p=0.6, AUC from generate() scores,
#   chem ids consistent with the embedding tables (processed).
# - qwen3 / qwen3_5: backbone generation_config, AUC from raw logits,
#   chemhiv ids from summary/ (the vocab order those checkpoints were trained on;
#   --chem_feat_ids still uses the value name 'jh_summary').
BACKBONE_PRESETS = {
    'llama': dict(top_p=0.6, auc_source='gen_scores', chem_feat_ids='processed'),
    'qwen3': dict(top_p=None, auc_source='logits', chem_feat_ids='jh_summary'),
    'qwen3_5': dict(top_p=None, auc_source='logits', chem_feat_ids='jh_summary'),
    'qwen3_5_moe': dict(top_p=None, auc_source='logits', chem_feat_ids='jh_summary'),
}
FALLBACK_PRESET = dict(top_p=None, auc_source='logits', chem_feat_ids='processed')


def apply_backbone_preset(args, model_type):
    """Fill flags the user left unset (None) from the backbone preset; returns the resolved values."""
    preset = BACKBONE_PRESETS.get((model_type or '').lower(), FALLBACK_PRESET)
    for key, value in preset.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    return {key: getattr(args, key) for key in preset}


class Config(object):
    def __init__(self, **kwargs):
        """Configuration Class: set kwargs as class attributes with setattr"""
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def config_str(self):
        return pprint.pformat(self.__dict__)

    def __repr__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += self.config_str
        return config_str

    def save(self, path):
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            kwargs = yaml.safe_load(f)

        return Config(**kwargs)


if __name__ == '__main__':
    args = parse_args(True)
