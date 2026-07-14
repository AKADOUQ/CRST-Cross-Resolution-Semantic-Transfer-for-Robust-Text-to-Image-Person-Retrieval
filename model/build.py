from model import objectives
from .clip_model import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights
from .crst_modules import ResolutionConditionedReasoner, TextGuidedFeatureRefiner
import torch
import torch.nn as nn
from collections import OrderedDict


class CRST(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(
            args.pretrain_choice, args.img_size, args.stride_size
        )
        self.embed_dim = base_cfg['embed_dim']
        self.logit_scale = torch.ones([]) * (1 / args.temperature)

        self.use_rcr = ('mlm' in self.current_task) and (not getattr(args, 'no_rcr', False))
        self.use_tgr = not (getattr(args, 'no_tgr', False) or getattr(args, 'no_refiner', False))
        self.use_cr_rda = not (getattr(args, 'no_cr_rda', False) or getattr(args, 'no_distill', False))
        self.use_feat_consistency = not (getattr(args, 'no_feat', False) or getattr(args, 'no_distill', False))

        self.rcr = ResolutionConditionedReasoner(self.embed_dim) if self.use_rcr else None
        self.tgr = TextGuidedFeatureRefiner(
            self.embed_dim,
            num_heads=getattr(args, 'tgr_heads', 8),
            dropout=getattr(args, 'tgr_dropout', 0.0),
        ) if self.use_tgr else None

        if 'id' in args.loss_names:
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.02)
            nn.init.constant_(self.classifier.bias.data, val=0.0)

        if 'mlm' in args.loss_names:
            self.cross_attn = nn.MultiheadAttention(
                self.embed_dim, self.embed_dim // 64, batch_first=True
            )
            self.cross_modal_transformer = Transformer(
                width=self.embed_dim,
                layers=args.cmt_depth,
                heads=self.embed_dim // 64,
            )
            scale = self.cross_modal_transformer.width ** -0.5
            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)

            proj_std = scale * ((2 * self.cross_modal_transformer.layers) ** -0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width) ** -0.5
            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

            self.mlm_head = nn.Sequential(OrderedDict([
                ('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                ('gelu', QuickGELU()),
                ('ln', LayerNorm(self.embed_dim)),
                ('fc', nn.Linear(self.embed_dim, args.vocab_size)),
            ]))
            nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')

    def _use_res_embedding(self):
        return not getattr(self.args, 'no_res_embed', False)

    def _hr_res_label(self, images):
        return torch.zeros(images.size(0), dtype=torch.long, device=images.device)

    def _pool_text_eot(self, text_tokens, text_ids):
        eot_indices = text_ids.argmax(dim=-1)
        return text_tokens[torch.arange(text_tokens.shape[0], device=text_tokens.device), eot_indices].float()

    def cross_former(self, q, k, v):
        x = self.cross_attn(
            self.ln_pre_t(q),
            self.ln_pre_i(k),
            self.ln_pre_i(v),
            need_weights=False,
        )[0]
        x = x.permute(1, 0, 2)
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_post(x)
        return x

    def encode_image(self, image, res_label=None):
        if not self._use_res_embedding():
            res_label = None
        x = self.base_model.encode_image(image, res_label)
        return x[:, 0, :].float()

    def encode_text_tokens(self, text):
        return self.base_model.encode_text(text)

    def encode_text(self, text):
        text_tokens = self.encode_text_tokens(text)
        return self._pool_text_eot(text_tokens, text)

    def _text_padding_mask(self, caption_ids):
        # CLIP padding id is 0. Keep at least SOT/EOT unmasked.
        return caption_ids.eq(0)

    def forward(self, batch):
        ret = dict()

        images_hr = batch['images']
        caption_ids = batch['caption_ids']
        pids = batch['pids']
        images_lr = batch.get('images_lr', None)
        res_label = batch.get('res_label', None)

        logit_scale = self.logit_scale
        ret.update({'temperature': 1 / logit_scale})

        if self.training and images_lr is not None:
            if res_label is None:
                res_label = self._hr_res_label(images_lr)
            res_label = res_label.to(device=images_lr.device, dtype=torch.long).clamp(0, 3)

            text_seq = self.encode_text_tokens(caption_ids)
            text_feats = self._pool_text_eot(text_seq, caption_ids)
            text_key_padding_mask = self._text_padding_mask(caption_ids)

            hr_res_label = self._hr_res_label(images_hr) if self._use_res_embedding() else None
            lr_res_label = res_label if self._use_res_embedding() else None

            image_seq_hr = self.base_model.encode_image(images_hr, hr_res_label)
            i_feats_hr = image_seq_hr[:, 0, :].float()

            image_seq_lr = self.base_model.encode_image(images_lr, lr_res_label)
            i_feats_lr = image_seq_lr[:, 0, :].float()

            if self.tgr is not None:
                i_feats_lr_refined = self.tgr(i_feats_lr, text_seq, text_key_padding_mask)
            else:
                i_feats_lr_refined = i_feats_lr

            paired_loss_weight = getattr(self.args, 'paired_loss_weight', 1.0)

            if 'id' in self.current_task:
                image_logits_hr = self.classifier(i_feats_hr.half()).float()
                text_logits = self.classifier(text_feats.half()).float()
                id_loss_hr = objectives.compute_id(image_logits_hr, text_logits, pids)

                image_logits_lr = self.classifier(i_feats_lr_refined.half()).float()
                id_loss_lr = objectives.compute_id(image_logits_lr, text_logits, pids)
                id_loss = id_loss_hr + paired_loss_weight * id_loss_lr
                ret.update({'id_loss': id_loss * self.args.id_loss_weight})

                image_pred = torch.argmax(image_logits_hr, dim=1)
                text_pred = torch.argmax(text_logits, dim=1)
                ret.update({'img_acc': (image_pred == pids).float().mean()})
                ret.update({'txt_acc': (text_pred == pids).float().mean()})

            if 'sdm' in self.current_task:
                sdm_loss_hr = objectives.compute_sdm(i_feats_hr, text_feats, pids, logit_scale)
                sdm_loss_lr = objectives.compute_sdm(i_feats_lr_refined, text_feats, pids, logit_scale)
                ret.update({'sdm_loss': sdm_loss_hr + paired_loss_weight * sdm_loss_lr})

            if 'mlm' in self.current_task:
                mlm_ids = batch['mlm_ids']
                mlm_feats = self.base_model.encode_text(mlm_ids)

                if self.rcr is not None:
                    gated_image_seq, rho = self.rcr(image_seq_lr, res_label)
                    x = self.cross_former(mlm_feats, gated_image_seq, gated_image_seq)
                    loss_name = 'rcr_loss'
                    ret.update({'rcr_gate_mean': rho.detach().mean()})
                else:
                    # Standard masked grounding branch for ablation.
                    x = self.cross_former(mlm_feats, image_seq_hr, image_seq_hr)
                    loss_name = 'mlm_loss'

                scores = self.mlm_head(x).float().reshape(-1, self.args.vocab_size)
                mlm_labels = batch['mlm_labels'].reshape(-1)
                ret.update({loss_name: objectives.compute_mlm(scores, mlm_labels) * self.args.mlm_loss_weight})

                pred = scores.max(1)[1]
                mlm_label_idx = torch.nonzero(mlm_labels, as_tuple=False).view(-1)
                if mlm_label_idx.numel() > 0:
                    acc = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()
                    ret.update({'mlm_acc': acc})

            if self.use_feat_consistency and self.tgr is not None:
                feat_loss = objectives.compute_feature_consistency(i_feats_lr_refined, i_feats_hr.detach())
                ret.update({'feat_loss': feat_loss * getattr(self.args, 'feat_loss_weight', 0.1)})

            if self.use_cr_rda:
                cr_rda_loss = objectives.compute_cr_rda(i_feats_hr.detach(), i_feats_lr_refined, text_feats, logit_scale)
                ret.update({'cr_rda_loss': cr_rda_loss * getattr(self.args, 'cr_rda_loss_weight', 0.1)})

        else:
            if res_label is None:
                res_label = self._hr_res_label(images_hr)
            res_label = res_label.to(device=images_hr.device, dtype=torch.long).clamp(0, 3)
            res_label = res_label if self._use_res_embedding() else None

            text_feats = self.encode_text(caption_ids)
            image_seq = self.base_model.encode_image(images_hr, res_label)
            i_feats = image_seq[:, 0, :].float()

            if 'itc' in self.current_task:
                ret.update({'itc_loss': objectives.compute_itc(i_feats, text_feats, logit_scale)})
            if 'sdm' in self.current_task:
                ret.update({'sdm_loss': objectives.compute_sdm(i_feats, text_feats, pids, logit_scale)})
            if 'id' in self.current_task:
                image_logits = self.classifier(i_feats.half()).float()
                text_logits = self.classifier(text_feats.half()).float()
                ret.update({'id_loss': objectives.compute_id(image_logits, text_logits, pids) * self.args.id_loss_weight})
                image_pred = torch.argmax(image_logits, dim=1)
                text_pred = torch.argmax(text_logits, dim=1)
                ret.update({'img_acc': (image_pred == pids).float().mean()})
                ret.update({'txt_acc': (text_pred == pids).float().mean()})

        return ret


def build_model(args, num_classes=11003):
    model = CRST(args, num_classes)
    convert_weights(model)
    return model