# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np
import torch
from maskrcnn_benchmark.modeling import registry
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter 

from maskrcnn_benchmark.layers import smooth_l1_loss, kl_div_loss, entropy_loss, Label_Smoothing_Regression
from maskrcnn_benchmark.modeling.utils import cat
from .model_msg_passing import IMPContext
from .model_vtranse import VTransEFeature
from .model_vctree import VCTreeLSTMContext
from .model_motifs import LSTMContext, FrequencyBias
from .model_motifs_with_attribute import AttributeLSTMContext
from .model_transformer import TransformerContext
from .utils_relation import layer_init, get_box_info, get_box_pair_info
from maskrcnn_benchmark.data import get_dataset_statistics
from .hierarchy_utils import (
    aggregate_fine_probs_to_coarse,
    build_expert_metadata,
    build_hierarchy_metadata,
    build_lite_expert_metadata,
    build_overlap_metadata,
)
from .utils_motifs import rel_vectors, obj_edge_vectors, to_onehot, nms_overlaps, encode_box_info 

from .utils_motifs import to_onehot, encode_box_info
from maskrcnn_benchmark.modeling.make_layers import make_fc




@registry.ROI_RELATION_PREDICTOR.register("PrototypeEmbeddingNetwork")
class PrototypeEmbeddingNetwork(nn.Module):
    def __init__(self, config, in_channels):
        super(PrototypeEmbeddingNetwork, self).__init__()

        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.cfg = config

        assert in_channels is not None
        self.in_channels = in_channels
        self.obj_dim = in_channels
        

        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        statistics = get_dataset_statistics(config)

        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics[
            'att_classes']
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_att_cls == len(att_classes)
        assert self.num_rel_cls == len(rel_classes)
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.num_obj_classes = len(obj_classes)
        
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM 
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM

        self.mlp_dim = 2048 # config.MODEL.ROI_RELATION_HEAD.PENET_MLP_DIM
        self.post_emb = nn.Linear(self.obj_dim, self.mlp_dim * 2)  

        self.embed_dim = 300 # config.MODEL.ROI_RELATION_HEAD.PENET_EMBED_DIM
        dropout_p = 0.2 # config.MODEL.ROI_RELATION_HEAD.PENET_DROPOUT
        
        
        obj_embed_vecs = obj_edge_vectors(obj_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim)  # load Glove for objects
        rel_embed_vecs = rel_vectors(rel_classes, wv_dir=config.GLOVE_DIR, wv_dim=self.embed_dim)   # load Glove for predicates
        self.obj_embed = nn.Embedding(self.num_obj_cls, self.embed_dim)
        self.rel_embed = nn.Embedding(self.num_rel_cls, self.embed_dim)
        with torch.no_grad():
            self.obj_embed.weight.copy_(obj_embed_vecs, non_blocking=True)
            self.rel_embed.weight.copy_(rel_embed_vecs, non_blocking=True)
       
        self.W_sub = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)
        self.W_obj = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)
        self.W_pred = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)

        self.gate_sub = nn.Linear(self.mlp_dim*2, self.mlp_dim)  
        self.gate_obj = nn.Linear(self.mlp_dim*2, self.mlp_dim)
        self.gate_pred = nn.Linear(self.mlp_dim*2, self.mlp_dim)

        self.vis2sem = nn.Sequential(*[
            nn.Linear(self.mlp_dim, self.mlp_dim*2), nn.ReLU(True),
            nn.Dropout(dropout_p), nn.Linear(self.mlp_dim*2, self.mlp_dim)
        ])

        self.project_head = MLP(self.mlp_dim, self.mlp_dim, self.mlp_dim*2, 2)

        self.linear_sub = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.linear_obj = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.linear_pred = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.linear_rel_rep = nn.Linear(self.mlp_dim, self.mlp_dim)
        
        self.norm_sub = nn.LayerNorm(self.mlp_dim)
        self.norm_obj = nn.LayerNorm(self.mlp_dim)
        self.norm_rel_rep = nn.LayerNorm(self.mlp_dim)

        self.dropout_sub = nn.Dropout(dropout_p)
        self.dropout_obj = nn.Dropout(dropout_p)
        self.dropout_rel_rep = nn.Dropout(dropout_p)
        
        self.dropout_rel = nn.Dropout(dropout_p)
        self.dropout_pred = nn.Dropout(dropout_p)
       
        self.down_samp = MLP(self.pooling_dim, self.mlp_dim, self.mlp_dim, 2) 

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.fine_margin = 1.0
        self.fine_topk_neg = 10
        self.prototype_margin = 7.0
        self.prototype_topk_neg = 1

        ##### refine object labels
        self.pos_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.BatchNorm1d(32, momentum= 0.001),
            nn.Linear(32, 128), nn.ReLU(inplace=True),
        ])

        self.obj_embed1 = nn.Embedding(self.num_obj_classes, self.embed_dim)
        with torch.no_grad():
            self.obj_embed1.weight.copy_(obj_embed_vecs, non_blocking=True)

        self.obj_dim = in_channels
        self.out_obj = make_fc(self.hidden_dim, self.num_obj_classes) 
        self.lin_obj_cyx = make_fc(self.obj_dim + self.embed_dim + 128, self.hidden_dim)

        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'
        
        self.nms_thresh = self.cfg.TEST.RELATION.LATER_NMS_PREDICTION_THRES

    def _compute_relation_representation(self, proposals, rel_pair_idxs, roi_features, union_features):
        entity_dists, entity_preds = self.refine_obj_labels(roi_features, proposals)

        entity_rep = self.post_emb(roi_features)
        entity_rep = entity_rep.view(entity_rep.size(0), 2, self.mlp_dim)

        sub_rep = entity_rep[:, 1].contiguous().view(-1, self.mlp_dim)
        obj_rep = entity_rep[:, 0].contiguous().view(-1, self.mlp_dim)
        entity_embeds = self.obj_embed(entity_preds)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        sub_reps = sub_rep.split(num_objs, dim=0)
        obj_reps = obj_rep.split(num_objs, dim=0)
        entity_preds = entity_preds.split(num_objs, dim=0)
        entity_embeds = entity_embeds.split(num_objs, dim=0)

        fusion_so = []
        pair_preds = []

        for pair_idx, sub_rep_i, obj_rep_i, entity_pred_i, entity_embed_i in zip(
            rel_pair_idxs, sub_reps, obj_reps, entity_preds, entity_embeds
        ):
            s_embed = self.W_sub(entity_embed_i[pair_idx[:, 0]])
            o_embed = self.W_obj(entity_embed_i[pair_idx[:, 1]])

            sem_sub = self.vis2sem(sub_rep_i[pair_idx[:, 0]])
            sem_obj = self.vis2sem(obj_rep_i[pair_idx[:, 1]])

            gate_sem_sub = torch.sigmoid(self.gate_sub(cat((s_embed, sem_sub), dim=-1)))
            gate_sem_obj = torch.sigmoid(self.gate_obj(cat((o_embed, sem_obj), dim=-1)))

            sub = s_embed + sem_sub * gate_sem_sub
            obj = o_embed + sem_obj * gate_sem_obj

            sub = self.norm_sub(self.dropout_sub(torch.relu(self.linear_sub(sub))) + sub)
            obj = self.norm_obj(self.dropout_obj(torch.relu(self.linear_obj(obj))) + obj)

            fusion_so.append(fusion_func(sub, obj))
            pair_preds.append(torch.stack((entity_pred_i[pair_idx[:, 0]], entity_pred_i[pair_idx[:, 1]]), dim=1))

        fusion_so = cat(fusion_so, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        if union_features is None:
            sem_pred = torch.zeros_like(fusion_so)
        else:
            sem_pred = self.vis2sem(self.down_samp(union_features))
        gate_sem_pred = torch.sigmoid(self.gate_pred(cat((fusion_so, sem_pred), dim=-1)))

        rel_rep = fusion_so - sem_pred * gate_sem_pred
        return entity_dists, rel_rep, pair_pred, num_objs, num_rels

    def _stabilize_relation_representation(self, rel_rep):
        return self.norm_rel_rep(self.dropout_rel_rep(torch.relu(self.linear_rel_rep(rel_rep))) + rel_rep)

    def _compute_proto_logits(self, rel_rep, predicate_proto, project_head, rel_dropout, pred_dropout, logit_scale):
        rel_rep = project_head(rel_dropout(torch.relu(rel_rep)))
        predicate_proto = project_head(pred_dropout(torch.relu(predicate_proto)))

        rel_rep_norm = F.normalize(rel_rep, dim=1)
        predicate_proto_norm = F.normalize(predicate_proto, dim=1)
        rel_dists = rel_rep_norm @ predicate_proto_norm.t() * logit_scale.exp()
        return rel_dists, rel_rep, predicate_proto, predicate_proto_norm

    def _prototype_similarity_regularization(self, predicate_proto_norm):
        simil_mat = predicate_proto_norm @ predicate_proto_norm.detach().t()
        denom = float(max(simil_mat.size(0) * simil_mat.size(1), 1))
        return torch.norm(torch.norm(simil_mat, p=2, dim=1), p=1) / denom

    def _prototype_distance_regularization(self, predicate_proto, margin, topk_neg):
        num_proto = predicate_proto.size(0)
        if num_proto <= 1:
            return predicate_proto.sum() * 0.0

        proto_dis_mat = torch.cdist(predicate_proto, predicate_proto.detach(), p=2) ** 2
        sorted_proto_dis_mat, _ = torch.sort(proto_dis_mat, dim=1)
        topk = min(max(int(topk_neg), 1), num_proto - 1)
        nearest_neg = sorted_proto_dis_mat[:, 1:topk + 1].mean(dim=1)
        return torch.clamp(margin - nearest_neg, min=0.0).mean()

    def _relation_to_proto_margin_loss(self, rel_rep, predicate_proto, rel_labels, margin, topk_neg):
        num_proto = predicate_proto.size(0)
        if rel_rep.size(0) == 0 or num_proto <= 1:
            return rel_rep.sum() * 0.0

        distance_set = torch.cdist(rel_rep, predicate_proto, p=2) ** 2
        batch_index = torch.arange(rel_labels.size(0), device=rel_labels.device)
        distance_set_pos = distance_set[batch_index, rel_labels]

        neg_mask = torch.ones_like(distance_set, dtype=torch.bool)
        neg_mask[batch_index, rel_labels] = False
        distance_set_neg = distance_set.masked_fill(~neg_mask, float("inf"))

        topk = min(max(int(topk_neg), 1), num_proto - 1)
        sorted_distance_set_neg, _ = torch.sort(distance_set_neg, dim=1)
        topk_sorted_distance_set_neg = sorted_distance_set_neg[:, :topk].mean(dim=1)
        return torch.clamp(distance_set_pos - topk_sorted_distance_set_neg + margin, min=0.0).mean()

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):

        add_losses = {}
        add_data = {}

        entity_dists, rel_rep, pair_pred, num_objs, num_rels = self._compute_relation_representation(
            proposals, rel_pair_idxs, roi_features, union_features
        )
        predicate_proto = self.W_pred(self.rel_embed.weight)
        rel_rep = self._stabilize_relation_representation(rel_rep)
        rel_dists, rel_rep, predicate_proto, predicate_proto_norm = self._compute_proto_logits(
            rel_rep, predicate_proto, self.project_head, self.dropout_rel, self.dropout_pred, self.logit_scale
        )

        entity_dists = entity_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        if self.training:
            rel_labels = cat(rel_labels, dim=0)
            add_losses.update({"l21_loss": self._prototype_similarity_regularization(predicate_proto_norm)})
            add_losses.update({
                "dist_loss2": self._prototype_distance_regularization(
                    predicate_proto, self.prototype_margin, self.prototype_topk_neg
                )
            })
            add_losses.update({
                "loss_dis": self._relation_to_proto_margin_loss(
                    rel_rep, predicate_proto, rel_labels, self.fine_margin, self.fine_topk_neg
                )
            })
 
        return entity_dists, rel_dists, add_losses, add_data


    def refine_obj_labels(self, roi_features, proposals):
        use_gt_label = self.training or self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL
        obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0) if use_gt_label else None
        pos_embed = self.pos_embed(encode_box_info(proposals))

        # label/logits embedding will be used as input
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            obj_labels = obj_labels.long()
            obj_embed = self.obj_embed1(obj_labels)
        else:
            obj_logits = cat([proposal.get_field("predict_logits") for proposal in proposals], dim=0).detach()
            obj_embed = F.softmax(obj_logits, dim=1) @ self.obj_embed1.weight

        assert proposals[0].mode == 'xyxy'

        pos_embed = self.pos_embed(encode_box_info(proposals))
        num_objs = [len(p) for p in proposals]
        obj_pre_rep_for_pred = self.lin_obj_cyx(cat([roi_features, obj_embed, pos_embed], -1))

        if self.mode == 'predcls':
            obj_labels = obj_labels.long()
            obj_preds = obj_labels
            obj_dists = to_onehot(obj_preds, self.num_obj_classes)
        else:
            obj_dists = self.out_obj(obj_pre_rep_for_pred)  # 512 -> 151
            use_decoder_nms = self.mode == 'sgdet' and not self.training
            if use_decoder_nms:
                boxes_per_cls = [proposal.get_field('boxes_per_cls') for proposal in proposals]
                obj_preds = self.nms_per_cls(obj_dists, boxes_per_cls, num_objs).long()
            else:
                obj_preds = (obj_dists[:, 1:].max(1)[1] + 1).long()
        
        return obj_dists, obj_preds

    def nms_per_cls(self, obj_dists, boxes_per_cls, num_objs):
        obj_dists = obj_dists.split(num_objs, dim=0)
        obj_preds = []
        for i in range(len(num_objs)):
            is_overlap = nms_overlaps(boxes_per_cls[i]).cpu().numpy() >= self.nms_thresh # (#box, #box, #class)

            out_dists_sampled = F.softmax(obj_dists[i], -1).cpu().numpy()
            out_dists_sampled[:, 0] = -1

            out_label = obj_dists[i].new(num_objs[i]).fill_(0)

            for i in range(num_objs[i]):
                box_ind, cls_ind = np.unravel_index(out_dists_sampled.argmax(), out_dists_sampled.shape)
                out_label[int(box_ind)] = int(cls_ind)
                out_dists_sampled[is_overlap[box_ind,:,cls_ind], cls_ind] = 0.0
                out_dists_sampled[box_ind] = -1.0 # This way we won't re-sample

            obj_preds.append(out_label.long())
        obj_preds = torch.cat(obj_preds, dim=0)
        return obj_preds



class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)  
        return x
    
    
def fusion_func(x, y):
    return F.relu(x + y) - (x - y) ** 2


class PrototypeResidualAdapter(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(PrototypeResidualAdapter, self).__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, prototype_bank):
        adapted = self.norm(prototype_bank)
        adapted = F.gelu(self.fc1(adapted))
        adapted = self.fc2(adapted)
        return adapted


class CoarseAdaptedFullLogitExpert(nn.Module):
    def __init__(self, proto_dim, hidden_dim, ownership_mask, inside_scale, outside_scale, temperature):
        super(CoarseAdaptedFullLogitExpert, self).__init__()
        self.adapter = PrototypeResidualAdapter(proto_dim, hidden_dim)
        self.inside_scale = float(inside_scale)
        self.outside_scale = float(outside_scale)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / max(float(temperature), 1e-6)))
        self.register_buffer("ownership_mask", ownership_mask.float().view(-1, 1))

    def forward(self, query_norm, base_proto):
        delta_proto = self.adapter(base_proto)
        proto_scale = self.outside_scale + (self.inside_scale - self.outside_scale) * self.ownership_mask
        adapted_proto = base_proto + delta_proto * proto_scale
        adapted_proto_norm = F.normalize(adapted_proto, dim=1)
        expert_logits = query_norm @ adapted_proto_norm.t() * self.logit_scale.exp()
        return expert_logits, adapted_proto, adapted_proto_norm


class OwnerOnlyPrototypeExpert(nn.Module):
    def __init__(self, proto_dim, hidden_dim, ownership_mask):
        super(OwnerOnlyPrototypeExpert, self).__init__()
        self.adapter = PrototypeResidualAdapter(proto_dim, hidden_dim)
        self.register_buffer("ownership_mask", ownership_mask.float().view(-1, 1))

    def forward(self, query_norm, base_proto, base_logit_scale):
        delta_proto = self.adapter(base_proto)
        adapted_proto = base_proto + delta_proto * self.ownership_mask
        adapted_proto_norm = F.normalize(adapted_proto, dim=1)
        expert_logits = query_norm @ adapted_proto_norm.t() * base_logit_scale.exp()
        return expert_logits, adapted_proto, adapted_proto_norm


class OverlapGate(nn.Module):
    def __init__(self, input_dim, hidden_dim, overlap_mask):
        super(OverlapGate, self).__init__()
        self.gate = MLP(input_dim, hidden_dim, overlap_mask.numel(), 2)
        self.register_buffer("overlap_mask", overlap_mask.float())

    def forward(self, rel_rep, coarse_probs):
        gate_logits = self.gate(cat((rel_rep, coarse_probs), dim=-1))
        overlap_gate = torch.sigmoid(gate_logits)
        return overlap_gate * self.overlap_mask.view(1, -1)


class SpatialOverlapAssistHead(nn.Module):
    def __init__(self, rel_input_dim, hidden_dim, num_rel_cls, overlap_mask):
        super(SpatialOverlapAssistHead, self).__init__()
        self.head = MLP(rel_input_dim, hidden_dim, num_rel_cls, 2)
        self.register_buffer("overlap_mask", overlap_mask.float())

    def forward(self, rel_rep):
        spatial_logits = self.head(rel_rep)
        return spatial_logits * self.overlap_mask.view(1, -1)



@registry.ROI_RELATION_PREDICTOR.register("PrototypeEmbeddingNetwork_Experts_lite")
class PrototypeEmbeddingNetwork_Experts_lite(PrototypeEmbeddingNetwork):
    def __init__(self, config, in_channels):
        super(PrototypeEmbeddingNetwork_Experts_lite, self).__init__(config, in_channels)

        hierarchy_cfg = config.MODEL.ROI_RELATION_HEAD.HIERARCHY
        self.hierarchy_enabled = hierarchy_cfg.ENABLED
        hierarchy_meta = build_hierarchy_metadata(self.rel_classes, hierarchy_cfg)

        self.coarse_rel_classes = hierarchy_meta["coarse_predicates"]
        self.num_coarse_rel_cls = len(self.coarse_rel_classes)

        self.register_buffer(
            "fine_to_coarse_idx",
            torch.tensor(hierarchy_meta["fine_to_coarse_id"], dtype=torch.long),
        )
        self.register_buffer("coarse_to_fine_mask", hierarchy_meta["coarse_to_fine_mask"])

        coarse_embed_weights = torch.zeros(self.num_coarse_rel_cls, self.embed_dim)
        for coarse_idx in range(self.num_coarse_rel_cls):
            member_idx = (self.fine_to_coarse_idx == coarse_idx).nonzero(as_tuple=False).view(-1)
            if member_idx.numel() > 0:
                coarse_embed_weights[coarse_idx] = self.rel_embed.weight.detach()[member_idx].mean(dim=0)

        self.coarse_rel_embed = nn.Embedding(self.num_coarse_rel_cls, self.embed_dim)
        with torch.no_grad():
            self.coarse_rel_embed.weight.copy_(coarse_embed_weights, non_blocking=True)

        self.W_pred_coarse = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)
        self.coarse_project_head = MLP(self.mlp_dim, self.mlp_dim, self.mlp_dim * 2, 2)
        self.dropout_rel_coarse = nn.Dropout(self.dropout_rel.p)
        self.dropout_pred_coarse = nn.Dropout(self.dropout_pred.p)
        self.coarse_logit_scale = nn.Parameter(
            torch.ones([]) * np.log(1 / hierarchy_cfg.COARSE_TEMPERATURE)
        )

        self.consistency_type = str(getattr(hierarchy_cfg, "CONSISTENCY_TYPE", "symmetric")).lower()
        self.mask_epsilon = hierarchy_cfg.MASK_EPSILON
        self.hierarchy_loss_weight = hierarchy_cfg.HIERARCHY_LOSS_WEIGHT
        self.coarse_loss_weight = hierarchy_cfg.COARSE_LOSS_WEIGHT
        self.fine_proto_weight = hierarchy_cfg.FINE_PROTO_WEIGHT
        self.coarse_proto_weight = hierarchy_cfg.COARSE_PROTO_WEIGHT
        self.proto_reg_weight = hierarchy_cfg.PROTO_REG_WEIGHT
        self.fine_margin = hierarchy_cfg.FINE_MARGIN
        self.fine_topk_neg = hierarchy_cfg.FINE_TOPK_NEG
        self.coarse_margin = hierarchy_cfg.COARSE_MARGIN
        self.coarse_topk_neg = hierarchy_cfg.COARSE_TOPK_NEG
        self.prototype_margin = hierarchy_cfg.PROTOTYPE_MARGIN
        self.prototype_topk_neg = hierarchy_cfg.PROTOTYPE_TOPK_NEG

        expert_cfg = config.MODEL.ROI_RELATION_HEAD.EXPERTS_LITE
        self.experts_lite_enabled = expert_cfg.ENABLED
        expert_meta = build_lite_expert_metadata(self.rel_classes, config.MODEL.ROI_RELATION_HEAD.HIERARCHY)

        self.expert_names = expert_meta["expert_names"]
        self.num_experts = len(self.expert_names)
        self.owner_loss_weight = float(expert_cfg.OWNER_LOSS_WEIGHT)
        self.owned_proto_weight = float(expert_cfg.OWNED_PROTO_WEIGHT)
        self.owner_logit_alpha = float(expert_cfg.OWNER_LOGIT_ALPHA)
        self.routing_topk = int(getattr(expert_cfg, "ROUTING_TOPK", 2))

        self.register_buffer("expert_owner_masks", expert_meta["expert_owner_masks"])
        self.register_buffer("coarse_to_expert_idx", expert_meta["coarse_to_expert_idx"])
        expert_coarse_indices = torch.full((self.num_experts,), -1, dtype=torch.long)
        valid_coarse_idx = (self.coarse_to_expert_idx >= 0).nonzero(as_tuple=False).view(-1)
        if valid_coarse_idx.numel() > 0:
            expert_indices = self.coarse_to_expert_idx[valid_coarse_idx]
            expert_coarse_indices[expert_indices] = valid_coarse_idx
        self.register_buffer("expert_coarse_indices", expert_coarse_indices)
        self.expert_class_indices = [
            torch.tensor(class_indices, dtype=torch.long)
            for class_indices in expert_meta["expert_class_indices"]
        ]

        proto_dim = self.mlp_dim * 2
        self.expert_bank = nn.ModuleList(
            [
                OwnerOnlyPrototypeExpert(
                    proto_dim,
                    int(expert_cfg.ADAPTER_HIDDEN_DIM),
                    self.expert_owner_masks[expert_idx],
                )
                for expert_idx in range(self.num_experts)
            ]
        )

    def _hierarchy_consistency_loss(self, coarse_probs, coarse_probs_from_fine):
        coarse_probs = coarse_probs.clamp(min=self.mask_epsilon)
        coarse_probs_from_fine = coarse_probs_from_fine.clamp(min=self.mask_epsilon)
        if self.consistency_type == "teacher":
            return F.kl_div(
                coarse_probs_from_fine.log(),
                coarse_probs.detach(),
                reduction="batchmean",
            )
        if self.consistency_type != "symmetric":
            raise ValueError("Unsupported hierarchy consistency type: {}".format(self.consistency_type))
        return 0.5 * (
            F.kl_div(coarse_probs.log(), coarse_probs_from_fine, reduction="batchmean") +
            F.kl_div(coarse_probs_from_fine.log(), coarse_probs, reduction="batchmean")
        )

    def _forward_global_fine_base(self, rel_rep):
        fine_predicate_proto = self.W_pred(self.rel_embed.weight)
        fine_rel_rep = self.project_head(self.dropout_rel(torch.relu(rel_rep)))
        fine_predicate_proto = self.project_head(self.dropout_pred(torch.relu(fine_predicate_proto)))
        fine_rel_rep_norm = F.normalize(fine_rel_rep, dim=1)
        fine_proto_norm = F.normalize(fine_predicate_proto, dim=1)
        fine_logits = fine_rel_rep_norm @ fine_proto_norm.t() * self.logit_scale.exp()
        return fine_logits, fine_rel_rep, fine_rel_rep_norm, fine_predicate_proto, fine_proto_norm

    def _select_owner_expert_idx(self, coarse_probs, rel_labels=None):
        if self.training and rel_labels is not None:
            coarse_labels = self.fine_to_coarse_idx[rel_labels]
        else:
            coarse_labels = coarse_probs.argmax(dim=1)

        owner_expert_idx = self.coarse_to_expert_idx[coarse_labels]
        return coarse_labels, owner_expert_idx

    def _build_topk_route_weights(self, coarse_probs):
        if self.num_experts == 0:
            return coarse_probs.new_zeros(coarse_probs.size(0), 0)

        expert_probs = coarse_probs.index_select(1, self.expert_coarse_indices)
        topk = min(max(self.routing_topk, 1), expert_probs.size(1))
        topk_vals, topk_idx = torch.topk(expert_probs, k=topk, dim=1)

        route_weights = torch.zeros_like(expert_probs)
        route_weights.scatter_(1, topk_idx, topk_vals)
        return route_weights

    def _apply_topk_expert_correction(self, base_logits, expert_logits_stack, route_weights):
        if self.num_experts == 0 or route_weights.numel() == 0:
            return base_logits

        expert_residual = expert_logits_stack - base_logits.unsqueeze(1)
        expert_residual = expert_residual * self.expert_owner_masks.unsqueeze(0)
        weighted_residual = (route_weights.unsqueeze(-1) * expert_residual).sum(dim=1)
        return base_logits + self.owner_logit_alpha * weighted_residual

    def _compute_owned_expert_proto_loss(self, expert_proto_list, expert_proto_norm_list, base_proto):
        owned_proto_losses = []

        for expert_idx, (expert_proto, expert_proto_norm) in enumerate(zip(expert_proto_list, expert_proto_norm_list)):
            owner_mask = self.expert_owner_masks[expert_idx].bool()
            if int(owner_mask.sum().item()) <= 1:
                continue

            owned_proto_losses.append(
                self._prototype_similarity_regularization(expert_proto_norm[owner_mask]) +
                self._prototype_distance_regularization(
                    expert_proto[owner_mask], self.prototype_margin, self.prototype_topk_neg
                )
            )

        zero_tensor = base_proto.sum() * 0.0
        return torch.stack(owned_proto_losses).mean() if owned_proto_losses else zero_tensor

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        if not self.experts_lite_enabled:
            return super(PrototypeEmbeddingNetwork_Experts_lite, self).forward(
                proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger
            )

        add_losses = {}
        add_data = {}

        entity_dists, rel_rep, pair_pred, num_objs, num_rels = self._compute_relation_representation(
            proposals, rel_pair_idxs, roi_features, union_features
        )
        rel_rep = self._stabilize_relation_representation(rel_rep)

        fine_logits, fine_rel_rep, fine_rel_rep_norm, fine_predicate_proto, fine_proto_norm = self._forward_global_fine_base(rel_rep)
        coarse_predicate_proto = self.W_pred_coarse(self.coarse_rel_embed.weight)
        coarse_logits, coarse_rel_rep, coarse_predicate_proto, coarse_proto_norm = self._compute_proto_logits(
            rel_rep,
            coarse_predicate_proto,
            self.coarse_project_head,
            self.dropout_rel_coarse,
            self.dropout_pred_coarse,
            self.coarse_logit_scale,
        )
        coarse_probs = F.softmax(coarse_logits, dim=1)
        route_weights = self._build_topk_route_weights(coarse_probs)

        flat_rel_labels = cat(rel_labels, dim=0) if self.training and rel_labels is not None else None
        coarse_labels, owner_expert_idx = self._select_owner_expert_idx(coarse_probs, flat_rel_labels)

        expert_logits = []
        expert_proto_list = []
        expert_proto_norm_list = []
        for expert in self.expert_bank:
            expert_logit, expert_proto, expert_proto_norm = expert(
                fine_rel_rep_norm, fine_predicate_proto, self.logit_scale
            )
            expert_logits.append(expert_logit)
            expert_proto_list.append(expert_proto)
            expert_proto_norm_list.append(expert_proto_norm)

        if expert_logits:
            expert_logits_stack = torch.stack(expert_logits, dim=1)
        else:
            expert_logits_stack = fine_logits.new_zeros(fine_logits.size(0), 0, fine_logits.size(1))

        final_logits = self._apply_topk_expert_correction(fine_logits, expert_logits_stack, route_weights)

        entity_dists = entity_dists.split(num_objs, dim=0)
        rel_dists = final_logits.split(num_rels, dim=0)
        add_data["coarse_rel_dists"] = coarse_logits.split(num_rels, dim=0)
        add_data["base_fine_rel_dists"] = fine_logits.split(num_rels, dim=0)
        add_data["final_rel_dists"] = rel_dists
        add_data["expert_route_weights"] = route_weights.split(num_rels, dim=0)

        if self.training:
            zero_tensor = fine_logits.sum() * 0.0

            add_losses["l21_loss"] = self.proto_reg_weight * self.fine_proto_weight * self._prototype_similarity_regularization(fine_proto_norm)
            add_losses["dist_loss2"] = self.proto_reg_weight * self.fine_proto_weight * self._prototype_distance_regularization(
                fine_predicate_proto, self.prototype_margin, self.prototype_topk_neg
            )
            add_losses["loss_dis"] = self.fine_proto_weight * self._relation_to_proto_margin_loss(
                fine_rel_rep, fine_predicate_proto, flat_rel_labels, self.fine_margin, self.fine_topk_neg
            )

            add_losses["loss_coarse_cls"] = self.coarse_loss_weight * F.cross_entropy(coarse_logits, coarse_labels.long())
            add_losses["loss_coarse_l21"] = self.proto_reg_weight * self.coarse_proto_weight * self._prototype_similarity_regularization(coarse_proto_norm)
            add_losses["loss_coarse_proto_dist"] = self.proto_reg_weight * self.coarse_proto_weight * self._prototype_distance_regularization(
                coarse_predicate_proto, self.prototype_margin, self.prototype_topk_neg
            )
            add_losses["loss_coarse_dis"] = self.coarse_proto_weight * self._relation_to_proto_margin_loss(
                coarse_rel_rep, coarse_predicate_proto, coarse_labels, self.coarse_margin, self.coarse_topk_neg
            )

            valid_owner_mask = owner_expert_idx >= 0
            if int(valid_owner_mask.sum().item()) > 0:
                owner_sample_idx = valid_owner_mask.nonzero(as_tuple=False).view(-1)
                owner_logits = expert_logits_stack[owner_sample_idx, owner_expert_idx[owner_sample_idx]]
                add_losses["loss_expert_owner"] = self.owner_loss_weight * F.cross_entropy(
                    owner_logits, flat_rel_labels[owner_sample_idx].long()
                )
            else:
                add_losses["loss_expert_owner"] = zero_tensor

            owned_proto_loss = self._compute_owned_expert_proto_loss(
                expert_proto_list, expert_proto_norm_list, fine_predicate_proto
            )
            add_losses["loss_expert_owned_proto"] = self.proto_reg_weight * self.owned_proto_weight * owned_proto_loss

            fine_probs = F.softmax(final_logits, dim=1)
            coarse_probs_from_fine = aggregate_fine_probs_to_coarse(fine_probs, self.coarse_to_fine_mask)
            add_losses["loss_hierarchy_consistency"] = self.hierarchy_loss_weight * self._hierarchy_consistency_loss(
                coarse_probs, coarse_probs_from_fine
            )

        return entity_dists, rel_dists, add_losses, add_data



@registry.ROI_RELATION_PREDICTOR.register("TransformerPredictor")
class TransformerPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(TransformerPredictor, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        # load parameters
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        
        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics['att_classes']
        assert self.num_obj_cls==len(obj_classes)
        assert self.num_att_cls==len(att_classes)
        assert self.num_rel_cls==len(rel_classes)
        # module construct
        self.context_layer = TransformerContext(config, obj_classes, rel_classes, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)
        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.ctx_compress = nn.Linear(self.hidden_dim * 2, self.num_rel_cls)

        # initialize layer parameters 
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.rel_compress, xavier=True)
        layer_init(self.ctx_compress, xavier=True)
        layer_init(self.post_cat, xavier=True)
        
        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(config, statistics)

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """
        if self.attribute_on:
            obj_dists, obj_preds, att_dists, edge_ctx = self.context_layer(roi_features, proposals, logger)
        else:
            obj_dists, obj_preds, edge_ctx = self.context_layer(roi_features, proposals, logger)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)
        
        # from object level feature to pairwise relation level feature
        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1))
            pair_preds.append(torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1))
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        ctx_gate = self.post_cat(prod_rep)

        # use union box and mask convolution
        if self.use_vision:
            if self.union_single_not_match:
                visual_rep = ctx_gate * self.up_dim(union_features)
            else:
                visual_rep = ctx_gate * union_features

        rel_dists = self.rel_compress(visual_rep) + self.ctx_compress(prod_rep)
                
        # use frequence bias
        if self.use_bias:
            rel_dists = rel_dists + self.freq_bias.index_with_labels(pair_pred)

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        add_losses = {}

        if self.attribute_on:
            att_dists = att_dists.split(num_objs, dim=0)
            return (obj_dists, att_dists), rel_dists, add_losses
        else:
            return obj_dists, rel_dists, add_losses


@registry.ROI_RELATION_PREDICTOR.register("IMPPredictor")
class IMPPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(IMPPredictor, self).__init__()
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.use_bias = False

        assert in_channels is not None

        self.context_layer = IMPContext(config, self.num_obj_cls, self.num_rel_cls, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        
        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        # freq 
        if self.use_bias:
            statistics = get_dataset_statistics(config)
            self.freq_bias = FrequencyBias(config, statistics)


    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        if self.union_single_not_match:
            union_features = self.up_dim(union_features)

        # encode context infomation
        obj_dists, rel_dists = self.context_layer(roi_features, proposals, union_features, rel_pair_idxs, logger)

        num_objs = [len(b) for b in proposals]
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        assert len(num_rels) == len(num_objs)

        if self.use_bias:
            obj_preds = obj_dists.max(-1)[1]
            obj_preds = obj_preds.split(num_objs, dim=0)

            pair_preds = []
            for pair_idx, obj_pred in zip(rel_pair_idxs, obj_preds):
                pair_preds.append( torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1) )
            pair_pred = cat(pair_preds, dim=0)

            rel_dists = rel_dists + self.freq_bias.index_with_labels(pair_pred.long())

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        # we use obj_preds instead of pred from obj_dists
        # because in decoder_rnn, preds has been through a nms stage
        add_losses = {}

        return obj_dists, rel_dists, add_losses



@registry.ROI_RELATION_PREDICTOR.register("MotifPredictor")
class MotifPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(MotifPredictor, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        
        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics['att_classes']
        assert self.num_obj_cls==len(obj_classes)
        assert self.num_att_cls==len(att_classes)
        assert self.num_rel_cls==len(rel_classes)
        # init contextual lstm encoding
        if self.attribute_on:
            self.context_layer = AttributeLSTMContext(config, obj_classes, att_classes, rel_classes, in_channels)
        else:
            self.context_layer = LSTMContext(config, obj_classes, rel_classes, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)
        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls, bias=True)

        # initialize layer parameters 
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)
        layer_init(self.rel_compress, xavier=True)
        
        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(config, statistics)

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        # encode context infomation
        if self.attribute_on:
            obj_dists, obj_preds, att_dists, edge_ctx = self.context_layer(roi_features, proposals, logger)
        else:
            obj_dists, obj_preds, edge_ctx, _ = self.context_layer(roi_features, proposals, logger)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)
        
        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append( torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1) )
            pair_preds.append( torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1) )
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        prod_rep = self.post_cat(prod_rep)

        if self.use_vision:
            if self.union_single_not_match:
                prod_rep = prod_rep * self.up_dim(union_features)
            else:
                prod_rep = prod_rep * union_features

        rel_dists = self.rel_compress(prod_rep)

        if self.use_bias:
            rel_dists = rel_dists + self.freq_bias.index_with_labels(pair_pred.long())

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        # we use obj_preds instead of pred from obj_dists
        # because in decoder_rnn, preds has been through a nms stage
        add_losses = {}

        if self.attribute_on:
            att_dists = att_dists.split(num_objs, dim=0)
            return (obj_dists, att_dists), rel_dists, add_losses
        else:
            return obj_dists, rel_dists, add_losses


@registry.ROI_RELATION_PREDICTOR.register("VCTreePredictor")
class VCTreePredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(VCTreePredictor, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        
        assert in_channels is not None
        num_inputs = in_channels

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics['att_classes']
        assert self.num_obj_cls==len(obj_classes)
        assert self.num_att_cls==len(att_classes)
        assert self.num_rel_cls==len(rel_classes)
        # init contextual lstm encoding
        self.context_layer = VCTreeLSTMContext(config, obj_classes, rel_classes, statistics, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)

        # learned-mixin
        #self.uni_gate = nn.Linear(self.pooling_dim, self.num_rel_cls)
        #self.frq_gate = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.ctx_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        #self.uni_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        #layer_init(self.uni_gate, xavier=True)
        #layer_init(self.frq_gate, xavier=True)
        layer_init(self.ctx_compress, xavier=True)
        #layer_init(self.uni_compress, xavier=True)

        # initialize layer parameters 
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)
        
        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        self.freq_bias = FrequencyBias(config, statistics)

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        # encode context infomation
        obj_dists, obj_preds, edge_ctx, binary_preds = self.context_layer(roi_features, proposals, rel_pair_idxs, logger)

        # post decode
        edge_rep = F.relu(self.post_emb(edge_ctx))
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)
        
        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append( torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1) )
            pair_preds.append( torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1) )
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        prod_rep = self.post_cat(prod_rep)

        # learned-mixin Gate
        #uni_gate = torch.tanh(self.uni_gate(self.drop(prod_rep)))
        #frq_gate = torch.tanh(self.frq_gate(self.drop(prod_rep)))

        if self.union_single_not_match:
            union_features = self.up_dim(union_features)

        ctx_dists = self.ctx_compress(prod_rep * union_features)
        #uni_dists = self.uni_compress(self.drop(union_features))
        frq_dists = self.freq_bias.index_with_labels(pair_pred.long())

        rel_dists = ctx_dists + frq_dists
        #rel_dists = ctx_dists + uni_gate * uni_dists + frq_gate * frq_dists

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        # we use obj_preds instead of pred from obj_dists
        # because in decoder_rnn, preds has been through a nms stage
        add_losses = {}

        if self.training:
            binary_loss = []
            for bi_gt, bi_pred in zip(rel_binarys, binary_preds):
                bi_gt = (bi_gt > 0).float()
                binary_loss.append(F.binary_cross_entropy_with_logits(bi_pred, bi_gt))
            add_losses["binary_loss"] = sum(binary_loss) / len(binary_loss)

        return obj_dists, rel_dists, add_losses


@registry.ROI_RELATION_PREDICTOR.register("CausalAnalysisPredictor")
class CausalAnalysisPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(CausalAnalysisPredictor, self).__init__()
        self.cfg = config
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.spatial_for_vision = config.MODEL.ROI_RELATION_HEAD.CAUSAL.SPATIAL_FOR_VISION
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.fusion_type = config.MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE
        self.separate_spatial = config.MODEL.ROI_RELATION_HEAD.CAUSAL.SEPARATE_SPATIAL
        self.use_vtranse = config.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER == "vtranse"
        self.effect_type = config.MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE
        
        assert in_channels is not None
        num_inputs = in_channels

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
        assert self.num_obj_cls==len(obj_classes)
        assert self.num_rel_cls==len(rel_classes)
        # init contextual lstm encoding
        if config.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER == "motifs":
            self.context_layer = LSTMContext(config, obj_classes, rel_classes, in_channels)
        elif config.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER == "vctree":
            self.context_layer = VCTreeLSTMContext(config, obj_classes, rel_classes, statistics, in_channels)
        elif config.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER == "vtranse":
            self.context_layer = VTransEFeature(config, obj_classes, rel_classes, in_channels)
        else:
            print('ERROR: Invalid Context Layer')

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        
        if self.use_vtranse:
            self.edge_dim = self.pooling_dim
            self.post_emb = nn.Linear(self.hidden_dim, self.pooling_dim * 2)
            self.ctx_compress = nn.Linear(self.pooling_dim, self.num_rel_cls, bias=False)
        else:
            self.edge_dim = self.hidden_dim
            self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
            self.post_cat = nn.Sequential(*[nn.Linear(self.hidden_dim * 2, self.pooling_dim),
                                            nn.ReLU(inplace=True),])
            self.ctx_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.vis_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)

        if self.fusion_type == 'gate':
            self.ctx_gate_fc = nn.Linear(self.pooling_dim, self.num_rel_cls)
            layer_init(self.ctx_gate_fc, xavier=True)
        
        # initialize layer parameters 
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        if not self.use_vtranse:
            layer_init(self.post_cat[0], xavier=True)
            layer_init(self.ctx_compress, xavier=True)
        layer_init(self.vis_compress, xavier=True)
        
        assert self.pooling_dim == config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM

        # convey statistics into FrequencyBias to avoid loading again
        self.freq_bias = FrequencyBias(config, statistics)

        # add spatial emb for visual feature
        if self.spatial_for_vision:
            self.spt_emb = nn.Sequential(*[nn.Linear(32, self.hidden_dim), 
                                            nn.ReLU(inplace=True),
                                            nn.Linear(self.hidden_dim, self.pooling_dim),
                                            nn.ReLU(inplace=True)
                                        ])
            layer_init(self.spt_emb[0], xavier=True)
            layer_init(self.spt_emb[2], xavier=True)

        self.label_smooth_loss = Label_Smoothing_Regression(e=1.0)

        # untreated average features
        self.effect_analysis = config.MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_ANALYSIS
        self.average_ratio = 0.0005

        self.register_buffer("untreated_spt", torch.zeros(32))
        self.register_buffer("untreated_conv_spt", torch.zeros(self.pooling_dim))
        self.register_buffer("avg_post_ctx", torch.zeros(self.pooling_dim))
        self.register_buffer("untreated_feat", torch.zeros(self.pooling_dim))

        
    def pair_feature_generate(self, roi_features, proposals, rel_pair_idxs, num_objs, obj_boxs, logger, ctx_average=False):
        # encode context infomation
        obj_dists, obj_preds, edge_ctx, binary_preds = self.context_layer(roi_features, proposals, rel_pair_idxs, logger, ctx_average=ctx_average)
        obj_dist_prob = F.softmax(obj_dists, dim=-1)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.edge_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.edge_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.edge_dim)
        # split
        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)
        obj_prob_list = obj_dist_prob.split(num_objs, dim=0)
        obj_dist_list = obj_dists.split(num_objs, dim=0)
        ctx_reps = []
        pair_preds = []
        pair_obj_probs = []
        pair_bboxs_info = []
        for pair_idx, head_rep, tail_rep, obj_pred, obj_box, obj_prob in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds, obj_boxs, obj_prob_list):
            if self.use_vtranse:
                ctx_reps.append( head_rep[pair_idx[:,0]] - tail_rep[pair_idx[:,1]] )
            else:
                ctx_reps.append( torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1) )
            pair_preds.append( torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1) )
            pair_obj_probs.append( torch.stack((obj_prob[pair_idx[:,0]], obj_prob[pair_idx[:,1]]), dim=2) )
            pair_bboxs_info.append( get_box_pair_info(obj_box[pair_idx[:,0]], obj_box[pair_idx[:,1]]) )
        pair_obj_probs = cat(pair_obj_probs, dim=0)
        pair_bbox = cat(pair_bboxs_info, dim=0)
        pair_pred = cat(pair_preds, dim=0)
        ctx_rep = cat(ctx_reps, dim=0)
        if self.use_vtranse:
            post_ctx_rep = ctx_rep
        else:
            post_ctx_rep = self.post_cat(ctx_rep)

        return post_ctx_rep, pair_pred, pair_bbox, pair_obj_probs, binary_preds, obj_dist_prob, edge_rep, obj_dist_list
        
        

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        obj_boxs = [get_box_info(p.bbox, need_norm=True, proposal=p) for p in proposals]

        assert len(num_rels) == len(num_objs)

        post_ctx_rep, pair_pred, pair_bbox, pair_obj_probs, binary_preds, obj_dist_prob, edge_rep, obj_dist_list = self.pair_feature_generate(roi_features, proposals, rel_pair_idxs, num_objs, obj_boxs, logger)

        if (not self.training) and self.effect_analysis:
            with torch.no_grad():
                avg_post_ctx_rep, _, _, avg_pair_obj_prob, _, _, _, _ = self.pair_feature_generate(roi_features, proposals, rel_pair_idxs, num_objs, obj_boxs, logger, ctx_average=True)

        if self.separate_spatial:
            union_features, spatial_conv_feats = union_features
            post_ctx_rep = post_ctx_rep * spatial_conv_feats
        
        if self.spatial_for_vision:
            post_ctx_rep = post_ctx_rep * self.spt_emb(pair_bbox)

        rel_dists = self.calculate_logits(union_features, post_ctx_rep, pair_pred, use_label_dist=False)
        rel_dist_list = rel_dists.split(num_rels, dim=0)

        add_losses = {}
        # additional loss
        if self.training:
            rel_labels = cat(rel_labels, dim=0)

            # binary loss for VCTree
            if binary_preds is not None:
                binary_loss = []
                for bi_gt, bi_pred in zip(rel_binarys, binary_preds):
                    bi_gt = (bi_gt > 0).float()
                    binary_loss.append(F.binary_cross_entropy_with_logits(bi_pred, bi_gt))
                add_losses["binary_loss"] = sum(binary_loss) / len(binary_loss)

            # branch constraint: make sure each branch can predict independently
            add_losses['auxiliary_ctx'] = F.cross_entropy(self.ctx_compress(post_ctx_rep), rel_labels)
            if not (self.fusion_type == 'gate'):
                add_losses['auxiliary_vis'] = F.cross_entropy(self.vis_compress(union_features), rel_labels)
                add_losses['auxiliary_frq'] = F.cross_entropy(self.freq_bias.index_with_labels(pair_pred.long()), rel_labels)

            # untreated average feature
            if self.spatial_for_vision:
                self.untreated_spt = self.moving_average(self.untreated_spt, pair_bbox)
            if self.separate_spatial:
                self.untreated_conv_spt = self.moving_average(self.untreated_conv_spt, spatial_conv_feats)
            self.avg_post_ctx = self.moving_average(self.avg_post_ctx, post_ctx_rep)
            self.untreated_feat = self.moving_average(self.untreated_feat, union_features)

        elif self.effect_analysis:
            with torch.no_grad():
                # untreated spatial
                if self.spatial_for_vision:
                    avg_spt_rep = self.spt_emb(self.untreated_spt.clone().detach().view(1, -1))
                # untreated context
                avg_ctx_rep = avg_post_ctx_rep * avg_spt_rep if self.spatial_for_vision else avg_post_ctx_rep  
                avg_ctx_rep = avg_ctx_rep * self.untreated_conv_spt.clone().detach().view(1, -1) if self.separate_spatial else avg_ctx_rep
                # untreated visual
                avg_vis_rep = self.untreated_feat.clone().detach().view(1, -1)
                # untreated category dist
                avg_frq_rep = avg_pair_obj_prob

            if self.effect_type == 'TDE':   # TDE of CTX
                rel_dists = self.calculate_logits(union_features, post_ctx_rep, pair_obj_probs) - self.calculate_logits(union_features, avg_ctx_rep, pair_obj_probs)
            elif self.effect_type == 'NIE': # NIE of FRQ
                rel_dists = self.calculate_logits(union_features, avg_ctx_rep, pair_obj_probs) - self.calculate_logits(union_features, avg_ctx_rep, avg_frq_rep)
            elif self.effect_type == 'TE':  # Total Effect
                rel_dists = self.calculate_logits(union_features, post_ctx_rep, pair_obj_probs) - self.calculate_logits(union_features, avg_ctx_rep, avg_frq_rep)
            else:
                assert self.effect_type == 'none'
                pass
            rel_dist_list = rel_dists.split(num_rels, dim=0)

        return obj_dist_list, rel_dist_list, add_losses

    def moving_average(self, holder, input):
        assert len(input.shape) == 2
        with torch.no_grad():
            holder = holder * (1 - self.average_ratio) + self.average_ratio * input.mean(0).view(-1)
        return holder

    def calculate_logits(self, vis_rep, ctx_rep, frq_rep, use_label_dist=True, mean_ctx=False):
        if use_label_dist:
            frq_dists = self.freq_bias.index_with_probability(frq_rep)
        else:
            frq_dists = self.freq_bias.index_with_labels(frq_rep.long())

        if mean_ctx:
            ctx_rep = ctx_rep.mean(-1).unsqueeze(-1)
        vis_dists = self.vis_compress(vis_rep)
        ctx_dists = self.ctx_compress(ctx_rep)

        if self.fusion_type == 'gate':
            ctx_gate_dists = self.ctx_gate_fc(ctx_rep)
            union_dists = ctx_dists * torch.sigmoid(vis_dists + frq_dists + ctx_gate_dists)
            #union_dists = (ctx_dists.exp() * torch.sigmoid(vis_dists + frq_dists + ctx_constraint) + 1e-9).log()    # improve on zero-shot, but low mean recall and TDE recall
            #union_dists = ctx_dists * torch.sigmoid(vis_dists * frq_dists)                                          # best conventional Recall results
            #union_dists = (ctx_dists.exp() + vis_dists.exp() + frq_dists.exp() + 1e-9).log()                        # good zero-shot Recall
            #union_dists = ctx_dists * torch.max(torch.sigmoid(vis_dists), torch.sigmoid(frq_dists))                 # good zero-shot Recall
            #union_dists = ctx_dists * torch.sigmoid(vis_dists) * torch.sigmoid(frq_dists)                           # balanced recall and mean recall
            #union_dists = ctx_dists * (torch.sigmoid(vis_dists) + torch.sigmoid(frq_dists)) / 2.0                   # good zero-shot Recall
            #union_dists = ctx_dists * torch.sigmoid((vis_dists.exp() + frq_dists.exp() + 1e-9).log())               # good zero-shot Recall, bad for all of the rest
            
        elif self.fusion_type == 'sum':
            union_dists = vis_dists + ctx_dists + frq_dists
        else:
            print('invalid fusion type')

        return union_dists

    def binary_ce_loss(self, logits, gt):
        batch_size, num_cat = logits.shape
        answer = torch.zeros((batch_size, num_cat), device=gt.device).float()
        answer[torch.arange(batch_size, device=gt.device), gt.long()] = 1.0
        return F.binary_cross_entropy_with_logits(logits, answer) * num_cat

    def fusion(self, x, y):
        return F.relu(x + y) - (x - y) ** 2


def make_roi_relation_predictor(cfg, in_channels):
    func = registry.ROI_RELATION_PREDICTOR[cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR]
    return func(cfg, in_channels)
