import logging
import os
import torch
import numpy as np
import json
from tqdm import tqdm
from functools import reduce
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from maskrcnn_benchmark.data import get_dataset_statistics
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.utils.miscellaneous import intersect_2d, argsort_desc, bbox_overlaps
from abc import ABC, abstractmethod

class SceneGraphEvaluation(ABC):

    def __init__(self, result_dict):
        super().__init__()
        self.result_dict = result_dict

    @abstractmethod
    def register_container(self, mode):
        print('Register Result Container')
        pass

    @abstractmethod
    def generate_print_string(self, mode):
        print('Generate Print String')
        pass

class SGRecall(SceneGraphEvaluation):

    def __init__(self, result_dict):
        super(SGRecall, self).__init__(result_dict)

    def register_container(self, mode):
        self.result_dict[mode + '_recall'] = {20: [], 50: [], 100: []}

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[mode + '_recall'].items():
            result_str += '    R @ %d: %.4f; ' % (k, np.mean(v))
        result_str += ' for mode=%s, type=Recall(Main).' % mode
        result_str += '\n'
        return result_str

    def calculate_recall(self, global_container, local_container, mode):
        pred_rel_inds = local_container['pred_rel_inds']
        rel_scores = local_container['rel_scores']
        gt_rels = local_container['gt_rels']
        gt_classes = local_container['gt_classes']
        gt_boxes = local_container['gt_boxes']
        pred_classes = local_container['pred_classes']
        pred_boxes = local_container['pred_boxes']
        obj_scores = local_container['obj_scores']
        iou_thres = global_container['iou_thres']
        pred_rels = np.column_stack((pred_rel_inds, 1 + rel_scores[:, 1:].argmax(1)))
        pred_scores = rel_scores[:, 1:].max(1)
        gt_triplets, gt_triplet_boxes, _ = _triplet(gt_rels, gt_classes, gt_boxes)
        local_container['gt_triplets'] = gt_triplets
        local_container['gt_triplet_boxes'] = gt_triplet_boxes
        pred_triplets, pred_triplet_boxes, pred_triplet_scores = _triplet(pred_rels, pred_classes, pred_boxes, pred_scores, obj_scores)
        pred_to_gt = _compute_pred_matches(gt_triplets, pred_triplets, gt_triplet_boxes, pred_triplet_boxes, iou_thres, phrdet=mode == 'phrdet')
        local_container['pred_to_gt'] = pred_to_gt
        for k in self.result_dict[mode + '_recall']:
            match = reduce(np.union1d, pred_to_gt[:k])
            rec_i = float(len(match)) / float(gt_rels.shape[0])
            self.result_dict[mode + '_recall'][k].append(rec_i)
        return local_container

class SGNoGraphConstraintRecall(SceneGraphEvaluation):

    def __init__(self, result_dict):
        super(SGNoGraphConstraintRecall, self).__init__(result_dict)

    def register_container(self, mode):
        self.result_dict[mode + '_recall_nogc'] = {20: [], 50: [], 100: []}

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[mode + '_recall_nogc'].items():
            result_str += ' ng-R @ %d: %.4f; ' % (k, np.mean(v))
        result_str += ' for mode=%s, type=No Graph Constraint Recall(Main).' % mode
        result_str += '\n'
        return result_str

    def calculate_recall(self, global_container, local_container, mode):
        obj_scores = local_container['obj_scores']
        pred_rel_inds = local_container['pred_rel_inds']
        rel_scores = local_container['rel_scores']
        pred_boxes = local_container['pred_boxes']
        pred_classes = local_container['pred_classes']
        gt_rels = local_container['gt_rels']
        obj_scores_per_rel = obj_scores[pred_rel_inds].prod(1)
        nogc_overall_scores = obj_scores_per_rel[:, None] * rel_scores[:, 1:]
        nogc_score_inds = argsort_desc(nogc_overall_scores)[:100]
        nogc_pred_rels = np.column_stack((pred_rel_inds[nogc_score_inds[:, 0]], nogc_score_inds[:, 1] + 1))
        nogc_pred_scores = rel_scores[nogc_score_inds[:, 0], nogc_score_inds[:, 1] + 1]
        nogc_pred_triplets, nogc_pred_triplet_boxes, _ = _triplet(nogc_pred_rels, pred_classes, pred_boxes, nogc_pred_scores, obj_scores)
        gt_triplets = local_container['gt_triplets']
        gt_triplet_boxes = local_container['gt_triplet_boxes']
        iou_thres = global_container['iou_thres']
        nogc_pred_to_gt = _compute_pred_matches(gt_triplets, nogc_pred_triplets, gt_triplet_boxes, nogc_pred_triplet_boxes, iou_thres, phrdet=mode == 'phrdet')
        local_container['nogc_pred_to_gt'] = nogc_pred_to_gt
        for k in self.result_dict[mode + '_recall_nogc']:
            match = reduce(np.union1d, nogc_pred_to_gt[:k])
            rec_i = float(len(match)) / float(gt_rels.shape[0])
            self.result_dict[mode + '_recall_nogc'][k].append(rec_i)
        return local_container

class SGZeroShotRecall(SceneGraphEvaluation):

    def __init__(self, result_dict):
        super(SGZeroShotRecall, self).__init__(result_dict)

    def register_container(self, mode):
        self.result_dict[mode + '_zeroshot_recall'] = {20: [], 50: [], 100: []}

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[mode + '_zeroshot_recall'].items():
            result_str += '   zR @ %d: %.4f; ' % (k, np.mean(v))
        result_str += ' for mode=%s, type=Zero Shot Recall.' % mode
        result_str += '\n'
        return result_str

    def prepare_zeroshot(self, global_container, local_container):
        gt_rels = local_container['gt_rels']
        gt_classes = local_container['gt_classes']
        zeroshot_triplets = global_container['zeroshot_triplet']
        sub_id, ob_id, pred_label = (gt_rels[:, 0], gt_rels[:, 1], gt_rels[:, 2])
        gt_triplets = np.column_stack((gt_classes[sub_id], gt_classes[ob_id], pred_label))
        self.zeroshot_idx = np.where(intersect_2d(gt_triplets, zeroshot_triplets).sum(-1) > 0)[0].tolist()

    def calculate_recall(self, global_container, local_container, mode):
        pred_to_gt = local_container['pred_to_gt']
        for k in self.result_dict[mode + '_zeroshot_recall']:
            match = reduce(np.union1d, pred_to_gt[:k])
            if len(self.zeroshot_idx) > 0:
                if not isinstance(match, (list, tuple)):
                    match_list = match.tolist()
                else:
                    match_list = match
                zeroshot_match = len(self.zeroshot_idx) + len(match_list) - len(set(self.zeroshot_idx + match_list))
                zero_rec_i = float(zeroshot_match) / float(len(self.zeroshot_idx))
                self.result_dict[mode + '_zeroshot_recall'][k].append(zero_rec_i)

class SGNGZeroShotRecall(SceneGraphEvaluation):

    def __init__(self, result_dict):
        super(SGNGZeroShotRecall, self).__init__(result_dict)

    def register_container(self, mode):
        self.result_dict[mode + '_ng_zeroshot_recall'] = {20: [], 50: [], 100: []}

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[mode + '_ng_zeroshot_recall'].items():
            result_str += 'ng-zR @ %d: %.4f; ' % (k, np.mean(v))
        result_str += ' for mode=%s, type=No Graph Constraint Zero Shot Recall.' % mode
        result_str += '\n'
        return result_str

    def prepare_zeroshot(self, global_container, local_container):
        gt_rels = local_container['gt_rels']
        gt_classes = local_container['gt_classes']
        zeroshot_triplets = global_container['zeroshot_triplet']
        sub_id, ob_id, pred_label = (gt_rels[:, 0], gt_rels[:, 1], gt_rels[:, 2])
        gt_triplets = np.column_stack((gt_classes[sub_id], gt_classes[ob_id], pred_label))
        self.zeroshot_idx = np.where(intersect_2d(gt_triplets, zeroshot_triplets).sum(-1) > 0)[0].tolist()

    def calculate_recall(self, global_container, local_container, mode):
        pred_to_gt = local_container['nogc_pred_to_gt']
        for k in self.result_dict[mode + '_ng_zeroshot_recall']:
            match = reduce(np.union1d, pred_to_gt[:k])
            if len(self.zeroshot_idx) > 0:
                if not isinstance(match, (list, tuple)):
                    match_list = match.tolist()
                else:
                    match_list = match
                zeroshot_match = len(self.zeroshot_idx) + len(match_list) - len(set(self.zeroshot_idx + match_list))
                zero_rec_i = float(zeroshot_match) / float(len(self.zeroshot_idx))
                self.result_dict[mode + '_ng_zeroshot_recall'][k].append(zero_rec_i)

class SGPairAccuracy(SceneGraphEvaluation):

    def __init__(self, result_dict):
        super(SGPairAccuracy, self).__init__(result_dict)

    def register_container(self, mode):
        self.result_dict[mode + '_accuracy_hit'] = {20: [], 50: [], 100: []}
        self.result_dict[mode + '_accuracy_count'] = {20: [], 50: [], 100: []}

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[mode + '_accuracy_hit'].items():
            a_hit = np.mean(v)
            a_count = np.mean(self.result_dict[mode + '_accuracy_count'][k])
            result_str += '    A @ %d: %.4f; ' % (k, a_hit / a_count)
        result_str += ' for mode=%s, type=TopK Accuracy.' % mode
        result_str += '\n'
        return result_str

    def prepare_gtpair(self, local_container):
        pred_pair_idx = local_container['pred_rel_inds'][:, 0] * 1024 + local_container['pred_rel_inds'][:, 1]
        gt_pair_idx = local_container['gt_rels'][:, 0] * 1024 + local_container['gt_rels'][:, 1]
        self.pred_pair_in_gt = (pred_pair_idx[:, None] == gt_pair_idx[None, :]).sum(-1) > 0

    def calculate_recall(self, global_container, local_container, mode):
        pred_to_gt = local_container['pred_to_gt']
        gt_rels = local_container['gt_rels']
        for k in self.result_dict[mode + '_accuracy_hit']:
            if mode != 'sgdet':
                gt_pair_pred_to_gt = []
                for p, flag in zip(pred_to_gt, self.pred_pair_in_gt):
                    if flag:
                        gt_pair_pred_to_gt.append(p)
                if len(gt_pair_pred_to_gt) > 0:
                    gt_pair_match = reduce(np.union1d, gt_pair_pred_to_gt[:k])
                else:
                    gt_pair_match = []
                self.result_dict[mode + '_accuracy_hit'][k].append(float(len(gt_pair_match)))
                self.result_dict[mode + '_accuracy_count'][k].append(float(gt_rels.shape[0]))

class SGMeanRecall(SceneGraphEvaluation):

    def __init__(self, result_dict, num_rel, ind_to_predicates, print_detail=False):
        super(SGMeanRecall, self).__init__(result_dict)
        self.num_rel = num_rel
        self.print_detail = print_detail
        self.rel_name_list = ind_to_predicates[1:]

    def register_container(self, mode):
        self.result_dict[mode + '_mean_recall'] = {20: 0.0, 50: 0.0, 100: 0.0}
        self.result_dict[mode + '_mean_recall_collect'] = {20: [[] for i in range(self.num_rel)], 50: [[] for i in range(self.num_rel)], 100: [[] for i in range(self.num_rel)]}
        self.result_dict[mode + '_mean_recall_list'] = {20: [], 50: [], 100: []}

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[mode + '_mean_recall'].items():
            result_str += '   mR @ %d: %.4f; ' % (k, float(v))
        result_str += ' for mode=%s, type=Mean Recall.' % mode
        result_str += '\n'
        if self.print_detail:
            result_str += '----------------------- Details ------------------------\n'
            for n, r in zip(self.rel_name_list, self.result_dict[mode + '_mean_recall_list'][100]):
                result_str += '({}:{:.4f}) '.format(str(n), r)
            result_str += '\n'
            result_str += '--------------------------------------------------------\n'
        return result_str

    def collect_mean_recall_items(self, global_container, local_container, mode):
        pred_to_gt = local_container['pred_to_gt']
        gt_rels = local_container['gt_rels']
        for k in self.result_dict[mode + '_mean_recall_collect']:
            match = reduce(np.union1d, pred_to_gt[:k])
            recall_hit = [0] * self.num_rel
            recall_count = [0] * self.num_rel
            for idx in range(gt_rels.shape[0]):
                local_label = gt_rels[idx, 2]
                recall_count[int(local_label)] += 1
                recall_count[0] += 1
            for idx in range(len(match)):
                local_label = gt_rels[int(match[idx]), 2]
                recall_hit[int(local_label)] += 1
                recall_hit[0] += 1
            for n in range(self.num_rel):
                if recall_count[n] > 0:
                    self.result_dict[mode + '_mean_recall_collect'][k][n].append(float(recall_hit[n] / recall_count[n]))

    def calculate_mean_recall(self, mode):
        for k, v in self.result_dict[mode + '_mean_recall'].items():
            sum_recall = 0
            num_rel_no_bg = self.num_rel - 1
            for idx in range(num_rel_no_bg):
                if len(self.result_dict[mode + '_mean_recall_collect'][k][idx + 1]) == 0:
                    tmp_recall = 0.0
                else:
                    tmp_recall = np.mean(self.result_dict[mode + '_mean_recall_collect'][k][idx + 1])
                self.result_dict[mode + '_mean_recall_list'][k].append(tmp_recall)
                sum_recall += tmp_recall
            self.result_dict[mode + '_mean_recall'][k] = sum_recall / float(num_rel_no_bg)
        return

class SGNGMeanRecall(SceneGraphEvaluation):

    def __init__(self, result_dict, num_rel, ind_to_predicates, print_detail=False):
        super(SGNGMeanRecall, self).__init__(result_dict)
        self.num_rel = num_rel
        self.print_detail = print_detail
        self.rel_name_list = ind_to_predicates[1:]

    def register_container(self, mode):
        self.result_dict[mode + '_ng_mean_recall'] = {20: 0.0, 50: 0.0, 100: 0.0}
        self.result_dict[mode + '_ng_mean_recall_collect'] = {20: [[] for i in range(self.num_rel)], 50: [[] for i in range(self.num_rel)], 100: [[] for i in range(self.num_rel)]}
        self.result_dict[mode + '_ng_mean_recall_list'] = {20: [], 50: [], 100: []}

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[mode + '_ng_mean_recall'].items():
            result_str += 'ng-mR @ %d: %.4f; ' % (k, float(v))
        result_str += ' for mode=%s, type=No Graph Constraint Mean Recall.' % mode
        result_str += '\n'
        if self.print_detail:
            result_str += '----------------------- Details ------------------------\n'
            for n, r in zip(self.rel_name_list, self.result_dict[mode + '_ng_mean_recall_list'][100]):
                result_str += '({}:{:.4f}) '.format(str(n), r)
            result_str += '\n'
            result_str += '--------------------------------------------------------\n'
        return result_str

    def collect_mean_recall_items(self, global_container, local_container, mode):
        pred_to_gt = local_container['nogc_pred_to_gt']
        gt_rels = local_container['gt_rels']
        for k in self.result_dict[mode + '_ng_mean_recall_collect']:
            match = reduce(np.union1d, pred_to_gt[:k])
            recall_hit = [0] * self.num_rel
            recall_count = [0] * self.num_rel
            for idx in range(gt_rels.shape[0]):
                local_label = gt_rels[idx, 2]
                recall_count[int(local_label)] += 1
                recall_count[0] += 1
            for idx in range(len(match)):
                local_label = gt_rels[int(match[idx]), 2]
                recall_hit[int(local_label)] += 1
                recall_hit[0] += 1
            for n in range(self.num_rel):
                if recall_count[n] > 0:
                    self.result_dict[mode + '_ng_mean_recall_collect'][k][n].append(float(recall_hit[n] / recall_count[n]))

    def calculate_mean_recall(self, mode):
        for k, v in self.result_dict[mode + '_ng_mean_recall'].items():
            sum_recall = 0
            num_rel_no_bg = self.num_rel - 1
            for idx in range(num_rel_no_bg):
                if len(self.result_dict[mode + '_ng_mean_recall_collect'][k][idx + 1]) == 0:
                    tmp_recall = 0.0
                else:
                    tmp_recall = np.mean(self.result_dict[mode + '_ng_mean_recall_collect'][k][idx + 1])
                self.result_dict[mode + '_ng_mean_recall_list'][k].append(tmp_recall)
                sum_recall += tmp_recall
            self.result_dict[mode + '_ng_mean_recall'][k] = sum_recall / float(num_rel_no_bg)
        return

class SGAccumulateRecall(SceneGraphEvaluation):

    def __init__(self, result_dict):
        super(SGAccumulateRecall, self).__init__(result_dict)

    def register_container(self, mode):
        self.result_dict[mode + '_accumulate_recall'] = {20: 0.0, 50: 0.0, 100: 0.0}

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[mode + '_accumulate_recall'].items():
            result_str += '   aR @ %d: %.4f; ' % (k, float(v))
        result_str += ' for mode=%s, type=Accumulate Recall.' % mode
        result_str += '\n'
        return result_str

    def calculate_accumulate(self, mode):
        for k, v in self.result_dict[mode + '_accumulate_recall'].items():
            self.result_dict[mode + '_accumulate_recall'][k] = float(self.result_dict[mode + '_recall_hit'][k][0]) / float(self.result_dict[mode + '_recall_count'][k][0] + 1e-10)
        return

class SGSemanticRecall(SceneGraphEvaluation):

    def __init__(self, result_dict, similarity_matrix_path=None):
        super(SGSemanticRecall, self).__init__(result_dict)
        if similarity_matrix_path is None:
            similarity_matrix_path = '/public/home/v-chengwy/MLLM_Reasoning/CLIP_main/normalized_similarity_results/normalized_similarity_matrix_min_max_thr0.8.npy'
        try:
            self.similarity_matrix = np.load(similarity_matrix_path)
            print(f'Loaded similarity matrix from {similarity_matrix_path}, shape: {self.similarity_matrix.shape}')
        except Exception as e:
            print(f'Warning: Could not load similarity matrix from {similarity_matrix_path}: {e}')
            print('Using identity matrix as fallback (exact match only)')
            self.similarity_matrix = None

    def register_container(self, mode):
        self.result_dict[mode + '_semantic_recall'] = {20: [], 50: [], 100: []}

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[mode + '_semantic_recall'].items():
            result_str += ' semR @ %d: %.4f; ' % (k, np.mean(v))
        result_str += ' for mode=%s, type=Semantic Recall.' % mode
        result_str += '\n'
        return result_str

    def _get_similarity_vectorized(self, pred_labels, gt_label):
        if self.similarity_matrix is None:
            return (pred_labels == gt_label).astype(np.float32)
        pred_idx = pred_labels.astype(np.int32) - 1
        gt_idx = int(gt_label) - 1
        if gt_idx < 0 or gt_idx >= self.similarity_matrix.shape[1]:
            return np.zeros(len(pred_labels), dtype=np.float32)
        valid_mask = (pred_idx >= 0) & (pred_idx < self.similarity_matrix.shape[0])
        result = np.zeros(len(pred_labels), dtype=np.float32)
        if valid_mask.any():
            result[valid_mask] = self.similarity_matrix[pred_idx[valid_mask], gt_idx]
        return result

    def calculate_recall(self, global_container, local_container, mode):
        obj_scores = local_container['obj_scores']
        pred_rel_inds = local_container['pred_rel_inds']
        rel_scores = local_container['rel_scores']
        pred_boxes = local_container['pred_boxes']
        pred_classes = local_container['pred_classes']
        gt_rels = local_container['gt_rels']
        gt_classes = local_container['gt_classes']
        gt_boxes = local_container['gt_boxes']
        iou_thres = global_container['iou_thres']
        num_gt = gt_rels.shape[0]
        if num_gt == 0:
            for k in self.result_dict[mode + '_semantic_recall']:
                self.result_dict[mode + '_semantic_recall'][k].append(0.0)
            return local_container
        obj_scores_per_rel = obj_scores[pred_rel_inds].prod(1)
        nogc_overall_scores = obj_scores_per_rel[:, None] * rel_scores[:, 1:]
        nogc_score_inds = argsort_desc(nogc_overall_scores)[:100]
        nogc_pred_rels = np.column_stack((pred_rel_inds[nogc_score_inds[:, 0]], nogc_score_inds[:, 1] + 1))
        gt_triplets, gt_triplet_boxes, _ = _triplet(gt_rels, gt_classes, gt_boxes)
        local_container['gt_triplets'] = gt_triplets
        local_container['gt_triplet_boxes'] = gt_triplet_boxes
        pred_triplets = np.column_stack((pred_classes[nogc_pred_rels[:, 0]], nogc_pred_rels[:, 2], pred_classes[nogc_pred_rels[:, 1]]))
        pred_sub_boxes = pred_boxes[nogc_pred_rels[:, 0]]
        pred_obj_boxes = pred_boxes[nogc_pred_rels[:, 1]]
        gt_sub_boxes = gt_triplet_boxes[:, :4]
        gt_obj_boxes = gt_triplet_boxes[:, 4:]
        sub_ious = bbox_overlaps(gt_sub_boxes, pred_sub_boxes[:100])
        obj_ious = bbox_overlaps(gt_obj_boxes, pred_obj_boxes[:100])
        sub_class_match = gt_triplets[:, 0:1] == pred_triplets[:100, 0:1].T
        obj_class_match = gt_triplets[:, 2:3] == pred_triplets[:100, 2:3].T
        struct_match = sub_class_match & obj_class_match & (sub_ious >= iou_thres) & (obj_ious >= iou_thres)
        for k in self.result_dict[mode + '_semantic_recall']:
            k_struct_match = struct_match[:, :k]
            k_pred_labels = pred_triplets[:k, 1]
            total_semantic_score = 0.0
            for gt_idx in range(num_gt):
                matched_pred_indices = np.where(k_struct_match[gt_idx])[0]
                if len(matched_pred_indices) == 0:
                    continue
                gt_pred_label = gt_triplets[gt_idx, 1]
                matched_pred_labels = k_pred_labels[matched_pred_indices]
                similarities = self._get_similarity_vectorized(matched_pred_labels, gt_pred_label)
                total_semantic_score += similarities.max()
            semantic_recall = total_semantic_score / float(num_gt)
            self.result_dict[mode + '_semantic_recall'][k].append(semantic_recall)
        return local_container

class SGMeanSemanticRecall(SceneGraphEvaluation):

    def __init__(self, result_dict, num_rel, ind_to_predicates, similarity_matrix_path=None, print_detail=False):
        super(SGMeanSemanticRecall, self).__init__(result_dict)
        self.num_rel = num_rel
        self.print_detail = print_detail
        self.rel_name_list = ind_to_predicates[1:]
        if similarity_matrix_path is None:
            similarity_matrix_path = '/public/home/v-chengwy/MLLM_Reasoning/CLIP_main/normalized_similarity_results/normalized_similarity_matrix_min_max_thr0.8.npy'
        try:
            self.similarity_matrix = np.load(similarity_matrix_path)
            print(f'[MeanSemanticRecall] Loaded similarity matrix from {similarity_matrix_path}, shape: {self.similarity_matrix.shape}')
        except Exception as e:
            print(f'[MeanSemanticRecall] Warning: Could not load similarity matrix: {e}')
            self.similarity_matrix = None

    def register_container(self, mode):
        self.result_dict[mode + '_mean_semantic_recall'] = {20: 0.0, 50: 0.0, 100: 0.0}
        self.result_dict[mode + '_mean_semantic_recall_collect'] = {20: [[] for _ in range(self.num_rel)], 50: [[] for _ in range(self.num_rel)], 100: [[] for _ in range(self.num_rel)]}
        self.result_dict[mode + '_mean_semantic_recall_list'] = {20: [], 50: [], 100: []}

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[mode + '_mean_semantic_recall'].items():
            result_str += 'm-semR @ %d: %.4f; ' % (k, float(v))
        result_str += ' for mode=%s, type=Mean Semantic Recall.' % mode
        result_str += '\n'
        if self.print_detail:
            result_str += '----------------------- Details ------------------------\n'
            for n, r in zip(self.rel_name_list, self.result_dict[mode + '_mean_semantic_recall_list'][100]):
                result_str += '({}:{:.4f}) '.format(str(n), r)
            result_str += '\n'
            result_str += '--------------------------------------------------------\n'
        return result_str

    def _get_similarity_vectorized(self, pred_labels, gt_label):
        if self.similarity_matrix is None:
            return (pred_labels == gt_label).astype(np.float32)
        pred_idx = pred_labels.astype(np.int32) - 1
        gt_idx = int(gt_label) - 1
        if gt_idx < 0 or gt_idx >= self.similarity_matrix.shape[1]:
            return np.zeros(len(pred_labels), dtype=np.float32)
        valid_mask = (pred_idx >= 0) & (pred_idx < self.similarity_matrix.shape[0])
        result = np.zeros(len(pred_labels), dtype=np.float32)
        if valid_mask.any():
            result[valid_mask] = self.similarity_matrix[pred_idx[valid_mask], gt_idx]
        return result

    def collect_mean_recall_items(self, global_container, local_container, mode):
        obj_scores = local_container['obj_scores']
        pred_rel_inds = local_container['pred_rel_inds']
        rel_scores = local_container['rel_scores']
        pred_boxes = local_container['pred_boxes']
        pred_classes = local_container['pred_classes']
        gt_rels = local_container['gt_rels']
        gt_classes = local_container['gt_classes']
        gt_boxes = local_container['gt_boxes']
        iou_thres = global_container['iou_thres']
        num_gt = gt_rels.shape[0]
        if num_gt == 0:
            return
        obj_scores_per_rel = obj_scores[pred_rel_inds].prod(1)
        nogc_overall_scores = obj_scores_per_rel[:, None] * rel_scores[:, 1:]
        nogc_score_inds = argsort_desc(nogc_overall_scores)[:100]
        nogc_pred_rels = np.column_stack((pred_rel_inds[nogc_score_inds[:, 0]], nogc_score_inds[:, 1] + 1))
        gt_triplets, gt_triplet_boxes, _ = _triplet(gt_rels, gt_classes, gt_boxes)
        pred_triplets = np.column_stack((pred_classes[nogc_pred_rels[:, 0]], nogc_pred_rels[:, 2], pred_classes[nogc_pred_rels[:, 1]]))
        pred_sub_boxes = pred_boxes[nogc_pred_rels[:, 0]]
        pred_obj_boxes = pred_boxes[nogc_pred_rels[:, 1]]
        gt_sub_boxes = gt_triplet_boxes[:, :4]
        gt_obj_boxes = gt_triplet_boxes[:, 4:]
        sub_ious = bbox_overlaps(gt_sub_boxes, pred_sub_boxes[:100])
        obj_ious = bbox_overlaps(gt_obj_boxes, pred_obj_boxes[:100])
        sub_class_match = gt_triplets[:, 0:1] == pred_triplets[:100, 0:1].T
        obj_class_match = gt_triplets[:, 2:3] == pred_triplets[:100, 2:3].T
        struct_match = sub_class_match & obj_class_match & (sub_ious >= iou_thres) & (obj_ious >= iou_thres)
        for k in self.result_dict[mode + '_mean_semantic_recall_collect']:
            k_struct_match = struct_match[:, :k]
            k_pred_labels = pred_triplets[:k, 1]
            similarity_per_class = [[] for _ in range(self.num_rel)]
            for gt_idx in range(num_gt):
                matched_pred_indices = np.where(k_struct_match[gt_idx])[0]
                gt_pred_label = gt_triplets[gt_idx, 1]
                if len(matched_pred_indices) == 0:
                    max_similarity = 0.0
                else:
                    matched_pred_labels = k_pred_labels[matched_pred_indices]
                    similarities = self._get_similarity_vectorized(matched_pred_labels, gt_pred_label)
                    max_similarity = similarities.max()
                if int(gt_pred_label) < self.num_rel:
                    similarity_per_class[int(gt_pred_label)].append(max_similarity)
            for n in range(self.num_rel):
                if len(similarity_per_class[n]) > 0:
                    mean_sim = float(np.mean(similarity_per_class[n]))
                    self.result_dict[mode + '_mean_semantic_recall_collect'][k][n].append(mean_sim)

    def calculate_mean_recall(self, mode):
        for k in self.result_dict[mode + '_mean_semantic_recall']:
            sum_recall = 0
            num_rel_no_bg = self.num_rel - 1
            for idx in range(num_rel_no_bg):
                if len(self.result_dict[mode + '_mean_semantic_recall_collect'][k][idx + 1]) == 0:
                    tmp_recall = 0.0
                else:
                    tmp_recall = np.mean(self.result_dict[mode + '_mean_semantic_recall_collect'][k][idx + 1])
                self.result_dict[mode + '_mean_semantic_recall_list'][k].append(tmp_recall)
                sum_recall += tmp_recall
            self.result_dict[mode + '_mean_semantic_recall'][k] = sum_recall / float(num_rel_no_bg)
        return

class SGCategoryWeightedRecall(SceneGraphEvaluation):
    METRIC_NAME = 'DA-R'
    METRIC_TYPE = 'Diversity-Aware Recall'
    PREDICATE_CATEGORIES = {'Spatial-Positional': ['above', 'across', 'around', 'between', 'in', 'on', 'under', 'behind', 'in front of', 'near', 'near by', 'to left of', 'to right of', 'along', 'along back of', 'along bottom of', 'along edge of', 'along side of', 'along top of', 'on top of', 'on the bottom of', 'on the back of', 'on the front of', 'on the right side of', 'on the left side of', 'on the edge of', 'on the surface of', 'in the left side of', 'in the right side of', 'in the middle of', 'near the bottom of', 'near the edge of', 'near the front of', 'near the side of', 'near the top of', 'surrounded by'], 'Action-Interaction': ['biting', 'brushing', 'buying', 'carrying', 'catching', 'chasing', 'chewing', 'cleaning', 'climbing', 'cooking', 'cutting', 'decorating', 'drinking', 'eating', 'feeding', 'flying over', 'flying in', 'following', 'guiding', 'helping', 'herding', 'hitting', 'holding', 'hugging', 'jumping from', 'jumping over', 'kicking', 'kissing', 'leaning on', 'leaving', 'licking', 'opening', 'picking', 'pulling', 'pushing', 'reading', 'slicing', 'swinging', 'throwing', 'washing', 'serving', 'playing', 'playing at', 'playing in', 'playing in front of', 'playing on', 'playing with', 'playing near', 'talking to', 'says', 'running on', 'sitting on', 'eating at', 'eating with', 'eating from', 'walking in', 'walking in front of', 'walking on', 'touching', 'watching', 'driving', 'driving on', 'riding', 'entering', 'exiting', 'coming from', 'floating in', 'parked in', 'parked on', 'parked on side of', 'parked on top of', 'falling off', 'crossing', 'looking at', 'facing'], 'Attachment-Containment': ['attached to', 'attached to back of', 'attached to front of', 'attached to side of', 'mounted on', 'mounted on top of', 'hanging on', 'hanging over', 'hanging from', 'hanging in', 'covering', 'covered with', 'filled with', 'containing', 'connected to', 'growing in', 'growing on', 'growing on edge of', 'growing on side of', 'growing on top of', 'painted on', 'painted on side of', 'painted on top of', 'printed on', 'decorated with', 'show', 'written on', 'reflected in'], 'Ownership-Usage-Possession': ['has', 'held by', 'used by', 'part of', 'using', 'wearing', 'worn by', 'made of', 'belonging to'], 'Static-State': ['lying on', 'resting on', 'supporting', 'standing on', 'standing behind', 'standing on edge of', 'standing near']}

    def __init__(self, result_dict, ind_to_predicates, print_detail=False):
        super(SGCategoryWeightedRecall, self).__init__(result_dict)
        self.print_detail = print_detail
        self.ind_to_predicates = ind_to_predicates
        self.predicate_to_idx = {name.lower(): idx for idx, name in enumerate(ind_to_predicates)}
        self.pred_idx_to_category = {}
        self.category_names = list(self.PREDICATE_CATEGORIES.keys())
        for cat_idx, (cat_name, predicates) in enumerate(self.PREDICATE_CATEGORIES.items()):
            for pred_name in predicates:
                pred_name_lower = pred_name.lower()
                if pred_name_lower in self.predicate_to_idx:
                    pred_idx = self.predicate_to_idx[pred_name_lower]
                    self.pred_idx_to_category[pred_idx] = cat_idx
        print(f'[CategoryWeightedRecall] Mapped {len(self.pred_idx_to_category)} predicates to {len(self.category_names)} categories')

    def register_container(self, mode):
        self.result_dict[mode + '_category_weighted_recall'] = {20: [], 50: [], 100: []}

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[mode + '_category_weighted_recall'].items():
            result_str += '%s @ %d: %.4f; ' % (self.METRIC_NAME, k, np.mean(v) if len(v) > 0 else 0.0)
        result_str += ' for mode=%s, type=%s.' % (mode, self.METRIC_TYPE)
        result_str += '\n'
        return result_str

    def _get_predicate_category(self, pred_idx):
        return self.pred_idx_to_category.get(pred_idx, -1)

    def _aggregate_gt_object_pairs(self, gt_rels, gt_boxes):
        pair_to_predicates = {}
        for rel in gt_rels:
            sub_idx, obj_idx, pred_label = (int(rel[0]), int(rel[1]), int(rel[2]))
            sub_box = tuple(gt_boxes[sub_idx])
            obj_box = tuple(gt_boxes[obj_idx])
            pair_key = (sub_box, obj_box, sub_idx, obj_idx)
            if pair_key not in pair_to_predicates:
                pair_to_predicates[pair_key] = []
            pair_to_predicates[pair_key].append(pred_label)
        aggregated_pairs = []
        for (sub_box, obj_box, sub_idx, obj_idx), predicates in pair_to_predicates.items():
            categories = set()
            for pred in predicates:
                cat = self._get_predicate_category(pred)
                if cat >= 0:
                    categories.add(cat)
            num_cats = len(categories)
            num_preds = len(predicates)
            weight = num_cats + (num_preds - num_cats) * 0.5
            aggregated_pairs.append({'sub_idx': sub_idx, 'obj_idx': obj_idx, 'sub_box': sub_box, 'obj_box': obj_box, 'predicates': predicates, 'categories': categories, 'weight': weight, 'hit_categories': set(), 'hit_score': 0.0})
        return aggregated_pairs

    def calculate_recall(self, global_container, local_container, mode):
        obj_scores = local_container['obj_scores']
        pred_rel_inds = local_container['pred_rel_inds']
        rel_scores = local_container['rel_scores']
        pred_boxes = local_container['pred_boxes']
        pred_classes = local_container['pred_classes']
        gt_rels = local_container['gt_rels']
        gt_classes = local_container['gt_classes']
        gt_boxes = local_container['gt_boxes']
        iou_thres = global_container['iou_thres']
        if len(gt_rels) == 0:
            for k in self.result_dict[mode + '_category_weighted_recall']:
                self.result_dict[mode + '_category_weighted_recall'][k].append(0.0)
            return local_container
        aggregated_gt = self._aggregate_gt_object_pairs(gt_rels, gt_boxes)
        total_weight = sum((p['weight'] for p in aggregated_gt))
        if total_weight == 0:
            for k in self.result_dict[mode + '_category_weighted_recall']:
                self.result_dict[mode + '_category_weighted_recall'][k].append(0.0)
            return local_container
        obj_scores_per_rel = obj_scores[pred_rel_inds].prod(1)
        nogc_overall_scores = obj_scores_per_rel[:, None] * rel_scores[:, 1:]
        nogc_score_inds = argsort_desc(nogc_overall_scores)[:100]
        pred_info = []
        for i, (pair_idx, rel_idx) in enumerate(nogc_score_inds):
            sub_idx = pred_rel_inds[pair_idx, 0]
            obj_idx = pred_rel_inds[pair_idx, 1]
            pred_label = rel_idx + 1
            pred_info.append({'rank': i, 'sub_idx': sub_idx, 'obj_idx': obj_idx, 'sub_cls': pred_classes[sub_idx], 'obj_cls': pred_classes[obj_idx], 'sub_box': pred_boxes[sub_idx], 'obj_box': pred_boxes[obj_idx], 'pred_label': pred_label})
        gt_triple_index = {}
        for gt_idx, gt_pair in enumerate(aggregated_gt):
            sub_cls = gt_classes[gt_pair['sub_idx']]
            obj_cls = gt_classes[gt_pair['obj_idx']]
            for pred_label in gt_pair['predicates']:
                key = (sub_cls, obj_cls, pred_label)
                if key not in gt_triple_index:
                    gt_triple_index[key] = []
                gt_triple_index[key].append(gt_idx)
        gt_sub_boxes = np.array([list(p['sub_box']) for p in aggregated_gt])
        gt_obj_boxes = np.array([list(p['obj_box']) for p in aggregated_gt])
        for k in self.result_dict[mode + '_category_weighted_recall']:
            for gt_pair in aggregated_gt:
                gt_pair['hit_categories'] = set()
                gt_pair['hit_score'] = 0.0
                gt_pair['hit_predicates'] = set()
            for pred in pred_info[:k]:
                pred_label = pred['pred_label']
                pred_triple_key = (pred['sub_cls'], pred['obj_cls'], pred_label)
                if pred_triple_key not in gt_triple_index:
                    continue
                candidate_gt_indices = gt_triple_index[pred_triple_key]
                pred_sub_box = pred['sub_box'].reshape(1, 4)
                pred_obj_box = pred['obj_box'].reshape(1, 4)
                for gt_idx in candidate_gt_indices:
                    gt_pair = aggregated_gt[gt_idx]
                    gt_sub_box = gt_sub_boxes[gt_idx:gt_idx + 1]
                    gt_obj_box = gt_obj_boxes[gt_idx:gt_idx + 1]
                    sub_iou = bbox_overlaps(gt_sub_box, pred_sub_box)[0, 0]
                    obj_iou = bbox_overlaps(gt_obj_box, pred_obj_box)[0, 0]
                    if sub_iou < iou_thres or obj_iou < iou_thres:
                        continue
                    if pred_label in gt_pair['hit_predicates']:
                        continue
                    gt_pair['hit_predicates'].add(pred_label)
                    cat = self._get_predicate_category(pred_label)
                    if cat >= 0 and cat in gt_pair['categories']:
                        if cat not in gt_pair['hit_categories']:
                            gt_pair['hit_score'] += 1.0
                            gt_pair['hit_categories'].add(cat)
                        else:
                            gt_pair['hit_score'] += 0.5
                    else:
                        gt_pair['hit_score'] += 0.5
            total_hit_score = sum((min(p['hit_score'], p['weight']) for p in aggregated_gt))
            weighted_recall = total_hit_score / total_weight
            self.result_dict[mode + '_category_weighted_recall'][k].append(weighted_recall)
        return local_container

class SGNGCategoryWeightedMeanRecall(SceneGraphEvaluation):
    METRIC_NAME = 'DA-mR'
    METRIC_TYPE = 'Diversity-Aware Mean Recall'
    PREDICATE_CATEGORIES = SGCategoryWeightedRecall.PREDICATE_CATEGORIES

    def __init__(self, result_dict, ind_to_predicates, print_detail=False):
        super(SGNGCategoryWeightedMeanRecall, self).__init__(result_dict)
        self.print_detail = print_detail
        self.ind_to_predicates = ind_to_predicates
        self.num_categories = len(self.PREDICATE_CATEGORIES)
        self.category_names = list(self.PREDICATE_CATEGORIES.keys())
        self.predicate_to_idx = {name.lower(): idx for idx, name in enumerate(ind_to_predicates)}
        self.pred_idx_to_category = {}
        for cat_idx, (cat_name, predicates) in enumerate(self.PREDICATE_CATEGORIES.items()):
            for pred_name in predicates:
                pred_name_lower = pred_name.lower()
                if pred_name_lower in self.predicate_to_idx:
                    pred_idx = self.predicate_to_idx[pred_name_lower]
                    self.pred_idx_to_category[pred_idx] = cat_idx
        print(f'[NGCategoryWeightedMeanRecall] Mapped {len(self.pred_idx_to_category)} predicates to {self.num_categories} categories')

    def register_container(self, mode):
        self.result_dict[mode + '_ng_category_weighted_mean_recall'] = {20: 0.0, 50: 0.0, 100: 0.0}
        self.result_dict[mode + '_ng_cwmr_collect'] = {20: [[] for _ in range(self.num_categories)], 50: [[] for _ in range(self.num_categories)], 100: [[] for _ in range(self.num_categories)]}
        self.result_dict[mode + '_ng_cwmr_list'] = {20: [], 50: [], 100: []}

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[mode + '_ng_category_weighted_mean_recall'].items():
            result_str += '%s @ %d: %.4f; ' % (self.METRIC_NAME, k, float(v))
        result_str += ' for mode=%s, type=%s.' % (mode, self.METRIC_TYPE)
        result_str += '\n'
        if self.print_detail:
            result_str += '----------------------- Category Details ------------------------\n'
            for cat_name, recall in zip(self.category_names, self.result_dict[mode + '_ng_cwmr_list'][100]):
                result_str += '({}:{:.4f}) '.format(cat_name, recall)
            result_str += '\n'
            result_str += '----------------------------------------------------------------\n'
        return result_str

    def _get_predicate_category(self, pred_idx):
        return self.pred_idx_to_category.get(pred_idx, -1)

    def _aggregate_gt_object_pairs(self, gt_rels, gt_boxes):
        pair_to_predicates = {}
        for rel in gt_rels:
            sub_idx, obj_idx, pred_label = (int(rel[0]), int(rel[1]), int(rel[2]))
            sub_box = tuple(gt_boxes[sub_idx])
            obj_box = tuple(gt_boxes[obj_idx])
            pair_key = (sub_box, obj_box, sub_idx, obj_idx)
            if pair_key not in pair_to_predicates:
                pair_to_predicates[pair_key] = []
            pair_to_predicates[pair_key].append(pred_label)
        aggregated_pairs = []
        for (sub_box, obj_box, sub_idx, obj_idx), predicates in pair_to_predicates.items():
            category_pred_count = {}
            for pred in predicates:
                cat = self._get_predicate_category(pred)
                if cat >= 0:
                    category_pred_count[cat] = category_pred_count.get(cat, 0) + 1
            category_weights = {}
            for cat, count in category_pred_count.items():
                category_weights[cat] = 1.0 + (count - 1) * 0.5
            aggregated_pairs.append({'sub_idx': sub_idx, 'obj_idx': obj_idx, 'sub_box': sub_box, 'obj_box': obj_box, 'predicates': predicates, 'category_pred_count': category_pred_count, 'category_weights': category_weights})
        return aggregated_pairs

    def collect_mean_recall_items(self, global_container, local_container, mode):
        obj_scores = local_container['obj_scores']
        pred_rel_inds = local_container['pred_rel_inds']
        rel_scores = local_container['rel_scores']
        pred_boxes = local_container['pred_boxes']
        pred_classes = local_container['pred_classes']
        gt_rels = local_container['gt_rels']
        gt_classes = local_container['gt_classes']
        gt_boxes = local_container['gt_boxes']
        iou_thres = global_container['iou_thres']
        if len(gt_rels) == 0:
            return
        aggregated_gt = self._aggregate_gt_object_pairs(gt_rels, gt_boxes)
        has_any_category = any((len(p['category_weights']) > 0 for p in aggregated_gt))
        if not has_any_category:
            return
        obj_scores_per_rel = obj_scores[pred_rel_inds].prod(1)
        nogc_overall_scores = obj_scores_per_rel[:, None] * rel_scores[:, 1:]
        nogc_score_inds = argsort_desc(nogc_overall_scores)[:100]
        pred_info = []
        for i, (pair_idx, rel_idx) in enumerate(nogc_score_inds):
            sub_idx = pred_rel_inds[pair_idx, 0]
            obj_idx = pred_rel_inds[pair_idx, 1]
            pred_label = rel_idx + 1
            pred_info.append({'rank': i, 'sub_idx': sub_idx, 'obj_idx': obj_idx, 'sub_cls': pred_classes[sub_idx], 'obj_cls': pred_classes[obj_idx], 'sub_box': pred_boxes[sub_idx], 'obj_box': pred_boxes[obj_idx], 'pred_label': pred_label})
        gt_triple_index = {}
        for gt_idx, gt_pair in enumerate(aggregated_gt):
            sub_cls = gt_classes[gt_pair['sub_idx']]
            obj_cls = gt_classes[gt_pair['obj_idx']]
            for pred_label in gt_pair['predicates']:
                key = (sub_cls, obj_cls, pred_label)
                if key not in gt_triple_index:
                    gt_triple_index[key] = []
                gt_triple_index[key].append(gt_idx)
        gt_sub_boxes = np.array([list(p['sub_box']) for p in aggregated_gt])
        gt_obj_boxes = np.array([list(p['obj_box']) for p in aggregated_gt])
        for k in self.result_dict[mode + '_ng_cwmr_collect']:
            for gt_pair in aggregated_gt:
                gt_pair['hit_categories'] = set()
                gt_pair['hit_predicates'] = set()
                gt_pair['category_hit_scores'] = {cat: 0.0 for cat in gt_pair['category_weights']}
            for pred in pred_info[:k]:
                pred_label = pred['pred_label']
                pred_triple_key = (pred['sub_cls'], pred['obj_cls'], pred_label)
                if pred_triple_key not in gt_triple_index:
                    continue
                candidate_gt_indices = gt_triple_index[pred_triple_key]
                pred_sub_box = pred['sub_box'].reshape(1, 4)
                pred_obj_box = pred['obj_box'].reshape(1, 4)
                for gt_idx in candidate_gt_indices:
                    gt_pair = aggregated_gt[gt_idx]
                    gt_sub_box = gt_sub_boxes[gt_idx:gt_idx + 1]
                    gt_obj_box = gt_obj_boxes[gt_idx:gt_idx + 1]
                    sub_iou = bbox_overlaps(gt_sub_box, pred_sub_box)[0, 0]
                    obj_iou = bbox_overlaps(gt_obj_box, pred_obj_box)[0, 0]
                    if sub_iou < iou_thres or obj_iou < iou_thres:
                        continue
                    if pred_label in gt_pair['hit_predicates']:
                        continue
                    gt_pair['hit_predicates'].add(pred_label)
                    cat = self._get_predicate_category(pred_label)
                    if cat >= 0 and cat in gt_pair['category_weights']:
                        if cat not in gt_pair['hit_categories']:
                            gt_pair['category_hit_scores'][cat] += 1.0
                            gt_pair['hit_categories'].add(cat)
                        else:
                            gt_pair['category_hit_scores'][cat] += 0.5
                    else:
                        pred_name = self.ind_to_predicates[pred_label] if pred_label < len(self.ind_to_predicates) else f'unknown(idx={pred_label})'
                        raise ValueError(f"[NGCategoryWeightedMeanRecall] Predicate '{pred_name}' (index={pred_label}) is not assigned to any semantic category. Check that PREDICATE_CATEGORIES covers every predicate.")
            category_total_weight = [0.0] * self.num_categories
            category_total_hit = [0.0] * self.num_categories
            for gt_pair in aggregated_gt:
                for cat, weight_c in gt_pair['category_weights'].items():
                    category_total_weight[cat] += weight_c
                    hit_c = gt_pair['category_hit_scores'].get(cat, 0.0)
                    category_total_hit[cat] += min(hit_c, weight_c)
            for cat_idx in range(self.num_categories):
                if category_total_weight[cat_idx] > 0:
                    recall_c = category_total_hit[cat_idx] / category_total_weight[cat_idx]
                    self.result_dict[mode + '_ng_cwmr_collect'][k][cat_idx].append(recall_c)

    def calculate_mean_recall(self, mode):
        for k in self.result_dict[mode + '_ng_category_weighted_mean_recall']:
            sum_recall = 0.0
            num_valid_categories = 0
            for cat_idx in range(self.num_categories):
                cat_recalls = self.result_dict[mode + '_ng_cwmr_collect'][k][cat_idx]
                if len(cat_recalls) > 0:
                    cat_mean_recall = np.mean(cat_recalls)
                    num_valid_categories += 1
                else:
                    cat_mean_recall = 0.0
                self.result_dict[mode + '_ng_cwmr_list'][k].append(cat_mean_recall)
                sum_recall += cat_mean_recall
            if num_valid_categories > 0:
                self.result_dict[mode + '_ng_category_weighted_mean_recall'][k] = sum_recall / float(num_valid_categories)
            else:
                self.result_dict[mode + '_ng_category_weighted_mean_recall'][k] = 0.0
        return

def _triplet(relations, classes, boxes, predicate_scores=None, class_scores=None):
    sub_id, ob_id, pred_label = (relations[:, 0], relations[:, 1], relations[:, 2])
    triplets = np.column_stack((classes[sub_id], pred_label, classes[ob_id]))
    triplet_boxes = np.column_stack((boxes[sub_id], boxes[ob_id]))
    triplet_scores = None
    if predicate_scores is not None and class_scores is not None:
        triplet_scores = np.column_stack((class_scores[sub_id], predicate_scores, class_scores[ob_id]))
    return (triplets, triplet_boxes, triplet_scores)

def _compute_pred_matches(gt_triplets, pred_triplets, gt_boxes, pred_boxes, iou_thres, phrdet=False):
    keeps = intersect_2d(gt_triplets, pred_triplets)
    gt_has_match = keeps.any(1)
    pred_to_gt = [[] for x in range(pred_boxes.shape[0])]
    for gt_ind, gt_box, keep_inds in zip(np.where(gt_has_match)[0], gt_boxes[gt_has_match], keeps[gt_has_match]):
        boxes = pred_boxes[keep_inds]
        if phrdet:
            gt_box_union = gt_box.reshape((2, 4))
            gt_box_union = np.concatenate((gt_box_union.min(0)[:2], gt_box_union.max(0)[2:]), 0)
            box_union = boxes.reshape((-1, 2, 4))
            box_union = np.concatenate((box_union.min(1)[:, :2], box_union.max(1)[:, 2:]), 1)
            inds = bbox_overlaps(gt_box_union[None], box_union)[0] >= iou_thres
        else:
            sub_iou = bbox_overlaps(gt_box[None, :4], boxes[:, :4])[0]
            obj_iou = bbox_overlaps(gt_box[None, 4:], boxes[:, 4:])[0]
            inds = (sub_iou >= iou_thres) & (obj_iou >= iou_thres)
        for i in np.where(keep_inds)[0][inds]:
            pred_to_gt[i].append(int(gt_ind))
    return pred_to_gt
