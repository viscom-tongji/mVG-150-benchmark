import torch
from yacs.config import CfgNode as CN


DEFAULT_PREDICATE_CATEGORIES = {
    "Spatial-Positional": [
        "above", "across", "around", "between", "in", "on", "under", "behind", "in front of", "near", "near by",
        "to left of", "to right of", "along", "along back of", "along bottom of", "along edge of", "along side of", "along top of",
        "on top of", "on the bottom of", "on the back of", "on the front of", "on the right side of", "on the left side of",
        "on the edge of", "on the surface of", "in the left side of", "in the right side of", "in the middle of",
        "near the bottom of", "near the edge of", "near the front of", "near the side of", "near the top of", "surrounded by"
    ],
    "Action-Interaction": [
        "biting", "brushing", "buying", "carrying", "catching", "chasing", "chewing", "cleaning", "climbing", "cooking",
        "cutting", "decorating", "drinking", "eating", "feeding", "flying over", "flying in", "following", "guiding",
        "helping", "herding", "hitting", "holding", "hugging", "jumping from", "jumping over", "kicking", "kissing", "leaning on",
        "leaving", "licking", "opening", "picking", "pulling", "pushing", "reading", "slicing", "swinging", "throwing", "washing",
        "serving", "playing", "playing at", "playing in", "playing in front of", "playing on", "playing with", "playing near",
        "talking to", "says", "running on", "sitting on", "eating at", "eating with", "eating from", "walking in", "walking in front of",
        "walking on", "touching", "watching", "driving", "driving on", "riding", "entering", "exiting", "coming from", "floating in",
        "parked in", "parked on", "parked on side of", "parked on top of", "falling off", "crossing", "looking at", "facing"
    ],
    "Attachment-Containment": [
        "attached to", "attached to back of", "attached to front of", "attached to side of", "mounted on", "mounted on top of",
        "hanging on", "hanging over", "hanging from", "hanging in", "covering", "covered with", "filled with", "containing",
        "connected to", "growing in", "growing on", "growing on edge of", "growing on side of", "growing on top of",
        "painted on", "painted on side of", "painted on top of", "printed on", "decorated with", "show", "written on",
        "reflected in"
    ],
    "Ownership-Usage-Possession": [
        "has", "held by", "used by", "part of", "using", "wearing", "worn by", "made of", "belonging to"
    ],
    "Static-State": [
        "lying on", "resting on", "supporting", "standing on", "standing behind", "standing on edge of", "standing near"
    ],
}


def normalize_predicate_name(name):
    return str(name).strip().lower()


def _cfg_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _cfg_mapping(value):
    if value is None:
        return {}
    if isinstance(value, CN):
        return {k: value[k] for k in value.keys()}
    if isinstance(value, dict):
        return dict(value)
    return {}


def _default_hierarchy(rel_classes):
    normalized_default = {}
    for coarse_name, fine_names in DEFAULT_PREDICATE_CATEGORIES.items():
        for fine_name in fine_names:
            normalized_default[normalize_predicate_name(fine_name)] = coarse_name

    coarse_predicates = ["__background__"] + list(DEFAULT_PREDICATE_CATEGORIES.keys()) + ["Uncategorized"]
    coarse2id = {name: idx for idx, name in enumerate(coarse_predicates)}
    fine_to_coarse_id = []

    for fine_idx, fine_name in enumerate(rel_classes):
        fine_name_norm = normalize_predicate_name(fine_name)
        if fine_idx == 0 or fine_name_norm == "__background__":
            fine_to_coarse_id.append(coarse2id["__background__"])
        else:
            coarse_name = normalized_default.get(fine_name_norm, "Uncategorized")
            fine_to_coarse_id.append(coarse2id[coarse_name])

    return coarse_predicates, list(rel_classes), fine_to_coarse_id


def build_hierarchy_metadata(rel_classes, hierarchy_cfg):
    coarse_cfg = _cfg_list(getattr(hierarchy_cfg, "COARSE_PREDICATES", []))
    fine_cfg = _cfg_list(getattr(hierarchy_cfg, "FINE_PREDICATES", []))
    fine_to_coarse_cfg = _cfg_mapping(getattr(hierarchy_cfg, "FINE_TO_COARSE", None))

    if not coarse_cfg or not fine_to_coarse_cfg:
        coarse_predicates, fine_predicates, fine_to_coarse_id = _default_hierarchy(rel_classes)
    else:
        fine_predicates = list(rel_classes)
        normalized_cfg = {normalize_predicate_name(k): str(v) for k, v in fine_to_coarse_cfg.items()}
        coarse_predicates = ["__background__"] + [name for name in coarse_cfg if name != "__background__"]
        if "Uncategorized" not in coarse_predicates:
            coarse_predicates.append("Uncategorized")
        coarse2id = {name: idx for idx, name in enumerate(coarse_predicates)}
        fine_to_coarse_id = []

        fine_source = fine_cfg if fine_cfg else fine_predicates[1:]
        fine_source_lookup = {normalize_predicate_name(name): name for name in fine_source}

        for fine_idx, fine_name in enumerate(fine_predicates):
            fine_name_norm = normalize_predicate_name(fine_name)
            if fine_idx == 0 or fine_name_norm == "__background__":
                fine_to_coarse_id.append(coarse2id["__background__"])
                continue

            fine_lookup_name = fine_source_lookup.get(fine_name_norm, fine_name)
            coarse_name = normalized_cfg.get(normalize_predicate_name(fine_lookup_name), "Uncategorized")
            if coarse_name not in coarse2id:
                coarse2id[coarse_name] = len(coarse_predicates)
                coarse_predicates.append(coarse_name)
            fine_to_coarse_id.append(coarse2id[coarse_name])

    coarse_to_fine = torch.zeros(len(coarse_predicates), len(rel_classes), dtype=torch.float32)
    for fine_idx, coarse_idx in enumerate(fine_to_coarse_id):
        coarse_to_fine[coarse_idx, fine_idx] = 1.0

    return {
        "coarse_predicates": coarse_predicates,
        "fine_predicates": list(rel_classes),
        "fine_to_coarse_id": fine_to_coarse_id,
        "coarse_to_fine_mask": coarse_to_fine,
    }


def aggregate_fine_probs_to_coarse(fine_probs, coarse_to_fine_mask):
    return fine_probs @ coarse_to_fine_mask.t()


def build_expert_metadata(rel_classes, hierarchy_cfg, expert_cfg=None):
    hierarchy_meta = build_hierarchy_metadata(rel_classes, hierarchy_cfg)
    coarse_predicates = hierarchy_meta["coarse_predicates"]
    fine_to_coarse_id = hierarchy_meta["fine_to_coarse_id"]

    num_coarse = len(coarse_predicates)
    num_rel = len(rel_classes)
    expert_owner_masks = torch.zeros(num_coarse, num_rel, dtype=torch.float32)
    for fine_idx, coarse_idx in enumerate(fine_to_coarse_id):
        expert_owner_masks[coarse_idx, fine_idx] = 1.0

    responsibility_in = 1.0
    responsibility_out = 0.2
    if expert_cfg is not None:
        responsibility_in = float(getattr(expert_cfg, "RESPONSIBILITY_IN_WEIGHT", 1.0))
        responsibility_out = float(getattr(expert_cfg, "RESPONSIBILITY_OUT_WEIGHT", 0.2))

    expert_responsibility_prior = torch.full(
        (num_coarse, num_rel), responsibility_out, dtype=torch.float32
    )
    expert_responsibility_prior[expert_owner_masks > 0] = responsibility_in

    return {
        "expert_names": list(coarse_predicates),
        "expert_owner_masks": expert_owner_masks,
        "expert_responsibility_prior": expert_responsibility_prior,
    }


def build_lite_expert_metadata(rel_classes, hierarchy_cfg):
    hierarchy_meta = build_hierarchy_metadata(rel_classes, hierarchy_cfg)
    coarse_predicates = hierarchy_meta["coarse_predicates"]
    fine_to_coarse_id = hierarchy_meta["fine_to_coarse_id"]

    num_coarse = len(coarse_predicates)
    num_rel = len(rel_classes)
    coarse_to_expert_idx = torch.full((num_coarse,), -1, dtype=torch.long)

    expert_names = []
    expert_owner_masks = []
    expert_class_indices = []

    for coarse_idx, coarse_name in enumerate(coarse_predicates):
        if coarse_name in ("__background__", "Uncategorized"):
            continue

        class_indices = [
            fine_idx
            for fine_idx, mapped_idx in enumerate(fine_to_coarse_id)
            if mapped_idx == coarse_idx and fine_idx != 0
        ]
        if not class_indices:
            continue

        expert_idx = len(expert_names)
        coarse_to_expert_idx[coarse_idx] = expert_idx

        owner_mask = torch.zeros(num_rel, dtype=torch.float32)
        owner_mask[class_indices] = 1.0

        expert_names.append(coarse_name)
        expert_owner_masks.append(owner_mask)
        expert_class_indices.append(class_indices)

    if expert_owner_masks:
        expert_owner_masks = torch.stack(expert_owner_masks, dim=0)
    else:
        expert_owner_masks = torch.zeros(0, num_rel, dtype=torch.float32)

    return {
        "expert_names": expert_names,
        "expert_owner_masks": expert_owner_masks,
        "expert_class_indices": expert_class_indices,
        "coarse_to_expert_idx": coarse_to_expert_idx,
    }


def build_overlap_metadata(rel_classes, expert_cfg=None):
    overlap_names = []
    if expert_cfg is not None:
        spatial_cfg = getattr(expert_cfg, "SPATIAL_OVERLAP", None)
        if spatial_cfg is not None:
            overlap_names = _cfg_list(getattr(spatial_cfg, "OVERLAP_PREDICATES", []))

    rel_lookup = {normalize_predicate_name(name): idx for idx, name in enumerate(rel_classes)}
    overlap_mask = torch.zeros(len(rel_classes), dtype=torch.float32)
    overlap_indices = []

    for overlap_name in overlap_names:
        rel_idx = rel_lookup.get(normalize_predicate_name(overlap_name))
        if rel_idx is None or rel_idx in overlap_indices:
            continue
        overlap_mask[rel_idx] = 1.0
        overlap_indices.append(rel_idx)

    return {
        "overlap_mask": overlap_mask,
        "overlap_indices": overlap_indices,
        "overlap_predicates": [rel_classes[idx] for idx in overlap_indices],
    }
