import json
import os
import re
import tempfile
from math import sqrt
from typing import List

try:
    import fcntl
except Exception:
    fcntl = None


def _extract_text_from_completions(completions):
    texts = []
    for completion in completions:
        if isinstance(completion, list):
            if len(completion) > 0 and isinstance(completion[0], dict) and "content" in completion[0]:
                texts.append(str(completion[0]["content"]))
            else:
                texts.append(str(completion[0]) if len(completion) > 0 else "")
        else:
            texts.append(str(completion))
    return texts


def _broadcast_to_len(x, n: int):
    if x is None:
        return [None] * n
    if isinstance(x, list) and len(x) == n:
        return x
    if isinstance(x, list) and len(x) > 0 and (n % len(x) == 0):
        k = n // len(x)
        out = []
        for v in x:
            out.extend([v] * k)
        return out
    return [x] * n


def _safe_key_cmp(a, b):
    if a is None or b is None:
        return False
    try:
        return abs(float(a[0]) - float(b[0])) < 1e-3 and abs(float(a[1]) - float(b[1])) < 1e-3
    except Exception:
        return False


def update_status_in_snapshot(snapshot_path, id_val, click_val, incoming_status):
    if not snapshot_path or not os.path.exists(snapshot_path):
        return False
    if id_val is None or click_val is None:
        return False

    try:
        with open(snapshot_path, 'r', encoding='utf-8') as fr:
            lines = [l for l in fr.readlines() if l.strip()]

        entries = []
        for ln in lines:
            try:
                obj = json.loads(ln)
                entries.append(obj)
            except Exception:
                continue

        updated = False
        for idx, obj in enumerate(entries):
            if not isinstance(obj, dict):
                continue
            if obj.get('id') == id_val and 'click' in obj and _safe_key_cmp(obj.get('click'), click_val):
                old_status = obj.get('status', None)
                if old_status is True:
                    return True
                if incoming_status is True:
                    obj['status'] = True
                    updated = True
                elif old_status is None:
                    obj['status'] = False
                    updated = True
                entries[idx] = obj
                break

        if not updated:
            return False

        tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(snapshot_path))
        try:
            with os.fdopen(tmp_fd, 'w', encoding='utf-8') as fw:
                for obj in entries:
                    fw.write(json.dumps(obj, ensure_ascii=False) + "\n")

            if fcntl is not None:
                with open(snapshot_path, 'r') as flockf:
                    fcntl.flock(flockf, fcntl.LOCK_EX)
                    os.replace(tmp_path, snapshot_path)
                    fcntl.flock(flockf, fcntl.LOCK_UN)
            else:
                os.replace(tmp_path, snapshot_path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
        return True
    except Exception as e:
        print(f"Error in update_status_in_snapshot: {e}")
        return False


def hybrid_label_confidence_reward(
    completions,
    gen_type: List[int],
    image_height,
    image_width,
    correct_box,
    dark_box,
    click,
    recorded_samples_path,
    snapshot_path,
    **kwargs,
):
    p_min = 0.1
    p_max = 10.0
    eps = 1e-6

    completion_texts = _extract_text_from_completions(completions)
    n = len(completion_texts)
    print(f"[RWD][ENTER] n={n} gen_type={gen_type} snapshot_path={snapshot_path} ids_kw={kwargs.get('id')}")

    types_list = _broadcast_to_len(gen_type, n)
    ids = kwargs.get("id")
    ids_list = _broadcast_to_len(ids, n)
    clicks_list = _broadcast_to_len(click, n)

    def _normalize_click_item(c):
        if c is None:
            return None
        if isinstance(c, (list, tuple)) and len(c) > 0 and isinstance(c[0], (list, tuple)):
            a = c[0]
            return [float(a[0]), float(a[1])]
        if isinstance(c, (list, tuple)) and len(c) >= 2:
            return [float(c[0]), float(c[1])]
        if isinstance(c, dict):
            if 'x' in c and 'y' in c:
                return [float(c['x']), float(c['y'])]
            if 0 in c and 1 in c:
                return [float(c[0]), float(c[1])]
        return None

    rewards = []
    recorded_samples = []

    def _clip(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    def _area_s0(s_img: float, s_b: float, s_d: float) -> float:
        return float(s_img) - float(s_b) - float(s_d)

    def _s_for_label(label: int, s_img: float, s_b: float, s_d: float) -> float:
        s0 = _area_s0(s_img, s_b, s_d)
        if label == 1:
            return float(s_b)
        if label == -1:
            return float(s_d)
        return float(s0)

    def _attention_scalar(pred_label: int, s_img: float, s_b: float, s_d: float) -> float:
        if s_img is None or s_b is None or s_d is None:
            return 0.0
        s_img = float(s_img)
        if s_img <= 0:
            return 0.0
        s0 = _area_s0(s_img, s_b, s_d)
        if pred_label == 1:
            s_hat = float(s_b)
        elif pred_label == -1:
            s_hat = float(s_d)
        else:
            s_hat = float(s0)
        return _clip(s_hat / s_img, 0.0, 1.0)

    def _severity_weight(label: int, pred_label: int, p_a: float) -> float:
        if label == -1 and pred_label == 1:
            return 10.0
        if label == 0 and pred_label in (1, -1):
            return 1.0 + float(p_a)
        return 1.0

    def _confidence_adjustment(label: int, pred_label: int, s_img, s_b, s_d, d_l, d_h) -> float:
        if (s_img is None) or (s_b is None) or (s_d is None) or (d_l is None) or (d_h is None):
            return 1.0
        s_img = float(s_img)
        if s_img <= 0:
            return 1.0
        s_l_val = sqrt(_s_for_label(label, s_img, s_b, s_d))
        raw = (1 / (float(d_h) + eps)) / (s_l_val / s_img)
        return _clip(raw, p_min, p_max)

    def _point_to_box_distance(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
        closest_x = max(x1, min(px, x2))
        closest_y = max(y1, min(py, y2))
        return ((px - closest_x) ** 2 + (py - closest_y) ** 2) ** 0.5

    gt_x1, gt_y1, gt_x2, gt_y2 = correct_box[0]["bbox"]
    dc_x1, dc_y1, dc_x2, dc_y2 = dark_box[0]["bbox"]
    x = click[0][0]
    y = click[0][1]

    s_img = image_height[0] * image_width[0]
    s_b = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
    s_d = (dc_x2 - dc_x1) * (dc_y2 - dc_y1)
    d_l = 0

    for i, (text, label) in enumerate(zip(completion_texts, types_list)):
        judge_match = re.search(r"<judge>\s*(-?1|0)\s*</judge>", text, re.DOTALL | re.IGNORECASE)
        conf_match = re.search(r"<conf>\s*([0-9]*\.?[0-9]+)\s*</conf>", text, re.DOTALL | re.IGNORECASE)

        if not judge_match or not conf_match:
            reward = -10.0
            recorded_samples.append({"text": text, "reason": "parse_fail"})
            rewards.append(float(reward))
            print(f"Hybrid Reward: {reward:.3f}, Recorded: {len(recorded_samples)} samples")
            continue

        pred_label = int(judge_match.group(1))
        conf = float(conf_match.group(1))
        conf = max(0.0, min(1.0, conf))

        if (label not in (-1, 0, 1)) or (pred_label not in (-1, 0, 1)):
            reward = -1.0
            recorded_samples.append({"text": text, "reason": "label_out_of_domain", "L": label, "hatL": pred_label, "conf": conf})
            rewards.append(float(reward))
            print(f"Hybrid Reward: {reward:.3f}, Recorded: {len(recorded_samples)} samples")
            continue

        if pred_label == label:
            reward = conf
        else:
            if pred_label == 1:
                d_h = _point_to_box_distance(x, y, gt_x1, gt_y1, gt_x2, gt_y2)
            elif pred_label == -1:
                d_h = _point_to_box_distance(x, y, dc_x1, dc_y1, dc_x2, dc_y2)
            else:
                d_h = min(
                    _point_to_box_distance(x, y, gt_x1, gt_y1, gt_x2, gt_y2),
                    _point_to_box_distance(x, y, dc_x1, dc_y1, dc_x2, dc_y2),
                )
            p_a = _attention_scalar(pred_label, s_img, s_b, s_d)
            w_s = _severity_weight(label, pred_label, p_a)
            p_c = _confidence_adjustment(label, pred_label, s_img, s_b, s_d, d_l, d_h)
            reward = p_c * w_s * (-conf)

            recorded_samples.append({
                "text": text,
                "reason": "label_mismatch",
                "L": label,
                "hatL": pred_label,
                "conf": conf,
                "p_c": p_c,
                "w_s": w_s,
                "p_a": p_a,
            })

        rewards.append(float(reward))
        print(f"Hybrid Reward: {reward:.3f}, Recorded: {len(recorded_samples)} samples")

        try:
            if snapshot_path:
                id_val = ids_list[i] if isinstance(ids_list, list) and len(ids_list) > i else None
                click_val = _normalize_click_item(clicks_list[i]) if isinstance(clicks_list, list) and len(clicks_list) > i else None
                incoming_status = True if pred_label == label else False
                update_status_in_snapshot(snapshot_path, id_val, click_val, incoming_status)
        except Exception as e:
            print(f"Error updating snapshot: {e}")

    if recorded_samples:
        print("Recorded mismatched samples:")
        for sample in recorded_samples:
            try:
                print(
                    f"  Reason: {sample['reason']}, L: {sample.get('L')}, hatL: {sample.get('hatL')}, "
                    f"Conf: {sample.get('conf')}, p_c: {sample.get('p_c')}, w_s: {sample.get('w_s')}, p_a: {sample.get('p_a')}, "
                    f"Text: {sample['text'][:100]}..."
                )
            except Exception:
                print(f"  Reason: {sample.get('reason')}, Text: {sample.get('text', '')[:100]}...")

        if recorded_samples_path:
            try:
                import time

                ts = time.time()
                pid = os.getpid()
                _rank = None
                try:
                    import torch
                    if hasattr(torch, 'distributed') and torch.distributed.is_available() and torch.distributed.is_initialized():
                        _rank = int(torch.distributed.get_rank())
                except Exception:
                    _rank = None

                run_ts = kwargs.get("run_ts") if isinstance(kwargs, dict) else None

                dirpath = os.path.dirname(recorded_samples_path)
                if dirpath:
                    os.makedirs(dirpath, exist_ok=True)
                with open(recorded_samples_path, "a", encoding="utf-8") as f:
                    for sample in recorded_samples:
                        sample_meta = dict(sample)
                        sample_meta["_ts_write"] = ts
                        sample_meta["_pid"] = pid
                        sample_meta["_rank"] = _rank
                        sample_meta["_run_ts"] = run_ts
                        f.write(json.dumps(sample_meta, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"Warning: failed to write recorded_samples to {recorded_samples_path}: {e}")

    return rewards


def label_confidence_reward(completions, type: List[int], **kwargs):
    completion_texts = _extract_text_from_completions(completions)
    n = len(completion_texts)
    types_list = _broadcast_to_len(type, n)

    rewards = []
    recorded_samples = []

    for text, expected_judge in zip(completion_texts, types_list):
        judge_match = re.search(r"<judge>\s*(-?1|0)\s*</judge>", text, re.DOTALL | re.IGNORECASE)
        conf_match = re.search(r"<conf>\s*([0-9]*\.?[0-9]+)\s*</conf>", text, re.DOTALL | re.IGNORECASE)

        if not judge_match or not conf_match:
            reward = -1.0
            recorded_samples.append({"text": text, "reason": "parse_fail"})
        else:
            judge = int(judge_match.group(1))
            conf = float(conf_match.group(1))
            conf = max(0.0, min(1.0, conf))

            if judge == expected_judge:
                reward = conf
            else:
                reward = -conf
                recorded_samples.append({"text": text, "reason": "judge_mismatch", "expected": expected_judge, "actual": judge, "conf": conf})

        rewards.append(float(reward))
        print(f"Confidence Reward: {reward:.3f}, Recorded: {len(recorded_samples)} samples")

    if recorded_samples:
        print("Recorded mismatched samples:")
        for sample in recorded_samples:
            try:
                print(f"  Reason: {sample['reason']}, Expected: {sample['expected']}, Actual: {sample['actual']}, Conf: {sample['conf']}, Text: {sample['text'][:100]}...")
            except Exception:
                print(f"  Reason: {sample['reason']}, Text: {sample['text'][:100]}...")

    return rewards
