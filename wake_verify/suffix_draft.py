"""Zero-parameter suffix n-gram lookup proposer with MARS placeholder fallback.

Used in Pillar 3 to demonstrate clean orthogonality between draft proposal quality
(numerator `1 + E[c]`) and Late-Entry + Wake Verification pass reduction (denominator `K / T`).
"""


def suffix_propose(known: list, B: int, fallback_props: list = None, max_n: int = 4, min_n: int = 2, stop_ids: set = None) -> list:
    """Search `known[:-1]` right-to-left for the most recent exact suffix match of length `max_n .. min_n`.

    If found, return up to `B` continuation tokens (stopping before `stop_ids` if given),
    topping up any remaining `< B` slots from `fallback_props` when available.
    """
    if B <= 0:
        return []
    fb = list(fallback_props) if fallback_props else []
    L = len(known)
    if L <= min_n:
        return fb[:B]

    # Limit search window to last 1024 tokens for O(1) bounded CPU cost (< 5 us)
    lo_search = max(0, L - 1024)
    for n in range(min(max_n, L - 1), min_n - 1, -1):
        pat = known[L - n :]
        last_tok = pat[-1]
        # Scan backwards from L - n - 1 down to lo_search
        for idx in range(L - n - 1, lo_search - 1, -1):
            if known[idx + n - 1] == last_tok and known[idx : idx + n] == pat:
                out = []
                p = idx + n
                while p < L and len(out) < B:
                    tk = known[p]
                    out.append(tk)
                    if stop_ids and tk in stop_ids:
                        break
                    p += 1
                if out:
                    if len(out) < B and len(fb) > len(out):
                        # Only pad from fallback if no stop token terminated the continuation
                        if not (stop_ids and out[-1] in stop_ids):
                            out.extend(fb[len(out) : B])
                    return out[:B]
    return fb[:B]
