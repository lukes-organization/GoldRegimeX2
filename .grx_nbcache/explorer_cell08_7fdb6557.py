# -----------------------------
# CPCV splitter with Purge + Embargo
# -----------------------------

@dataclass
class CPCVPurgedEmbargo:
    n_blocks: int = 6
    k_val_blocks: int = 2
    embargo_bars: int = 96

    def split(self, n: int, event_end_pos: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        all_idx = np.arange(n, dtype=np.int32)
        blocks = [np.array(x, dtype=np.int32) for x in np.array_split(all_idx, self.n_blocks)]
        block_ranges = [(b[0], b[-1]) for b in blocks if len(b) > 0]

        block_ids = list(range(len(blocks)))
        for combo in itertools.combinations(block_ids, self.k_val_blocks):
            val_blocks = [blocks[c] for c in combo]
            val_idx = np.sort(np.concatenate(val_blocks))

            val_ranges = [block_ranges[c] for c in combo]
            train_mask = np.ones(n, dtype=bool)
            train_mask[val_idx] = False
            train_idx = np.where(train_mask)[0]

            # Purge overlap with label windows
            keep = np.ones(len(train_idx), dtype=bool)
            for k, i in enumerate(train_idx):
                e_i = int(event_end_pos[i])
                for v_start, v_end in val_ranges:
                    if (i <= v_end) and (e_i >= v_start):
                        keep[k] = False
                        break
            train_idx = train_idx[keep]

            # Embargo after each validation range
            embargo_mask = np.zeros(n, dtype=bool)
            for _, v_end in val_ranges:
                e_s = v_end + 1
                e_e = min(n - 1, v_end + self.embargo_bars)
                if e_s <= e_e:
                    embargo_mask[e_s:e_e + 1] = True
            train_idx = train_idx[~embargo_mask[train_idx]]

            if len(train_idx) == 0 or len(val_idx) == 0:
                continue

            yield np.sort(train_idx), val_idx