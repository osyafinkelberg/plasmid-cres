_COMPLEMENT = str.maketrans("ATCGatcg", "TAGCtagc")


def substitution(wt_seq: str, pos: int, fro: str, to: str) -> str:
    assert wt_seq[pos] == fro, f"Expected '{fro}' at index {pos}, found '{wt_seq[pos]}'"
    return "".join([wt_seq[:pos], to, wt_seq[pos + 1:]])


def insertion(wt_seq: str, pos: int, nuc: str) -> str:
    # Places the new nucleotide after `pos`, shifting the existing base at `pos` + 1 to the right
    return "".join([wt_seq[:pos + 1], nuc, wt_seq[pos + 1:]])


def deletion(wt_seq: str, pos: int, nuc: str) -> str:
    assert wt_seq[pos] == nuc, f"Expected '{nuc}' at index {pos}, found '{wt_seq[pos]}'"
    return "".join([wt_seq[:pos], wt_seq[pos + 1:]])


def apply_mutations(wt_seq: str, mutations: list[dict]) -> str:
    """
    Applies a list of mutations to the wild-type sequence.

    Expects a list of dictionaries where all positions are 0-based:
    [
        {'type': 'substitution', 'position': 576, 'from': 'G', 'to': 'C'},
        {'type': 'insertion', 'position': 574, 'from': '', 'to': 'T'}
    ]
    """
    # Sort mutations by position in descending order (right-to-left)
    # This keeps upstream 0-based indices perfectly valid as we progress!
    sorted_mutations = sorted(mutations, key=lambda m: int(m['position']), reverse=True)

    mut_seq = wt_seq
    for mut in sorted_mutations:
        m_type = mut['type']
        pos = int(mut['position'])

        if m_type == 'substitution':
            mut_seq = substitution(mut_seq, pos, mut['from'], mut['to'])
        elif m_type == 'insertion':
            mut_seq = insertion(mut_seq, pos, mut['to'])
        elif m_type == 'deletion':
            mut_seq = deletion(mut_seq, pos, mut['from'])
        else:
            raise ValueError(f"Unknown mutation type: {m_type}")

    return mut_seq


def mutation_to_string(mutations: list[dict], position_shift: int = 0) -> str:
    """
    Accepts a list of mutation dicts and returns a standardized,
    comma-separated string representation sorted by position.
    """
    # Sort mutations by position in ascending order
    sorted_mutations = sorted(mutations, key=lambda m: int(m['position']))

    mutation_parts = []
    for mut in sorted_mutations:
        m_type = mut['type']
        pos = int(mut['position']) - position_shift

        if m_type == 'substitution':
            # Format: G194C
            mutation_parts.append(f"{mut['from']}{pos}{mut['to']}")
        elif m_type == 'insertion':
            # Format: INS5 28A
            mutation_parts.append(f"INS {pos}{mut['to']}")
        elif m_type == 'deletion':
            # Format: DEL 550T
            mutation_parts.append(f"DEL {pos}{mut['from']}")
        else:
            raise ValueError(f"Unknown mutation type: {m_type}")

    return ", ".join(mutation_parts)


def string_to_mutation(mutation_string: str, position_shift: int = 0) -> list[dict]:
    """
    Parses a comma-separated mutation string (as produced by mutation_to_string)
    back into a list of mutation dicts.

    The position_shift is added to every string position to recover the original
    dict positions: if mutation_to_string was called with position_shift=S, pass
    the same S here to obtain the original 0-based positions.

    Supported token formats (matching mutation_to_string output):
        Substitution : {from_nuc}{pos}{to_nuc}   e.g. "G576C"
        Insertion    : INS {pos}{nuc}             e.g. "INS 574T"
        Deletion     : DEL {pos}{nuc}             e.g. "DEL 576G"

    Returns an empty list for the string "WT" or an empty/blank string.
    """
    if not mutation_string or mutation_string.strip() == "WT":
        return []

    mutations = []
    for part in mutation_string.split(", "):
        part = part.strip()
        if not part:
            continue

        if part.startswith("INS "):
            rest = part[4:].strip()
            split_at = next((i for i, c in enumerate(rest) if not c.isdigit()), len(rest))
            pos = int(rest[:split_at]) + position_shift
            nuc = rest[split_at:]
            mutations.append({'type': 'insertion', 'position': pos, 'from': '', 'to': nuc})

        elif part.startswith("DEL "):
            rest = part[4:].strip()
            split_at = next((i for i, c in enumerate(rest) if not c.isdigit()), len(rest))
            pos = int(rest[:split_at]) + position_shift
            nuc = rest[split_at:]
            mutations.append({'type': 'deletion', 'position': pos, 'from': nuc, 'to': ''})

        else:
            # Substitution: {from_nuc}{pos}{to_nuc}
            from_nuc = part[0]
            to_nuc   = part[-1]
            pos = int(part[1:-1]) + position_shift
            mutations.append({'type': 'substitution', 'position': pos, 'from': from_nuc, 'to': to_nuc})

    return mutations


def mutations_to_rc(mutations: list[dict], seq_len: int) -> list[dict]:
    """
    Converts a list of forward-strand mutation dicts to reverse-complement strand
    mutation dicts.

    Args:
        mutations : dicts with 0-based positions in the **forward WT** sequence.
        seq_len   : length of the WT (unmutated) forward-strand sequence.

    Returns:
        Mutation dicts with 0-based positions in the **RC WT** sequence.

    Transformation rules  (p = 0-based forward position, L = seq_len):
        Substitution  at p (X → Y)   →  RC sub at L-1-p (compl(X) → compl(Y))
        Insertion of N after p       →  RC ins of compl(N) after L-2-p
        Deletion  of N at p          →  RC del of compl(N) at L-1-p

    Derivation for insertion:
        insertion(fwd, p, N)  places N at index p+1 in the mutated fwd sequence.
        Its RC complement appears at index (L+1-1) - (p+1) = L-1-p in the mutated
        RC sequence, which equals index p_rc+1 where p_rc = L-2-p.  Therefore the
        equivalent RC operation is insertion(rc_wt, L-2-p, compl(N)).
    """
    rc_mutations = []
    for mut in mutations:
        m_type = mut['type']
        p = int(mut['position'])

        if m_type == 'substitution':
            rc_mutations.append({
                'type': 'substitution',
                'position': seq_len - 1 - p,
                'from': mut['from'].translate(_COMPLEMENT),
                'to':   mut['to'].translate(_COMPLEMENT),
            })
        elif m_type == 'insertion':
            rc_mutations.append({
                'type': 'insertion',
                'position': seq_len - 2 - p,
                'from': '',
                'to': mut['to'].translate(_COMPLEMENT),
            })
        elif m_type == 'deletion':
            rc_mutations.append({
                'type': 'deletion',
                'position': seq_len - 1 - p,
                'from': mut['from'].translate(_COMPLEMENT),
                'to':   '',
            })
        else:
            raise ValueError(f"Unknown mutation type: {m_type}")

    return rc_mutations
