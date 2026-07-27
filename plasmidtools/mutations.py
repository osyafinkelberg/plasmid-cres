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


def mutation_string(mutations: list[dict], position_shift: int = 0) -> str:
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
