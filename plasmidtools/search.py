import bisect
from abc import ABC, abstractmethod
from pathlib import Path

import mappy as mp
import numpy as np
from pydivsufsort import divsufsort

# ─────────────────────────────────────────────────────────────────────────────
# Base class
# ─────────────────────────────────────────────────────────────────────────────


class PlasmidSearchBase(ABC):
    """
    Abstract base class for exact sequence matching engines over a
    circular plasmid database stored as doubled FASTA sequences.

    FASTA format contract
    ---------------------
    Header   : ">gbk_name|true_length"
    Sequence : the plasmid nucleotides concatenated with themselves
               (stored length = 2 × true_length).

    The duplication exposes origin-spanning (wrap-around) matches as
    ordinary substring hits, eliminating per-subclass boundary logic.
    A hit whose start position ≥ true_length is a duplicate of the
    corresponding hit in the first copy and must be discarded.

    Subclass contract
    -----------------
    Call ``super().__init__(fasta_db_path, _load_sequences=True)`` when the
    subclass needs the raw sequences in Python (BruteForce, AhoCorasick).
    Leave the flag False when the file is handed directly to a C/C++ library
    that performs its own I/O (Minimap).
    """

    def __init__(self, fasta_db_path: Path, _load_sequences: bool = False) -> None:
        self.db_path: str = str(fasta_db_path.resolve())

        # {gbk_name: true_length}  — always populated.
        self.plasmid_lengths: dict[str, int] = {}

        # {gbk_name: (doubled_seq_upper, true_length)}
        # Populated only when _load_sequences=True; subclasses that pass the
        # file path to an external aligner should leave this empty to avoid
        # loading ~500 MB of sequence data into the Python heap unnecessarily.
        self.seqs: dict[str, tuple[str, int]] = {}

        for name, seq, _ in mp.fastx_read(self.db_path):
            gbk_name, length_str = name.split("|")
            true_length = int(length_str)
            self.plasmid_lengths[gbk_name] = true_length
            if _load_sequences:
                self.seqs[gbk_name] = (seq.upper(), true_length)

        if not self.plasmid_lengths:
            raise RuntimeError(f"Failed to load sequences from {self.db_path}.")

    # ── Shared utility ────────────────────────────────────────────────────────

    @staticmethod
    def _reverse_complement(seq: str) -> str:
        """Return the reverse complement of a DNA sequence (A/C/G/T)."""
        table = str.maketrans("ACGTacgt", "TGCAtgca")
        return seq.translate(table)[::-1]

    # ── Interface ─────────────────────────────────────────────────────────────

    @abstractmethod
    def search_sequence(self, query: str) -> list[dict]:
        """
        Find all exact occurrences of ``query`` in the plasmid database.

        Returns a list of hit dicts, each containing:
            gbk_name    (str) – plasmid identifier
            position    (int) – 0-based start on the original circular sequence
            orientation (str) – "fwd" or "rev"

        Notes
        -----
        Subclasses differ in reverse-complement handling:
          • PlasmidSearchMinimap     – both strands, handled by minimap2.
          • PlasmidSearchAhoCorasick – both strands, handled internally.
        """


# ─────────────────────────────────────────────────────────────────────────────
# Concrete implementations
# ─────────────────────────────────────────────────────────────────────────────

class PlasmidSearchMinimap(PlasmidSearchBase):
    """
    A Python interface for arbitrary-length exact sequence matching 
    across a preprocessed circular plasmid database. Fast, approximate,
    may miss sequence occurrences due to filtering of frequent k-mers (Minimap2). 
    """

    def __init__(self, fasta_db_path: Path) -> None:
        # Sequences are read directly by the C++ aligner; no need to load
        # them into the Python heap.
        super().__init__(fasta_db_path, _load_sequences=False)

        self.index: mp.Aligner = mp.Aligner(fn_idx_in=self.db_path, preset="sr")

        # # NOTE: alternatively override 'sr' heuristics to catch ultra-short exact sequences
        # self.index: mp.Aligner = mp.Aligner(
        #     fn_idx_in=self.db_path,
        #     preset="ava-ont",       # Preserves mid_occ_frac=0.05 → max_occ ~100k → no k-mer dropping
        #     k=21,                   # Sr-like: larger k-mers are more specific, ~3x fewer seed pairs
        #     w=11,                   # Sr-like: coarser window vs ava-ont's w=5 (~2x fewer seeds)
        #     min_cnt=2,              # Require ≥2 chained seeds: eliminates single-seed noise chains
        #     min_chain_score=40,     # Prunes weak chains before DP; exact 200bp match scores ~192
        #     bw=20,                  # KEY PARAMETER: exact matches stay on the diagonal (0 gaps)
        #                             # Reduces DP from O(L²) → O(L × 20) per hit: 15-30x speedup
        #     best_n=100000,
        # )

        # # NOTE: alternatively override 'sr' heuristics to catch ultra-short exact sequences
        # self.index: mp.Aligner = mp.Aligner(
        #     fn_idx_in=self.db_path, 
        #     preset="sr",
        #     k=15,                # Shrink the k-mer seed size (default is 21)
        #     w=5,                 # Shrink the window to sample seeds more densely
        #     min_cnt=1,           # Allow a match even if only 1 seed is found
        #     min_chain_score=15   # Lower the required score threshold
        # )

        if not self.index:
            raise RuntimeError(f"C++ Index compilation failed for {self.db_path}.")

    def search_sequence(self, query_seq: str) -> list[dict[str, str | int]]:
        assert len(query_seq) >= 35
        query_seq = query_seq.upper()
        results: list[dict[str, str | int]] = []

        for hit in self.index.map(query_seq):
            if hit.mlen != len(query_seq) or hit.blen != len(query_seq):
                continue

            # Extract actual plasmid name and true length
            gbk_name, _ = hit.ctg.split("|")
            true_length = self.plasmid_lengths[gbk_name]

            # Hits whose start falls in the overlap part are exact
            # duplicates of hits already recorded from the first copy.
            position = int(hit.r_st)
            if position >= true_length:
                continue

            orientation: str = "fwd" if hit.strand == 1 else "rev"
            results.append({
                "gbk_name": gbk_name,
                "position": position,
                "orientation": orientation
            })

        return results


class PlasmidSearchSuffixArray(PlasmidSearchBase):
    """
    Exact-matching engine backed by a suffix array (SA) over the full
    concatenated plasmid database.

    Build once, query many times.  Unlike PlasmidSearchAhoCorasick, no
    prior knowledge of the query set is required; new queries arrive
    individually with no rebuild cost.

    Complexity
    ----------
    Build    : O(n log n) time, O(5n) working space  (divsufsort algorithm)
    Per query: O(m log n + |hits|) time               m = |query|, n = |database|

    When to prefer SA over Aho-Corasick
    ------------------------------------
    • Queries arrive one-by-one (interactive API, streaming pipeline).
    • The full query set is not known at index-build time.
    • For a known batch of ~100 k queries, AhoCorasick.batch_search is
      faster: one O(|database|) pass vs. 100 k independent O(m log n)
      searches.

    Memory footprint (55 k plasmids × 5 kbp median, doubled FASTA)
    --------------------------------------------------------------
    Concatenated text  : ~550 MB   (immutable bytes object)
    Suffix array       : ~2.2 GB   (int32 NumPy array, 4 bytes × 550 M entries)
    ──────────────────────────────────────────────────────────────
    Total at query time:  ~3   GB
    Peak during build  :  ~5   GB  (divsufsort needs ~3n working space)

    Circularity handling
    --------------------
    Identical to PlasmidSearchAhoCorasick: stored sequences are already
    doubled (stored length = 2 × true_length).  An SA hit whose start
    position is ≥ true_length within its plasmid is a duplicate of the
    corresponding first-copy hit and is silently discarded.
    """

    _SEP: bytes = b"\x00"   # NUL separator — ASCII 0 < A(65), so separator
                             # suffixes sort to the SA head and never pollute
                             # a nucleotide search range.

    def __init__(self, fasta_db_path: Path) -> None:
        super().__init__(fasta_db_path, _load_sequences=True)
        self._build_index()

    # ── Index construction ────────────────────────────────────────────────────

    def _build_index(self) -> None:
        """
        Concatenate all doubled sequences with NUL separators and build the SA.

        Text layout
        -----------
        [doubled_seq_1][NUL][doubled_seq_2][NUL] ... [doubled_seq_N][NUL]

        NUL is chosen as the separator for two reasons:
          1. Lexicographic safety: NUL (0x00) < A (0x41), so separator
             suffixes sort to the very front of the SA and are never
             reachable by a nucleotide binary search.
          2. Cross-plasmid isolation: any SA suffix that straddles a NUL
             byte contains a sub-ACGT character and cannot match a
             pure-nucleotide query, making inter-plasmid false matches
             structurally impossible.

        Data structures built here
        --------------------------
        self._text         : concatenated bytes object (~550 MB)
        self._plasmid_offsets : {gbk_name: (start_offset, true_length)}
        self._offset_table : list of (start_offset, gbk_name, true_length)
                             sorted by start_offset; fed to bisect for
                             O(log N_plasmids) position → plasmid lookup.
        self._offset_keys  : parallel list of start_offsets for bisect.
        self._sa           : int32 NumPy suffix array over self._text.
        """
        # ── 1. Concatenate sequences and record per-plasmid byte offsets ──────
        self._plasmid_offsets: dict[str, tuple[int, int]] = {}
        chunks: list[bytes] = []
        offset = 0

        for gbk_name, (seq, true_length) in self.seqs.items():
            seq_bytes: bytes = seq.encode()          # ASCII nucleotides → bytes
            self._plasmid_offsets[gbk_name] = (offset, true_length)
            chunks.append(seq_bytes)
            chunks.append(self._SEP)
            offset += len(seq_bytes) + 1             # +1 for the separator byte

        self._text: bytes = b"".join(chunks)

        # ── 2. Build sorted position-lookup structures ────────────────────────
        # Sorted by start_offset so bisect_right can locate the owning plasmid
        # in O(log 55 000) ≈ 16 comparisons for any absolute text position.
        self._offset_table: list[tuple[int, str, int]] = sorted(
            (off, name, tl)
            for name, (off, tl) in self._plasmid_offsets.items()
        )
        # Parallel list of start_offsets — bisect operates on this alone,
        # avoiding tuple comparison overhead on every call.
        self._offset_keys: list[int] = [row[0] for row in self._offset_table]

        # ── 3. Build suffix array ─────────────────────────────────────────────
        # divsufsort returns int32 for inputs < 2^31 bytes.
        # 550 MB << 2.1 GB, so int32 is always appropriate for this dataset.
        self._sa: np.ndarray = divsufsort(self._text)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _abs_to_plasmid(self, abs_pos: int) -> tuple[str | None, int | None]:
        """
        Map an absolute position in self._text to (gbk_name, local_position).

        Returns (None, None) for positions that must be discarded:
          • The NUL separator byte between two plasmids.
          • Any position inside the second copy of a doubled sequence
            (local ≥ true_length) — these are duplicates of first-copy hits.

        The owning plasmid is found with bisect_right in O(log N_plasmids).
        The subsequent arithmetic is O(1).
        """
        # bisect_right returns the insertion point to the right of any equal
        # key, so subtracting 1 gives the last plasmid whose start ≤ abs_pos.
        idx = bisect.bisect_right(self._offset_keys, abs_pos) - 1
        if idx < 0:
            return None, None                        # before the first plasmid

        start_off, gbk_name, true_length = self._offset_table[idx]
        local = abs_pos - start_off

        if local >= 2 * true_length:                 # NUL separator or beyond
            return None, None
        if local >= true_length:                     # second copy — discard
            return None, None

        return gbk_name, local                       # valid first-copy hit

    def _sa_range(self, pattern_bytes: bytes) -> tuple[int, int]:
        """
        Binary-search the SA for the half-open interval [lo, hi) such that
        every entry SA[lo:hi] points to a suffix that begins with pattern_bytes.

        Two independent binary searches are performed:
          lower bound — first i where text[SA[i] : SA[i]+m] ≥ pattern
          upper bound — first i where text[SA[i] : SA[i]+m] >  pattern

        Cost per call: O(m × log n) byte comparisons.
        For m = 650, n = 550 M: 650 × 30 ≈ 19 500 byte comparisons.
        Each comparison slices m bytes from an immutable bytes object; CPython
        reuses the underlying buffer, keeping allocation overhead negligible.
        """
        m = len(pattern_bytes)
        text = self._text
        sa = self._sa
        n = len(sa)

        # ── Lower bound ───────────────────────────────────────────────────────
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi) >> 1
            pos = int(sa[mid])
            if text[pos:pos + m] < pattern_bytes:
                lo = mid + 1
            else:
                hi = mid
        lower = lo

        # ── Upper bound (start from lower — already-searched space skipped) ───
        lo, hi = lower, n
        while lo < hi:
            mid = (lo + hi) >> 1
            pos = int(sa[mid])
            if text[pos:pos + m] <= pattern_bytes:
                lo = mid + 1
            else:
                hi = mid

        return lower, lo

    def _collect_hits(self, lo: int, hi: int, orientation: str) -> list[dict]:
        """
        Iterate over SA[lo:hi] and return one hit dict per valid match.

        NUL-separator positions and second-copy duplicates are eliminated by
        _abs_to_plasmid.  The loop is the only O(|hits|) step in a query.
        """
        hits: list[dict] = []
        for i in range(lo, hi):
            gbk_name, local_pos = self._abs_to_plasmid(int(self._sa[i]))
            if gbk_name is not None:
                hits.append({
                    "gbk_name":    gbk_name,
                    "position":    local_pos,
                    "orientation": orientation,
                })
        return hits

    # ── Public interface ──────────────────────────────────────────────────────

    def search_sequence(self, query: str) -> list[dict]:
        """
        Find all exact occurrences of ``query`` on both strands.

        RC hits are returned directly (orientation="rev"); the caller does
        NOT need to issue a second call.

        Each call is fully self-contained: two binary searches locate the
        matching SA ranges, then _collect_hits iterates over them linearly.
        Unlike PlasmidSearchAhoCorasick.batch_search, there is no database-
        scan cost to amortise — each call costs O(m log n) regardless of
        whether other queries are issued concurrently.
        """
        query = query.upper()
        fwd_bytes: bytes = query.encode()
        rev_bytes: bytes = self._reverse_complement(query).encode()

        lo, hi = self._sa_range(fwd_bytes)
        results: list[dict] = self._collect_hits(lo, hi, "fwd")

        # Palindrome guard: if fwd and RC are identical, both SA searches
        # would return the same range and every match would be reported twice.
        if rev_bytes != fwd_bytes:
            lo, hi = self._sa_range(rev_bytes)
            results.extend(self._collect_hits(lo, hi, "rev"))

        return results


if __name__ == "__main__":
    # --- Example usage ---
    CUR_DIR = Path(__file__).resolve()
    ADDGENE_DIR = CUR_DIR.parent.parent / "data/addgene"
    PLASMID_GBK = ADDGENE_DIR / "mammalian_plasmids.gbk"
    PLASMID_FASTA = ADDGENE_DIR / "mammalian_plasmids_wrapped.fasta"

    # # Initialize the API client
    # api = PlasmidSearchMinimap(PLASMID_FASTA)
    api = PlasmidSearchSuffixArray(PLASMID_FASTA)

    # # Query an arbitrary sequence
    query = "ATAGCGGCAGCCGTAGTAACAACAGTGGTGGCGCCGGTGGTGGTAGTGGCGGTAGCAGTAGCAGCAAAGGCG"
    hits = api.search_sequence(query)
    print(f"Found {len(hits)} hits: {hits}")
