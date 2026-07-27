## Addgene download

1. `addgene_plasmids_with_sequences.json` is the primary file downloaded from Addgene on 27.4.2026.

Private token (received after the request is approved) is available on the personal account page at [Addgene Developers Portal](https://developers.addgene.org/account/#access-section). See the [download documentation](https://docs.developers.addgene.org/docs/).

Download command:
```bash
TOKEN="<private TOKEN>"

# Bulk Download: Plasmids with Sequences (downloaded on 27.4.2026)
curl --request GET \
     --location \
     --header "Authorization: Token ${TOKEN}" \
     --url https://api.developers.addgene.org/download/plasmids_with_sequences/ \
     --output addgene_plasmids_with_sequences.json
```

2. `mammalian_plasmids.gbk` is a collection of GenBank records for mammalian expression vectors listed in `addgene_plasmids_with_sequences.json`. Each record is downloaded from Addgene using the following request (requires private token, done via Python in the [`01_addgene_gbk_download.py`](../scripts/01_addgene_gbk_download.py)):

```bash
# Download a specific plasmid of interest
PLASMID_ID=100004

curl --request GET \
  --location \
  --header "Authorization: Token ${TOKEN}" \
  --url https://api.developers.addgene.org/download/genbank/$PLASMID_ID/ \
  --output genbank_$PLASMID_ID.gbk
```

3. `citations_addgene.parquet` is obtained by programmatically accessing Addgene webpage for each plasmid and fetching the citations data from this page (done in the [`02_addgene_citations.py`](../scripts/02_addgene_citations.py) script).



## Buffer CREs formal procedure

Goal: Discovery of Widespread Plasmid Backbone Elements Overlapping Regulatory Signals

### 1. Terminology & Parameters

#### Terminology

* **Target Elements:** The predicted regulatory elements, specifically CREST CREs across all cell types and Puffin TSSs (forward and reverse).
* **Midpoint:** The exact center coordinate of a Target Element.
* **Core Window:** The restricted central span used to rigorously test for functional overlap.
  * *CRE:* Central 250 bp window (midpoint $\pm$ 125 bp)
  * *TSS:* Central 450 bp window (midpoint $\pm$ 225 bp)


* **Anchor Sequence:** The extended sequence extracted around the Target Element's midpoint, representing the biological context modeled by the neural networks. Used for downstream FM-index searches.
  * *CRE:* 200 bp window (midpoint $\pm$ 100 bp).
  * *TSS:* 650 bp window (midpoint $\pm$ 325 bp).

*NOTE*: core window might be wider / narrower than the anchor window. 


* **Hard Exclusion Feature:** Confirmed, annotated Addgene features greater than a defined length threshold (e.g., 100 bp) that imply a known function (e.g., CDS, origins, known promoters).
* **Soft Exclusion Feature:** Putative or unannotated inserts/regions that lack a definitive functional Addgene annotation but may harbor coding or non-coding function.

#### Global Parameters

* `MIN_FEATURE_EXCLUSION_LEN`: Minimum length for a Hard Exclusion Feature to trigger a drop (e.g., 100 bp).
* `CRE_CORE_WINDOW` / `TSS_CORE_WINDOW`: 250 bp / 450 bp.
* `CRE_ANCHOR_SIZE` / `TSS_ANCHOR_SIZE`: 200 bp / 650 bp.
* `MIN_ANCHOR_OVERLAP`: Minimum base-pair overlap required to merge shifted Anchor Sequences into a single identity group.

---

### 2. Phase 1: Discovery & Contextual Filtering

**Goal:** Identify all CREs and TSSs that reside in "buffer" (backbone) regions without disrupting known functional annotations at their core, and calculate their local signal strengths.

**Logical Steps:**

1. **Iterate Plasmids:** Loop through the entire plasmid dataset, loading predicted element positions (CREs, TSSs) and annotated features.
2. **Define Windows (Topology-Aware):** For every Target Element, calculate the **Midpoint**, **Core Window**, and **Anchor Window**. All coordinate arithmetic must strictly utilize modulo operations (`% plasmid_length`) to correctly wrap around the circular plasmid topology.
3. **Evaluate Overlap:**
* *Condition A (Drop):* If a **Hard Exclusion Feature** overlaps any part of the **Core Window**, discard the Target Element.
* *Condition B (Flag):* If a Hard Exclusion Feature overlaps the Target Element (but *strictly outside* the Core Window), keep the element but append the feature name/ID to `overlapped_features`.
* *Condition C (Flag):* If a **Soft Exclusion Feature** overlaps *anywhere* in the Core or Anchor windows, keep the element and append the putative insert ID to `overlapped_features`.


4. **Compute Signal Strength:** Load the localized signal predictions (CREST tiles or Puffin H5 tracks). Calculate the average signal strictly across the Target Element's defined base pairs to derive `cre_activity`.
5. **Output Generation:** Save the passing elements to `buffer_cres.parquet` with the following schema:
* `gbk_name`, `sequence_id`: Identifiers for the plasmid.
* `cre_raw_id`: Unique identifier for the regulatory element.
* `cre_type`: e.g., "CREST (HEK293T)", "Puffin (FANTOM_CAGE_fwd)".
* `cre_position`: Tuple of (start, end) or midpoint.
* `cre_length`: Length of the raw predicted element.
* `cre_activity`: Mean signal strength over the element span.
* `overlapped_features`: List of strings (features/inserts overlapping the Anchor/Core but not violating the exclusion rule).



---

Here is the updated Phase II specification incorporating all of your corrections. The logic is fully sound: by querying the *union* of the shifted variants, we ensure maximum sensitivity during the FM-index search, and isolating by `cre_type` keeps the biological contexts strictly delineated.

---
### 3. Phase 2: Global Cross-Reference & Deduplication

**Goal:** Extract the Anchor Sequences for the filtered elements, resolve slight positional shifts and reverse-complement redundancies within each regulatory type, and trace their prevalence across the global plasmid database. Crucially, hits across the global dataset must be cross-referenced with local plasmid annotations to differentiate true "buffer" occurrences from instances where the sequence is part of an annotated functional feature (e.g., a truncated promoter).

**Logical Steps:**

1. **Anchor Sequence Extraction:** Iterate through `buffer_cres.parquet`. Extract the exact nucleotide sequence corresponding to the Anchor Window (200 bp or 650 bp) for each element, ensuring circular wrap-around is handled.
2. **Sequence Deduplication & Shift Resolution (Isolated by `cre_type`):**
* Process elements strictly grouped by their `cre_type`.
* Since different elements might be shifted by a few base pairs relative to each other (yielding slightly different exact Anchor Sequences), group these sequences.
* **Reverse-Complement Resolution:** When evaluating overlap, compare the sequence against both the forward and reverse-complement (RC) of other sequences. If two Anchor Sequences share an exact matching overlap greater than `MIN_ANCHOR_OVERLAP` (e.g., shifted by 1-10 bp) in *either* orientation, they are treated as the same biological entity and assigned the same `anchor_id`. A relative strand orientation is temporarily assigned to align them.
* Merge their associated `cre_raw_id`s into a unified list.
* Retain the **union** of all exact anchor sequence variations belonging to this `anchor_id` for the downstream exact search.


3. **Global Exact Search:**
* For each unique `anchor_id`, query the **union** of its associated Anchor Sequences against the global plasmid dataset using the exact-match Suffix Array engine.
* **Orientation Agnostic Search & Tracking:** For every query sequence in the union, search both the *Forward* sequence and its *Reverse Complement*. Hits on either strand contribute to the same sequence identity.
* For every hit, determine and temporarily record the orientation (1 for forward match, -1 for reverse complement match).


4. **Functional Overlap Analysis & Citation Aggregation:**
* For each matched plasmid, record its `gbk_name`, the sequence's mapped `position`, and the matching `strand`.
* **Buffer Status Determination:** For every global hit, check the mapped coordinates against that specific plasmid's functional annotations.
* If the hit overlaps a known functional feature, mark it as `False` (not buffer) and record the overlapped feature's name.
* If the hit falls entirely within unannotated space, mark it as `True` (is buffer).


* **Aggregation:** Calculate `n_plasmids` and `n_citations` **only** for plasmids where the anchor resolves as a buffer sequence (i.e., `is_buffer == True`).
* **Feature Cross-Reference:** Aggregate all unique functional features overlapped across *all* instances of this anchor sequence into a single list. Check this aggregated list against a predefined database of known CREs to determine if the sequence corresponds to a recognized regulatory element.


5. **Output Generation:** Compile the aggregated global hits into a final dataset with the following schema:
* `anchor_id`: A unique ID for the grouped sequence cluster.
* `cre_type`: The regulatory element type (e.g., "CREST (HEK293T)") corresponding to this cluster.
* `cre_raw_ids`: List of all original CRE/TSS IDs from Phase 1 that belong to this sequence cluster.
* `cre_overlapped_features`: List of all unique features overlapped in the *source* plasmids (carried over from Phase 1).
* `gbk_names`: List of all plasmids containing this sequence across the global dataset.
* `positions`: List of coordinates mapping where the sequence occurs in the respective `gbk_names`.
* `strands`: List of integers (1 or -1) specifying the hit orientation, matching the order of elements in `gbk_names` and `positions`.
* `is_buffer`: List of booleans, aligned with `gbk_names`, indicating whether the hit in that specific plasmid falls in a buffer region (`True`) or overlaps an annotated feature (`False`).
* `n_plasmids`: Integer count of unique plasmids containing this sequence **specifically within a buffer region**.
* `n_citations`: Integer sum of all citations for the plasmids where the sequence is found **within a buffer region**.
* `overlapped_features`: List of strings containing all unique functional features that this anchor overlaps across the entire dataset.
* `overlaps_known_cre`: Boolean flag set to `True` if any feature in `overlapped_features` matches a known regulatory element (CRE).
* `vector_types`: List of strings representing the unique vector types (e.g., "Mammalian Expression", "Lentiviral") of all plasmids in which the anchor sequence was found.
