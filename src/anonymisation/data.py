"""
TAB dataset loader.

The Text Anonymization Benchmark is 1,268 ECHR court cases, manually
annotated with entity mentions tagged by type (PERSON, ORG, LOC, etc.) and
identifier role (DIRECT, QUASI, NO_MASK).

We pull it directly from the HuggingFace Hub. First load is ~50 MB and
caches locally; subsequent loads are instant.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:  # import only for type checkers — keeps `import anonymisation`
    from datasets import DatasetDict  # light (no datasets/pyarrow needed to import the package)

DEFAULT_HF_PATH = "ildpil/text-anonymization-benchmark"


def load_tab(hf_path: str = DEFAULT_HF_PATH) -> "DatasetDict":
    """Load TAB from HuggingFace. Returns a DatasetDict with train/validation/test splits."""
    from datasets import load_dataset  # lazy — only Phase-1+ data work pulls this dep
    return load_dataset(hf_path)


def get_dataset_summary(dataset: DatasetDict) -> Dict[str, Any]:
    """
    Compute split sizes, entity-type counts, and identifier-type counts
    across an entire DatasetDict.

    Returns
    -------
    dict with keys:
      - splits          : {split_name: doc_count}
      - entity_types    : Counter of entity_type → count
      - identifier_types: Counter of identifier_type → count
      - entity_by_id    : {entity_type: Counter(identifier_type → count)}
      - per_doc_counts  : {split_name: [entity_count_per_doc, ...]}
    """
    entity_types: Counter = Counter()
    identifier_types: Counter = Counter()
    entity_by_id: Dict[str, Counter] = defaultdict(Counter)
    per_doc_counts: Dict[str, list] = {split: [] for split in dataset}
    splits: Dict[str, int] = {}

    for split in dataset:
        splits[split] = len(dataset[split])
        for doc in dataset[split]:
            per_doc_counts[split].append(len(doc["entity_mentions"]))
            for em in doc["entity_mentions"]:
                entity_types[em["entity_type"]] += 1
                identifier_types[em["identifier_type"]] += 1
                entity_by_id[em["entity_type"]][em["identifier_type"]] += 1

    return {
        "splits": splits,
        "entity_types": entity_types,
        "identifier_types": identifier_types,
        "entity_by_id": dict(entity_by_id),
        "per_doc_counts": per_doc_counts,
    }
