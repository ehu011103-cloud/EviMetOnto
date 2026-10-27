from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from rdflib import Graph, Literal, OWL, RDF, RDFS, SKOS, URIRef
from sklearn.feature_extraction.text import TfidfVectorizer


ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = ROOT / "corpus_json"
RELATIONS_FILE = ROOT / "refined_relations_ir.json"
ONTOLOGY_FILE = ROOT / "100测试" / "final_4D_root_dedup_validated_latest.ttl"
OUT_DIR = Path(__file__).resolve().parent / "evidence_retrieval_results"

RELATION_EXPANSIONS = {
    "CAUSES": ["cause", "causes", "caused", "formation", "origin"],
    "CONTROLS": ["control", "controls", "controlled", "influence"],
    "INDICATES": ["indicate", "indicates", "indicator", "evidence"],
    "ASSOCIATED_WITH": ["associated", "association", "related"],
    "HOSTED_IN": ["hosted", "host rock", "hosted in"],
    "INTRUDES_INTO": ["intrude", "intrudes", "intrusion"],
    "ZONES_WITH": ["zone", "zoning", "zonation"],
    "DISTRIBUTED_ALONG": ["distributed", "along", "structure"],
    "PRECEDES": ["precede", "precedes", "earlier", "before"],
    "OVERPRINTS": ["overprint", "overprints", "overprinting"],
    "SYNCHRONOUS_WITH": ["synchronous", "contemporaneous"],
    "FORMED_DURING": ["formed", "during", "event", "period"],
}


def tokens(text: str) -> list[str]:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(text))
    text = text.replace("_", " ").replace("-", " ")
    return re.findall(r"[a-z0-9]+", text.lower())


def unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))


def load_corpus() -> tuple[list[str], list[str], dict[str, int]]:
    names = []
    texts = []
    by_name = {}
    for path in sorted(CORPUS_DIR.glob("*.json")):
        obj = json.loads(path.read_text(encoding="utf-8"))
        metadata = obj.get("metadata", {})
        content = obj.get("content", [])
        fields = [
            metadata.get("title", ""),
            metadata.get("abstract", ""),
            " ".join(metadata.get("keywords", []) or []),
            " ".join(content if isinstance(content, list) else [str(content)]),
        ]
        idx = len(names)
        names.append(path.name)
        texts.append(" ".join(fields))
        by_name[path.name] = idx
    return names, texts, by_name


def load_ontology() -> tuple[dict[str, set[URIRef]], dict[URIRef, set[URIRef]], dict[URIRef, set[URIRef]]]:
    graph = Graph()
    graph.parse(ONTOLOGY_FILE)
    classes = {s for s in graph.subjects(RDF.type, OWL.Class) if isinstance(s, URIRef)}
    classes |= {s for s in graph.subjects(RDF.type, RDFS.Class) if isinstance(s, URIRef)}
    parents: dict[URIRef, set[URIRef]] = defaultdict(set)
    children: dict[URIRef, set[URIRef]] = defaultdict(set)
    for child, parent in graph.subject_objects(RDFS.subClassOf):
        if isinstance(child, URIRef) and isinstance(parent, URIRef):
            classes.update((child, parent))
            parents[child].add(parent)
            children[parent].add(child)

    labels: dict[URIRef, set[str]] = defaultdict(set)
    for entity in classes:
        text = str(entity).rstrip("/#").rsplit("#", 1)[-1].rsplit("/", 1)[-1]
        labels[entity].add(text)
        for pred in (RDFS.label, SKOS.prefLabel, SKOS.altLabel):
            for value in graph.objects(entity, pred):
                if isinstance(value, Literal):
                    labels[entity].add(str(value))

    index: dict[str, set[URIRef]] = defaultdict(set)
    for entity, entity_labels in labels.items():
        for label in entity_labels:
            key = " ".join(tokens(label))
            if key:
                index[key].add(entity)
    return index, parents, children


def ancestors(node: URIRef, parents: dict[URIRef, set[URIRef]], limit: int = 2) -> set[URIRef]:
    found: set[URIRef] = set()
    frontier = {node}
    for _ in range(limit):
        next_frontier = set()
        for item in frontier:
            next_frontier.update(parents.get(item, set()))
        next_frontier -= found
        found.update(next_frontier)
        frontier = next_frontier
    return found


def descendants(node: URIRef, children: dict[URIRef, set[URIRef]], limit: int = 2) -> set[URIRef]:
    found: set[URIRef] = set()
    frontier = {node}
    for _ in range(limit):
        next_frontier = set()
        for item in frontier:
            next_frontier.update(children.get(item, set()))
        next_frontier -= found
        found.update(next_frontier)
        frontier = next_frontier
    return found


def local_label(uri: URIRef) -> str:
    return str(uri).rstrip("/#").rsplit("#", 1)[-1].rsplit("/", 1)[-1]


def matched_entities(phrase: str, label_index: dict[str, set[URIRef]]) -> list[URIRef]:
    phrase_tokens = set(tokens(phrase))
    if not phrase_tokens:
        return []
    matches: list[tuple[float, URIRef]] = []
    for label, entities in label_index.items():
        label_tokens = set(label.split())
        overlap = len(phrase_tokens & label_tokens) / max(len(phrase_tokens | label_tokens), 1)
        containment = len(phrase_tokens & label_tokens) / max(len(phrase_tokens), 1)
        score = max(overlap, containment * 0.85)
        if score >= 0.5:
            matches.extend((score, entity) for entity in entities)
    matches.sort(key=lambda item: (-item[0], local_label(item[1])))
    return list(dict.fromkeys(entity for _, entity in matches[:3]))


def ontology_expansion(phrase: str, label_index: dict[str, set[URIRef]], parents: dict[URIRef, set[URIRef]]) -> list[str]:
    expansion = []
    for entity in matched_entities(phrase, label_index):
        expansion.append(local_label(entity))
        expansion.extend(sorted(local_label(parent) for parent in ancestors(entity, parents)))
    return expansion


def coarse_phrase(phrase: str, label_index: dict[str, set[URIRef]], parents: dict[URIRef, set[URIRef]]) -> str:
    matches = matched_entities(phrase, label_index)
    if not matches:
        return phrase
    parent_nodes = ancestors(matches[0], parents, limit=1)
    return sorted(local_label(node) for node in parent_nodes)[0] if parent_nodes else phrase


def hierarchy_expansion(
    phrase: str,
    label_index: dict[str, set[URIRef]],
    parents: dict[URIRef, set[URIRef]],
    children: dict[URIRef, set[URIRef]],
) -> list[str]:
    expansion = []
    for entity in matched_entities(phrase, label_index):
        expansion.append(local_label(entity))
        expansion.extend(sorted(local_label(node) for node in ancestors(entity, parents, limit=1)))
        expansion.extend(sorted(local_label(node) for node in descendants(entity, children, limit=2)))
    return expansion


class BM25:
    def __init__(self, documents: list[str]) -> None:
        self.docs = [tokens(doc) for doc in documents]
        self.n = len(self.docs)
        self.avgdl = sum(len(doc) for doc in self.docs) / max(self.n, 1)
        self.df = Counter()
        for doc in self.docs:
            self.df.update(set(doc))

    def scores(self, query_tokens: list[str]) -> list[float]:
        query = Counter(query_tokens)
        k1, b = 1.5, 0.75
        scores = []
        for idx, doc in enumerate(self.docs):
            counts = Counter(doc)
            score = 0.0
            for term, qtf in query.items():
                if term not in counts:
                    continue
                df = self.df[term]
                idf = math.log(1.0 + (self.n - df + 0.5) / (df + 0.5))
                tf = counts[term]
                denom = tf + k1 * (1 - b + b * len(doc) / max(self.avgdl, 1e-9))
                score += idf * (tf * (k1 + 1) / denom) * (1 + math.log(qtf))
            scores.append(score)
        return scores

    def rank(self, query_tokens: list[str]) -> list[int]:
        scores = self.scores(query_tokens)
        return sorted(range(self.n), key=lambda idx: scores[idx], reverse=True)


def rank_score_vector(scores: list[float]) -> list[int]:
    return sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)


def normalise_scores(scores: list[float]) -> list[float]:
    low, high = min(scores), max(scores)
    if high <= low:
        return [0.0 for _ in scores]
    return [(score - low) / (high - low) for score in scores]


def metric_summary(ranks: list[list[int]], gold: list[int]) -> dict[str, float]:
    reciprocal = []
    recalls = {k: [] for k in (5, 10, 20)}
    ndcgs = []
    for ranking, target in zip(ranks, gold):
        try:
            pos = ranking.index(target) + 1
        except ValueError:
            pos = None
        reciprocal.append(1.0 / pos if pos else 0.0)
        for k in recalls:
            recalls[k].append(float(target in ranking[:k]))
        ndcgs.append(1.0 / math.log2(pos + 1) if pos and pos <= 10 else 0.0)
    result = {
        "n_queries": len(gold),
        "MRR@10": sum(reciprocal) / max(len(reciprocal), 1),
        "nDCG@10": sum(ndcgs) / max(len(ndcgs), 1),
    }
    result.update({f"Recall@{k}": sum(values) / max(len(values), 1) for k, values in recalls.items()})
    return result


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    names, texts, by_name = load_corpus()
    label_index, parents, children = load_ontology()
    relations = json.loads(RELATIONS_FILE.read_text(encoding="utf-8"))
    queries = []
    for item in relations:
        source = Path(str(item.get("source_file", ""))).name
        if source not in by_name:
            continue
        queries.append(item)

    bm25 = BM25(texts)
    word_vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
    char_vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), lowercase=True)
    word_matrix = word_vectorizer.fit_transform(texts)
    char_matrix = char_vectorizer.fit_transform(texts)
    result_rows = []
    all_ranks = {
        "BM25": [], "BM25+schema": [], "BM25+EviMetOnto": [],
        "BM25_coarse": [], "BM25+schema_coarse": [], "BM25+EviMetOnto_coarse": [],
        "TFIDF_word_coarse": [], "TFIDF_char_coarse": [], "TFIDF_word_char_hybrid_coarse": [],
    }
    score_vectors = {"coarse_raw": [], "coarse_evi": []}
    query_sources = []
    gold = []
    for qid, item in enumerate(queries, start=1):
        subject = str(item.get("subject", ""))
        relation = str(item.get("relation", ""))
        obj = str(item.get("object", ""))
        query = f"{subject} {relation} {obj}"
        raw_tokens = tokens(query)
        schema_tokens = raw_tokens + tokens(" ".join(RELATION_EXPANSIONS.get(relation, [])))
        evi_terms = ontology_expansion(subject, label_index, parents) + ontology_expansion(obj, label_index, parents)
        evi_tokens = schema_tokens + tokens(" ".join(evi_terms))
        coarse_subject = coarse_phrase(subject, label_index, parents)
        coarse_object = coarse_phrase(obj, label_index, parents)
        coarse_query = f"{coarse_subject} {relation} {coarse_object}"
        coarse_tokens = tokens(coarse_query)
        coarse_schema_tokens = coarse_tokens + tokens(" ".join(RELATION_EXPANSIONS.get(relation, [])))
        coarse_evi_terms = hierarchy_expansion(subject, label_index, parents, children)
        coarse_evi_terms += hierarchy_expansion(obj, label_index, parents, children)
        coarse_evi_tokens = coarse_schema_tokens + tokens(" ".join(coarse_evi_terms))
        word_query = word_vectorizer.transform([coarse_query])
        char_query = char_vectorizer.transform([coarse_query])
        word_scores = (word_matrix @ word_query.T).toarray().ravel().tolist()
        char_scores = (char_matrix @ char_query.T).toarray().ravel().tolist()
        hybrid_scores = [
            0.5 * word_score + 0.5 * char_score
            for word_score, char_score in zip(normalise_scores(word_scores), normalise_scores(char_scores))
        ]
        target = by_name[Path(str(item["source_file"])).name]
        rankings = {
            "BM25": bm25.rank(raw_tokens),
            "BM25+schema": bm25.rank(schema_tokens),
            "BM25+EviMetOnto": bm25.rank(evi_tokens),
            "BM25_coarse": bm25.rank(coarse_tokens),
            "BM25+schema_coarse": bm25.rank(coarse_schema_tokens),
            "BM25+EviMetOnto_coarse": bm25.rank(coarse_evi_tokens),
            "TFIDF_word_coarse": rank_score_vector(word_scores),
            "TFIDF_char_coarse": rank_score_vector(char_scores),
            "TFIDF_word_char_hybrid_coarse": rank_score_vector(hybrid_scores),
        }
        coarse_raw_scores = bm25.scores(coarse_tokens)
        coarse_evi_scores = bm25.scores(coarse_evi_tokens)
        score_vectors["coarse_raw"].append(coarse_raw_scores)
        score_vectors["coarse_evi"].append(coarse_evi_scores)
        query_sources.append(names[target])
        gold.append(target)
        for system, ranking in rankings.items():
            all_ranks[system].append(ranking)
        result_rows.append({
            "query_id": qid,
            "source_file": names[target],
            "subject": subject,
            "relation": relation,
            "object": obj,
            "coarse_query": coarse_query,
            "evi_expansion": " | ".join(unique(evi_terms)),
            "coarse_evi_expansion": " | ".join(unique(coarse_evi_terms)),
            **{f"{system}_rank": ranking.index(target) + 1 if target in ranking else "NA" for system, ranking in rankings.items()},
        })

    summaries = []
    for system, ranks in all_ranks.items():
        summaries.append({"system": system, **metric_summary(ranks, gold)})

    dev_indices = [
        idx for idx, source in enumerate(query_sources)
        if int(hashlib.sha1(source.encode("utf-8")).hexdigest()[:8], 16) % 5 == 0
    ]
    test_indices = [idx for idx in range(len(gold)) if idx not in dev_indices]
    best_alpha = None
    best_dev = None
    for alpha in [round(i / 10, 1) for i in range(1, 10)]:
        ranks = []
        targets = []
        for idx in dev_indices:
            raw = normalise_scores(score_vectors["coarse_raw"][idx])
            evi = normalise_scores(score_vectors["coarse_evi"][idx])
            ranks.append(rank_score_vector([(1 - alpha) * r + alpha * e for r, e in zip(raw, evi)]))
            targets.append(gold[idx])
        candidate = metric_summary(ranks, targets)
        if best_dev is None or candidate["nDCG@10"] > best_dev["nDCG@10"]:
            best_alpha, best_dev = alpha, candidate

    hybrid_ranks = []
    hybrid_targets = []
    for idx in test_indices:
        raw = normalise_scores(score_vectors["coarse_raw"][idx])
        evi = normalise_scores(score_vectors["coarse_evi"][idx])
        hybrid_ranks.append(rank_score_vector([(1 - best_alpha) * r + best_alpha * e for r, e in zip(raw, evi)]))
        hybrid_targets.append(gold[idx])
    hybrid_summary = {
        "system": "BM25+EviMetOnto_hybrid_coarse",
        **metric_summary(hybrid_ranks, hybrid_targets),
        "dev_queries": len(dev_indices),
        "test_queries": len(test_indices),
        "tuned_alpha_on_dev": best_alpha,
    }
    summaries.append(hybrid_summary)
    for system in (
        "BM25_coarse",
        "BM25+schema_coarse",
        "BM25+EviMetOnto_coarse",
        "TFIDF_word_coarse",
        "TFIDF_char_coarse",
        "TFIDF_word_char_hybrid_coarse",
    ):
        summaries.append({
            "system": f"{system}_same_test_split",
            **metric_summary(
                [all_ranks[system][idx] for idx in test_indices],
                [gold[idx] for idx in test_indices],
            ),
            "dev_queries": len(dev_indices),
            "test_queries": len(test_indices),
        })

    with (OUT_DIR / "evidence_retrieval_query_results.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=result_rows[0].keys())
        writer.writeheader()
        writer.writerows(result_rows)
    with (OUT_DIR / "evidence_retrieval_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        summary_fields = list(dict.fromkeys(field for row in summaries for field in row.keys()))
        writer = csv.DictWriter(handle, fieldnames=summary_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)
    manifest = {
        "task": "metallogenic evidence retrieval",
        "corpus_documents": len(names),
        "queries": len(queries),
        "positive_definition": "source document attached to the validated relation record",
        "systems": list(all_ranks) + ["BM25+EviMetOnto_hybrid_coarse"],
        "coarse_task": "replace matched specific concepts with one ontology parent, then expand the coarse query with ontology descendants",
        "status": "weakly supervised pilot; source-document positives are derived from the existing relation extraction records",
        "warning": "Do not report as an independent expert-labeled downstream benchmark without a separately judged query-document pool.",
        "dense_retrieval": "not run: sentence-transformers import fails because the local PyTorch DLL path is invalid",
    }
    (OUT_DIR / "evidence_retrieval_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
