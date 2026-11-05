"""Local, network-free controls; never edits archived experiments or manuscript.

Uses the archived queries/split. Coarse queries were ontology-generated: even
the no-expansion controls are NOT end-to-end ontology-free query generation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rdflib
import sklearn

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT
ARCHIVE = PACKAGE / 'reported_retrieval'
METRICS = ['MRR', 'MRR@10', 'nDCG@10', 'Recall@5', 'Recall@10', 'Recall@20']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def is_dev(source):
    return int(hashlib.sha1(source.encode()).hexdigest()[:8], 16) % 5 == 0


def measures(positions):
    p = np.asarray(positions, dtype=float)
    return np.column_stack([1/p, np.where(p <= 10, 1/p, 0),
                            np.where(p <= 10, 1/np.log2(p+1), 0),
                            p <= 5, p <= 10, p <= 20])


def rank(scores):
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))


def cluster_intervals(values, sources, draws=2000):
    """Resample source-document clusters, retain query weighting within draw."""
    groups = sorted(set(sources))
    sums = np.array([values[np.array(sources) == g].sum(axis=0) for g in groups])
    sizes = np.array([sources.count(g) for g in groups])
    rng = np.random.default_rng(20261023)
    samples = rng.integers(0, len(groups), size=(draws, len(groups)))
    means = sums[samples].sum(axis=1) / sizes[samples].sum(axis=1)[:, None]
    return np.quantile(means, [.025, .975], axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--corpus-dir', type=Path, required=True,
                        help='Authorized structured corpus directory; not distributed here.')
    parser.add_argument('--ontology', type=Path, required=True,
                        help='Exact legacy retrieval ontology, NOT the final full-corpus release graph.')
    args = parser.parse_args()
    # Refuse even partial overwrite of a previous run.
    args.out.mkdir(parents=True, exist_ok=False)
    pilot_path = ARCHIVE/'run_evidence_retrieval_pilot.py'
    spec = importlib.util.spec_from_file_location('archived_pilot', pilot_path)
    pilot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pilot)
    pilot.CORPUS_DIR = args.corpus_dir.resolve()
    pilot.ONTOLOGY_FILE = args.ontology.resolve()
    names, texts, by_name = pilot.load_corpus()
    queries = read_csv(ARCHIVE/'evidence_retrieval_query_results.csv')
    label_index, parents, children = pilot.load_ontology()
    bm = pilot.BM25(texts)
    # Cache counters only; preserve legacy floating-point operation order.
    counts = [Counter(d) for d in bm.docs]
    def scores(q):
        out = []
        for doc, count in zip(bm.docs, counts):
            score = 0.
            for term, qtf in Counter(q).items():
                if term not in count:
                    continue
                idf = math.log(1+(bm.n-bm.df[term]+.5)/(bm.df[term]+.5))
                tf = count[term]
                denom = tf+1.5*(1-.75+.75*len(doc)/max(bm.avgdl, 1e-9))
                score += idf*(tf*2.5/denom)*(1+math.log(qtf))
            out.append(score)
        return out
    systems = ['coarse', 'fine_only', 'coarse_schema', 'same_hint',
               'same_hint_schema', 'same_hint_schema_seed',
               'same_hint_schema_graph', 'legacy_graph']
    vectors = {s: [] for s in systems}
    query_tokens = []
    checks = []
    gold = []
    for row in queries:
        subject, obj, relation = row['subject'], row['object'], row['relation']
        coarse = pilot.tokens(row['coarse_query'])
        hint = pilot.tokens(subject+' '+obj)
        schema = pilot.tokens(' '.join(pilot.RELATION_EXPANSIONS.get(relation, [])))
        seed = []
        graph_terms = []
        for phrase in (subject, obj):
            seed += [pilot.local_label(n) for n in pilot.matched_entities(phrase, label_index)]
            graph_terms += pilot.hierarchy_expansion(phrase, label_index, parents, children)
        graph = pilot.tokens(' '.join(graph_terms))
        checks.append(pilot.unique(graph_terms) == ([x for x in row['coarse_evi_expansion'].split(' | ') if x]))
        tokens_by_system = {
            'coarse': coarse, 'fine_only': pilot.tokens(f'{subject} {relation} {obj}'),
            'coarse_schema': coarse+schema, 'same_hint': coarse+hint,
            'same_hint_schema': coarse+schema+hint,
            'same_hint_schema_seed': coarse+schema+hint+pilot.tokens(' '.join(seed)),
            'same_hint_schema_graph': coarse+schema+hint+graph,
            'legacy_graph': coarse+schema+graph,
        }
        for s, ts in tokens_by_system.items():
            vectors[s].append(scores(ts))
        query_tokens.append({'query_id': row['query_id'], 'tokens': tokens_by_system})
        gold.append(by_name[row['source_file']])
    dev = [i for i,q in enumerate(queries) if is_dev(q['source_file'])]
    test = [i for i,q in enumerate(queries) if not is_dev(q['source_file'])]
    assert (len(names), len(queries), len(dev), len(test)) == (100, 247, 89, 158)
    assert not ({queries[i]['source_file'] for i in dev} & {queries[i]['source_file'] for i in test})
    assert all(checks), f'Current graph expansions differ on {len(checks)-sum(checks)} queries; do not interpret as reproduction.'
    def positions(vs):
        return [rank(v).index(g)+1 for v,g in zip(vs, gold)]
    pos = {s: positions(vs) for s,vs in vectors.items()}
    verified = {}
    for system, column in [('coarse', 'BM25_coarse_rank'),
                           ('fine_only', 'BM25_rank'),
                           ('coarse_schema', 'BM25+schema_coarse_rank'),
                           ('legacy_graph', 'BM25+EviMetOnto_coarse_rank')]:
        verified[system] = sum(p == int(q[column]) for p,q in zip(pos[system], queries))
    assert all(v == len(queries) for v in verified.values()), verified
    tuning = []
    selected = {}
    for expanded in ['same_hint_schema', 'same_hint_schema_seed', 'same_hint_schema_graph', 'legacy_graph']:
        system = 'fusion_'+expanded
        candidates = []
        for alpha in [i/10 for i in range(1,10)]:
            vs = [[(1-alpha)*a+alpha*b for a,b in zip(pilot.normalise_scores(raw), pilot.normalise_scores(exp))]
                  for raw,exp in zip(vectors['coarse'], vectors[expanded])]
            pp = positions(vs)
            ndcg = float(measures([pp[i] for i in dev])[:,2].mean())
            tuning.append({'system': system, 'alpha': alpha, 'dev_nDCG@10': ndcg})
            candidates.append((ndcg, -alpha, alpha, vs, pp))
        best = max(candidates, key=lambda c: (c[0], c[1]))
        selected[system] = best[2]
        vectors[system], pos[system] = best[3], best[4]
    old = next(r for r in read_csv(ARCHIVE/'evidence_retrieval_summary.csv') if r['system']=='BM25+EviMetOnto_hybrid_coarse')
    old_metrics = measures([pos['fusion_legacy_graph'][i] for i in test]).mean(axis=0)
    assert selected['fusion_legacy_graph'] == float(old['tuned_alpha_on_dev'])
    assert abs(old_metrics[0]-float(old['MRR@10'])) < 1e-12
    assert abs(old_metrics[2]-float(old['nDCG@10'])) < 1e-12
    summaries, rank_rows = [], []
    sources = [queries[i]['source_file'] for i in test]
    for s in vectors:
        for split, indices in [('dev', dev), ('test', test)]:
            vals = measures([pos[s][i] for i in indices])
            means = vals.mean(axis=0)
            ci = cluster_intervals(vals, sources) if split=='test' else None
            for j,m in enumerate(METRICS):
                summaries.append({'system': s, 'split': split, 'n_queries': len(indices), 'metric': m,
                                  'value': means[j], 'ci_low': ci[0,j] if ci is not None else '',
                                  'ci_high': ci[1,j] if ci is not None else '', 'alpha': selected.get(s, '')})
        rank_rows += [{'query_id': q['query_id'], 'source_file': q['source_file'],
                       'split': 'dev' if i in dev else 'test', 'system': s,
                       'positive_rank': pos[s][i]} for i,q in enumerate(queries)]
    comparisons = [('legacy_graph', 'same_hint_schema'),
                   ('fusion_legacy_graph', 'fusion_same_hint_schema'),
                   ('same_hint_schema_graph', 'same_hint_schema'),
                   ('same_hint_schema_graph', 'same_hint_schema_seed'),
                   ('fusion_same_hint_schema_graph', 'fusion_same_hint_schema'),
                   ('same_hint', 'coarse')]
    deltas = []
    for a,b in comparisons:
        delta = measures([pos[a][i] for i in test])-measures([pos[b][i] for i in test])
        ci = cluster_intervals(delta, sources)
        for j,m in enumerate(METRICS):
            deltas.append({'A': a, 'B': b, 'metric': m, 'delta_A_minus_B': delta[:,j].mean(),
                           'ci_low': ci[0,j], 'ci_high': ci[1,j]})
    write_csv(args.out/'metrics.csv', summaries)
    write_csv(args.out/'positive_ranks.csv', rank_rows)
    write_csv(args.out/'dev_tuning.csv', tuning)
    write_csv(args.out/'paired_deltas.csv', deltas)
    with (args.out/'full_scores.jsonl').open('w', encoding='utf-8') as f:
        for i,q in enumerate(queries):
            f.write(json.dumps({'query_id': q['query_id'], 'scores': {s:vs[i] for s,vs in vectors.items()}})+'\n')
    (args.out/'query_tokens.json').write_text(json.dumps(query_tokens, indent=2), encoding='utf-8')
    inputs = [Path(__file__), pilot_path, pilot.ONTOLOGY_FILE,
              ARCHIVE/'evidence_retrieval_query_results.csv', ARCHIVE/'evidence_retrieval_summary.csv']
    inputs += sorted(pilot.CORPUS_DIR.glob('*.json'))
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(), 'python': platform.python_version(),
                'numpy': np.__version__, 'sklearn': sklearn.__version__, 'rdflib': rdflib.__version__,
                'documents': names, 'dev_queries': len(dev), 'test_queries': len(test),
                'test_source_documents': len(set(sources)), 'legacy_rank_matches': verified,
                'legacy_hybrid_reproduced': True, 'selected_alpha': selected,
                'bootstrap': '2000 paired source-document cluster draws; seed 20261023; percentile 95% CI; descriptive, no multiplicity correction',
                'relevance': 'Single source-document positive from relation record, NOT independent expert relevance judgments',
                'limitation': 'Archived coarse query construction uses ontology. No-ontology means no ontology expansion at retrieval time, NOT ontology-free query generation. Source-derived fine hints may strongly identify the known source.',
                'inputs_sha256': {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): sha(p) for p in inputs}}
    (args.out/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    write_csv(args.out/'output_sha256.csv', [{'file': p.name, 'sha256': sha(p)} for p in sorted(args.out.iterdir()) if p.is_file()])
    print(json.dumps({'output': str(args.out), 'legacy_reproduced': True, 'alpha': selected}, indent=2))


if __name__ == '__main__':
    main()
