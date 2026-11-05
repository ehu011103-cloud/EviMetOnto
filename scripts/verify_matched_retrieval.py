"""Reconstruct audit results from full score vectors; no source texts or APIs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from run_matched_retrieval import measures, rank, is_dev, cluster_intervals


def read_csv(p):
    with p.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def verify(root):
    audit = root/'retrieval_controls_20261023'
    manifest = json.loads((audit/'manifest.json').read_text(encoding='utf-8'))
    queries = read_csv(root/'reported_retrieval/evidence_retrieval_query_results.csv')
    vectors = [json.loads(x) for x in (audit/'full_scores.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(vectors) == len(queries) == 247
    names = manifest['documents']
    assert len(names) == 100 and len(set(names)) == 100
    systems = list(vectors[0]['scores'])
    assert len(systems) == 12
    targets = [names.index(q['source_file']) for q in queries]
    dev = [i for i,q in enumerate(queries) if is_dev(q['source_file'])]
    test = [i for i,q in enumerate(queries) if not is_dev(q['source_file'])]
    sources = [queries[i]['source_file'] for i in test]
    assert len(dev) == 89 and len(test) == 158 and len(set(sources)) == 19
    pos = {}
    for s in systems:
        assert all(len(v['scores'][s]) == 100 and np.isfinite(v['scores'][s]).all() for v in vectors)
        pos[s] = [rank(v['scores'][s]).index(t)+1 for v,t in zip(vectors,targets)]
    query_index = {q['query_id']:i for i,q in enumerate(queries)}
    for i,(q,v) in enumerate(zip(queries,vectors)):
        assert int(q['query_id']) == int(v['query_id'])
    ranks = read_csv(audit/'positive_ranks.csv')
    assert len(ranks) == 247*12
    assert len({(r['query_id'],r['system']) for r in ranks}) == 247*12
    for r in ranks:
        i = query_index[r['query_id']]
        assert int(r['positive_rank']) == pos[r['system']][i]
        assert r['split'] == ('dev' if i in dev else 'test')
    metric_names = ['MRR','MRR@10','nDCG@10','Recall@5','Recall@10','Recall@20']
    stats = {}
    cis = {}
    for s in systems:
        for split,inds in [('dev',dev),('test',test)]:
            values = measures([pos[s][i] for i in inds])
            stats[s,split] = values.mean(axis=0)
            if split == 'test': cis[s] = cluster_intervals(values,sources)
    rows = read_csv(audit/'metrics.csv')
    assert len(rows) == 12*2*6
    for r in rows:
        j = metric_names.index(r['metric'])
        assert abs(float(r['value'])-stats[r['system'],r['split']][j]) < 1e-12
        if r['split'] == 'test':
            assert abs(float(r['ci_low'])-cis[r['system']][0,j]) < 1e-12
            assert abs(float(r['ci_high'])-cis[r['system']][1,j]) < 1e-12
    for row in read_csv(audit/'paired_deltas.csv'):
        delta = measures([pos[row['A']][i] for i in test])-measures([pos[row['B']][i] for i in test])
        j = metric_names.index(row['metric'])
        ci = cluster_intervals(delta,sources)
        assert abs(float(row['delta_A_minus_B'])-delta[:,j].mean()) < 1e-12
        assert abs(float(row['ci_low'])-ci[0,j]) < 1e-12
        assert abs(float(row['ci_high'])-ci[1,j]) < 1e-12
    tuning = read_csv(audit/'dev_tuning.csv')
    for fusion,selected in manifest['selected_alpha'].items():
        expanded = fusion.removeprefix('fusion_')
        vals = []
        for alpha in [i/10 for i in range(1,10)]:
            positions = []
            for i,v in enumerate(vectors):
                def norm(a):
                    lo,hi=min(a),max(a)
                    return [(x-lo)/(hi-lo) for x in a] if hi>lo else [0.]*len(a)
                raw,exp=norm(v['scores']['coarse']),norm(v['scores'][expanded])
                combined=[(1-alpha)*a+alpha*b for a,b in zip(raw,exp)]
                positions.append(rank(combined).index(targets[i])+1)
                if alpha == selected:
                    np.testing.assert_allclose(combined,v['scores'][fusion],rtol=0,atol=1e-14)
            ndcg=measures([positions[i] for i in dev])[:,2].mean()
            recorded=next(r for r in tuning if r['system']==fusion and float(r['alpha'])==alpha)
            assert abs(ndcg-float(recorded['dev_nDCG@10'])) < 1e-12
            vals.append((ndcg,-alpha,alpha))
        assert max(vals)[2] == selected
    old = read_csv(audit/'output_sha256.csv')
    for r in old:
        assert hashlib.sha256((audit/r['file']).read_bytes()).hexdigest() == r['sha256']
    return {'status':'pass','queries':247,'systems':12,'test_source_documents':19,
            'verified':['score-derived ranks','all metric summaries','cluster intervals',
                        'paired differences','development tuning and fusion scores','original audit output hashes'],
            'boundary':'No text scoring or model inference performed; no independent relevance annotation; shared-model main diagnostic predictions still unavailable.'}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    result=verify(args.root.resolve())
    if args.output: args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
