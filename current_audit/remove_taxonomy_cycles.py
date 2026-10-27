from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import networkx as nx
from rdflib import Graph, RDF, RDFS, OWL, URIRef


def class_entities(graph: Graph) -> set[URIRef]:
    classes = {
        subject
        for class_type in (OWL.Class, RDFS.Class)
        for subject in graph.subjects(RDF.type, class_type)
        if isinstance(subject, URIRef)
    }
    for child, parent in graph.subject_objects(RDFS.subClassOf):
        classes.update(node for node in (child, parent) if isinstance(node, URIRef))
    return classes


def display_label(graph: Graph, node: URIRef) -> str:
    return next((str(label) for label in graph.objects(node, RDFS.label)), str(node).rsplit("#", 1)[-1])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a cycle-free audit copy by quarantining all asserted subclass edges inside cyclic SCCs."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()

    graph = Graph()
    graph.parse(args.input)
    classes = class_entities(graph)
    edges = {
        (child, parent)
        for child, parent in graph.subject_objects(RDFS.subClassOf)
        if isinstance(child, URIRef) and isinstance(parent, URIRef)
    }
    taxonomy = nx.DiGraph()
    taxonomy.add_nodes_from(classes)
    taxonomy.add_edges_from(edges)
    cyclic_components = [
        component
        for component in nx.strongly_connected_components(taxonomy)
        if len(component) > 1 or any(taxonomy.has_edge(node, node) for node in component)
    ]

    removed = []
    for component_index, component in enumerate(cyclic_components, start=1):
        for child, parent in sorted(
            (edge for edge in edges if edge[0] in component and edge[1] in component),
            key=lambda edge: (str(edge[0]), str(edge[1])),
        ):
            graph.remove((child, RDFS.subClassOf, parent))
            removed.append(
                {
                    "scc_id": component_index,
                    "scc_size": len(component),
                    "child": str(child),
                    "child_label": display_label(graph, child),
                    "parent": str(parent),
                    "parent_label": display_label(graph, parent),
                    "action": "quarantined_from_stable_taxonomy",
                    "reason": (
                        "The asserted edge belongs to a cyclic subclass component. "
                        "No direction was selected automatically; the edge requires expert review."
                    ),
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    graph.serialize(destination=args.output, format="turtle")
    with args.report.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(removed[0]) if removed else ["scc_id", "child", "parent"])
        writer.writeheader()
        writer.writerows(removed)
    args.summary.write_text(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(args.output),
                "cycle_resolution_policy": (
                    "Quarantine every asserted rdfs:subClassOf edge whose endpoints belong to the same cyclic SCC; "
                    "do not invent a preferred direction."
                ),
                "cyclic_components_before": len(cyclic_components),
                "cycle_internal_edges_quarantined": len(removed),
                "original_named_subclass_edges": len(edges),
                "retained_named_subclass_edges": len(edges) - len(removed),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"cyclic_components": len(cyclic_components), "edges_quarantined": len(removed)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
