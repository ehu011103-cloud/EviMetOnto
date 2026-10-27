"""Exercise concept-graph inputs and outputs with invented, API-free toy data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS


TOY_TURTLE = """
@prefix ex: <https://example.org/evimetonto-demo/> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
ex:Mineral a owl:Class .
ex:SulfideMineral a owl:Class ; rdfs:subClassOf ex:Mineral .
ex:Pyrite a owl:Class ; rdfs:subClassOf ex:SulfideMineral .
ex:Alteration a owl:Class .
ex:HydrothermalAlteration a owl:Class ; rdfs:subClassOf ex:Alteration .
ex:Silicification a owl:Class ; rdfs:subClassOf ex:HydrothermalAlteration .
ex:associatedWith a owl:ObjectProperty .
ex:Pyrite ex:associatedWith ex:HydrothermalAlteration .
"""


def demonstrate() -> dict:
    graph = Graph().parse(data=TOY_TURTLE, format="turtle")
    named = {s for s in graph.subjects(RDF.type, OWL.Class) if isinstance(s, URIRef)}
    hierarchy = nx.DiGraph()
    hierarchy.add_nodes_from(named)
    hierarchy.add_edges_from((s, o) for s, o in graph.subject_objects(RDFS.subClassOf) if s in named and o in named)
    isolated = len(list(nx.isolates(hierarchy)))
    largest = max(map(len, nx.weakly_connected_components(hierarchy)))
    measured = {
        "classes": len(named), "named_subclass_edges": hierarchy.number_of_edges(),
        "hierarchy_coverage_percent": 100 * (len(named) - isolated) / len(named),
        "isolated_classes": isolated,
        "largest_component_coverage_percent": 100 * largest / len(named),
        "max_depth": nx.dag_longest_path_length(hierarchy),
    }
    expected = {"classes": 6, "named_subclass_edges": 4, "hierarchy_coverage_percent": 100.0,
                "isolated_classes": 0, "largest_component_coverage_percent": 50.0, "max_depth": 2}
    assert measured == expected, (measured, expected)
    ex = Namespace("https://example.org/evimetonto-demo/")
    # Edges point from child to parent, so ancestors are more specific concepts.
    expanded = {ex.HydrothermalAlteration} | nx.ancestors(hierarchy, ex.HydrothermalAlteration)
    assert expanded == {ex.HydrothermalAlteration, ex.Silicification}
    return {
        "status": "pass", "input": "six invented classes embedded in this script",
        "structural_descriptors": measured,
        "query_concept": "HydrothermalAlteration",
        "query_with_descendant_expansion": sorted(str(c).rsplit("/", 1)[-1] for c in expanded),
        "scope": "Synthetic input/output example only; not a manuscript experiment, learned model, geological validation, or retrieval benchmark.",
        "model_api_calls": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON report destination.")
    args = parser.parse_args()
    result = json.dumps(demonstrate(), ensure_ascii=True, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result + "\n", encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
