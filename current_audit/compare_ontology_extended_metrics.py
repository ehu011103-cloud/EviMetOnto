from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

import networkx as nx
from rdflib import Graph, Literal, OWL, RDF, RDFS, SKOS, URIRef


ROOT = Path(__file__).resolve().parent
DEFINITION_PREDICATES = {
    RDFS.comment,
    SKOS.definition,
    URIRef("http://purl.obolibrary.org/obo/IAO_0000115"),
}
SYNONYM_PREDICATES = {
    SKOS.altLabel,
    URIRef("http://www.geneontology.org/formats/oboInOwl#hasExactSynonym"),
    URIRef("http://www.geneontology.org/formats/oboInOwl#hasNarrowSynonym"),
    URIRef("http://www.geneontology.org/formats/oboInOwl#hasBroadSynonym"),
    URIRef("http://www.geneontology.org/formats/oboInOwl#hasRelatedSynonym"),
}
PROPERTY_CHARACTERISTICS = {
    OWL.FunctionalProperty,
    OWL.InverseFunctionalProperty,
    OWL.TransitiveProperty,
    OWL.SymmetricProperty,
    OWL.AsymmetricProperty,
    OWL.ReflexiveProperty,
    OWL.IrreflexiveProperty,
}


def normalized_entropy(counts: list[int]) -> float | str:
    """Return Shannon entropy normalized to [0, 1] for a non-empty distribution."""
    total = sum(counts)
    if total == 0 or len(counts) < 2:
        return 0.0 if total else "NA"
    entropy = -sum((count / total) * math.log(count / total) for count in counts if count)
    return entropy / math.log(len(counts))


def resolve_files(spec: dict) -> list[Path]:
    files = [ROOT / path for path in spec.get("paths", [])]
    if "glob" in spec:
        files.extend(Path(path) for path in glob.glob(str(ROOT / spec["glob"])))
    excluded = {
        Path(path).resolve()
        for pattern in spec.get("exclude", [])
        for path in glob.glob(str(ROOT / pattern))
    }
    return sorted({path.resolve() for path in files if path.exists()} - excluded)


def parse_files(files: list[Path]) -> tuple[Graph, list[str]]:
    graph = Graph()
    failures = []
    for path in files:
        try:
            graph.parse(path)
        except Exception as exc:
            failures.append(f"{path}: {exc}")
    return graph, failures


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


def literal_annotated(graph: Graph, entity: URIRef, predicates: set[URIRef]) -> bool:
    return any(
        isinstance(value, Literal) and str(value).strip()
        for predicate in predicates
        for value in graph.objects(entity, predicate)
    )


def normalize_label(value: str) -> str:
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value)
    value = value.replace("_", " ").replace("-", " ")
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def add_rate(row: dict, prefix: str, numerator: int, denominator: int) -> None:
    row[f"{prefix}_count"] = numerator
    row[f"{prefix}_coverage"] = numerator / denominator if denominator else "NA"


def audit(name: str, spec: dict) -> tuple[dict, list[str], list[dict]]:
    files = resolve_files(spec)
    graph, failures = parse_files(files)
    classes = class_entities(graph)
    properties = {
        subject
        for subject in graph.subjects(RDF.type, OWL.ObjectProperty)
        if isinstance(subject, URIRef)
    }

    edges = {
        (child, parent)
        for child, parent in graph.subject_objects(RDFS.subClassOf)
        if isinstance(child, URIRef) and isinstance(parent, URIRef)
    }
    taxonomy = nx.DiGraph()
    taxonomy.add_nodes_from(classes)
    taxonomy.add_edges_from(edges)
    components = list(nx.strongly_connected_components(taxonomy))
    cyclic_components = [
        component
        for component in components
        if len(component) > 1 or any(taxonomy.has_edge(node, node) for node in component)
    ]
    condensed = nx.condensation(taxonomy, scc=components)
    max_taxonomy_depth = (
        nx.dag_longest_path_length(condensed)
        if condensed.number_of_nodes() and nx.is_directed_acyclic_graph(condensed)
        else 0
    )
    parents: dict[URIRef, set[URIRef]] = defaultdict(set)
    children: dict[URIRef, set[URIRef]] = defaultdict(set)
    for child, parent in edges:
        parents[child].add(parent)
        children[parent].add(child)
    multiply_inherited = sum(len(parents[concept]) > 1 for concept in classes)
    multi_parent_counts = [len(parents[concept]) for concept in classes if len(parents[concept]) > 1]

    # Taxonomy edges point from a child to its parent. Reverse the condensed DAG
    # to calculate root-to-leaf depths without being affected by cycles.
    taxonomy_flow = condensed.reverse(copy=False)
    longest_depth = {node: 0 for node in taxonomy_flow.nodes}
    for node in nx.topological_sort(taxonomy_flow):
        for child in taxonomy_flow.successors(node):
            longest_depth[child] = max(longest_depth[child], longest_depth[node] + 1)
    component_map = condensed.graph["mapping"]
    class_depths = [longest_depth[component_map[concept]] for concept in classes]
    leaf_components = {component_map[concept] for concept in classes if not children[concept]}
    leaf_depths = [longest_depth[component] for component in leaf_components]
    weak_components = list(nx.weakly_connected_components(taxonomy))
    largest_weak_component = max((len(component) for component in weak_components), default=0)
    isolated_classes = sum(not parents[concept] and not children[concept] for concept in classes)
    non_leaf_classes = sum(bool(children[concept]) for concept in classes)

    # A direct subclass edge is redundant when the same subsumption is already
    # implied through another asserted path. Cyclic SCC-internal edges are kept
    # outside this calculation and reported separately by the cycle metrics.
    reduction_input = condensed.copy()
    try:
        reduced_taxonomy = nx.transitive_reduction(reduction_input)
        redundant_component_edges = set(reduction_input.edges) - set(reduced_taxonomy.edges)
        redundant_named_edges = sum(
            1
            for child, parent in edges
            if (component_map[child], component_map[parent]) in redundant_component_edges
        )
    except Exception:
        redundant_named_edges = "NA"

    restriction_property_occurrences = [
        (restriction, prop)
        for restriction in graph.subjects(RDF.type, OWL.Restriction)
        for prop in graph.objects(restriction, OWL.onProperty)
        if isinstance(prop, URIRef)
    ]
    relation_property_vocabulary = properties | {
        prop for _, prop in restriction_property_occurrences
    }
    class_relation_assertions = [
        (subject, predicate, obj)
        for predicate in relation_property_vocabulary
        for subject, obj in graph.subject_objects(predicate)
        if isinstance(subject, URIRef) and isinstance(obj, URIRef) and subject in classes and obj in classes
    ]
    direct_relation_property_counts: dict[URIRef, int] = defaultdict(int)
    for _, predicate, _ in class_relation_assertions:
        direct_relation_property_counts[predicate] += 1
    direct_used_relation_properties = set(direct_relation_property_counts)
    relation_property_counts: dict[URIRef, int] = defaultdict(int, direct_relation_property_counts)
    for _, predicate in restriction_property_occurrences:
        relation_property_counts[predicate] += 1
    used_relation_properties = set(relation_property_counts)
    relation_axiom_occurrences = len(class_relation_assertions) + len(restriction_property_occurrences)

    equivalent_pairs = {
        (left, right)
        for left, right in graph.subject_objects(OWL.equivalentClass)
        if isinstance(left, URIRef) and isinstance(right, URIRef)
    }
    disjoint_pairs = {
        (left, right)
        for left, right in graph.subject_objects(OWL.disjointWith)
        if isinstance(left, URIRef) and isinstance(right, URIRef)
    }
    axiom_participants = {
        node for pair in equivalent_pairs | disjoint_pairs for node in pair if node in classes
    }
    expression_classes = {
        subject
        for predicate in (RDFS.subClassOf, OWL.equivalentClass)
        for subject, expression in graph.subject_objects(predicate)
        if isinstance(subject, URIRef) and subject in classes and not isinstance(expression, URIRef)
    }
    restriction_nodes = {
        node for node in graph.subjects(RDF.type, OWL.Restriction)
    }

    domain_range_pairs = {
        prop
        for prop in properties
        if any(graph.objects(prop, RDFS.domain)) and any(graph.objects(prop, RDFS.range))
    }
    characterized_properties = {
        prop
        for prop in properties
        if any((prop, RDF.type, characteristic) in graph for characteristic in PROPERTY_CHARACTERISTICS)
    }
    declared_classes = {
        subject
        for class_type in (OWL.Class, RDFS.Class)
        for subject in graph.subjects(RDF.type, class_type)
        if isinstance(subject, URIRef)
    }
    classes_in_taxonomy = {node for edge in edges for node in edge}
    deprecated_classes = {
        subject
        for subject, value in graph.subject_objects(OWL.deprecated)
        if subject in classes and str(value).lower() in {"true", "1"}
    }
    normalized_labels: dict[str, set[URIRef]] = defaultdict(set)
    original_labels: dict[tuple[str, URIRef], set[str]] = defaultdict(set)
    for concept in classes:
        for predicate in (RDFS.label, SKOS.prefLabel):
            for value in graph.objects(concept, predicate):
                if isinstance(value, Literal):
                    normalized = normalize_label(str(value))
                    if normalized:
                        normalized_labels[normalized].add(concept)
                        original_labels[(normalized, concept)].add(str(value))
    collision_sets = [
        (label, concepts)
        for label, concepts in normalized_labels.items()
        if len(concepts) > 1
    ]
    collision_entities = (
        set().union(*(concepts for _, concepts in collision_sets))
        if collision_sets
        else set()
    )
    collision_review = []
    same_parent_collision_pairs = 0
    for normalized, concepts in collision_sets:
        ordered = sorted(concepts, key=str)
        for index, left in enumerate(ordered):
            for right in ordered[index + 1 :]:
                same_parent_set = parents[left] == parents[right]
                same_parent_collision_pairs += same_parent_set
                collision_review.append(
                    {
                        "ontology": name,
                        "normalized_label": normalized,
                        "class_uri_left": str(left),
                        "label_left": " | ".join(sorted(original_labels[(normalized, left)])),
                        "direct_parents_left": " | ".join(sorted(map(str, parents[left]))),
                        "class_uri_right": str(right),
                        "label_right": " | ".join(sorted(original_labels[(normalized, right)])),
                        "direct_parents_right": " | ".join(sorted(map(str, parents[right]))),
                        "same_direct_parent_set": int(same_parent_set),
                    }
                )

    ontology_nodes = set(graph.subjects(RDF.type, OWL.Ontology))
    metadata_fields = {
        "ontology_title": {URIRef("http://purl.org/dc/terms/title"), URIRef("http://purl.org/dc/elements/1.1/title"), RDFS.label},
        "ontology_version": {OWL.versionInfo, OWL.versionIRI},
        "ontology_license": {URIRef("http://purl.org/dc/terms/license"), URIRef("http://creativecommons.org/ns#license")},
        "ontology_creator": {URIRef("http://purl.org/dc/terms/creator"), URIRef("http://purl.org/dc/elements/1.1/creator"), URIRef("http://www.w3.org/2000/01/rdf-schema#creator")},
    }
    metadata = {}
    for field, predicates in metadata_fields.items():
        metadata[field] = sum(
            1
            for node in ontology_nodes
            if any(True for predicate in predicates for _ in graph.objects(node, predicate))
        )

    row = {
        "ontology": name,
        "scope": spec.get("scope", ""),
        "source_files": len(files),
        "triples": len(graph),
        "classes": len(classes),
        "declared_classes": len(declared_classes),
        "named_subclass_edges": len(edges),
        "class_hierarchy_coverage": len(classes_in_taxonomy) / len(classes) if classes else 0,
        "root_classes": sum(not parents[concept] for concept in classes),
        "leaf_classes": sum(not children[concept] for concept in classes),
        "mean_direct_parents_per_class": len(edges) / len(classes) if classes else 0,
        "mean_direct_children_per_class": len(edges) / len(classes) if classes else 0,
        "multiply_inherited_classes": multiply_inherited,
        "multiply_inheritance_rate": multiply_inherited / len(classes) if classes else 0,
        "mean_direct_parents_among_multiply_inherited_classes": (
            statistics.mean(multi_parent_counts) if multi_parent_counts else 0
        ),
        "max_taxonomy_depth_scc_collapsed": max_taxonomy_depth,
        "mean_class_depth_scc_collapsed": statistics.mean(class_depths) if class_depths else 0,
        "mean_leaf_depth_scc_collapsed": statistics.mean(leaf_depths) if leaf_depths else 0,
        "median_leaf_depth_scc_collapsed": statistics.median(leaf_depths) if leaf_depths else 0,
        "mean_direct_children_per_non_leaf_class": (
            len(edges) / non_leaf_classes if non_leaf_classes else 0
        ),
        "weakly_connected_taxonomy_components": len(weak_components),
        "largest_weak_component_class_coverage": (
            largest_weak_component / len(classes) if classes else 0
        ),
        "isolated_classes": isolated_classes,
        "strongly_connected_components": len(components),
        "cyclic_components": len(cyclic_components),
        "classes_in_cycles": len(set().union(*cyclic_components)) if cyclic_components else 0,
        "redundant_named_subclass_edges": redundant_named_edges,
        "redundant_named_subclass_edge_rate": (
            redundant_named_edges / len(edges)
            if isinstance(redundant_named_edges, int) and edges
            else "NA"
        ),
        "object_properties": len(properties),
        "relation_property_vocabulary_size_including_referenced_properties": len(relation_property_vocabulary),
        "class_to_class_object_property_assertions": len(class_relation_assertions),
        "owl_object_property_restriction_occurrences": len(restriction_property_occurrences),
        "relation_axiom_occurrences_including_restrictions": relation_axiom_occurrences,
        "relation_axiom_properties_used": len(used_relation_properties),
        "object_properties_used_between_classes": len(direct_used_relation_properties),
        "object_property_use_coverage_between_classes": (
            len(direct_used_relation_properties) / len(relation_property_vocabulary)
            if relation_property_vocabulary
            else "NA"
        ),
        "assertion_level_relation_richness": (
            len(class_relation_assertions) / (len(class_relation_assertions) + len(edges))
            if class_relation_assertions or edges
            else "NA"
        ),
        "used_relation_property_distribution_entropy": normalized_entropy(
            list(direct_relation_property_counts.values())
        ),
        "relation_axiom_property_use_coverage": (
            len(used_relation_properties) / len(relation_property_vocabulary)
            if relation_property_vocabulary
            else "NA"
        ),
        "relation_axiom_richness_including_restrictions": (
            relation_axiom_occurrences / (relation_axiom_occurrences + len(edges))
            if relation_axiom_occurrences or edges
            else "NA"
        ),
        "relation_axiom_property_distribution_entropy": normalized_entropy(
            list(relation_property_counts.values())
        ),
        "owl_class_restriction_nodes": len(restriction_nodes),
        "classes_with_anonymous_class_expression": len(expression_classes),
        "anonymous_expression_class_coverage": len(expression_classes) / len(classes) if classes else 0,
        "equivalent_class_axioms_named_pairs": len(equivalent_pairs),
        "disjoint_class_axioms_named_pairs": len(disjoint_pairs),
        "classes_in_named_equivalence_or_disjointness": len(axiom_participants),
        "deprecated_classes": len(deprecated_classes),
        "normalized_label_collision_values": len(collision_sets),
        "classes_in_normalized_label_collisions": len(collision_entities),
        "normalized_label_collision_class_rate": len(collision_entities) / len(classes) if classes else 0,
        "normalized_label_collision_entity_pairs": sum(
            len(concepts) * (len(concepts) - 1) // 2
            for _, concepts in collision_sets
        ),
        "same_parent_set_collision_pairs": same_parent_collision_pairs,
        "object_properties_with_domain_and_range": len(domain_range_pairs),
        "object_property_domain_range_coverage": len(domain_range_pairs) / len(properties) if properties else "NA",
        "object_properties_with_formal_characteristic": len(characterized_properties),
        "object_property_characteristic_coverage": len(characterized_properties) / len(properties) if properties else "NA",
        "ontology_resources": len(ontology_nodes),
        **metadata,
    }
    for key, predicates in (
        ("class_explicit_label", {RDFS.label, SKOS.prefLabel}),
        ("class_definition_standard_plus_iao", DEFINITION_PREDICATES),
        ("class_synonym_skos_plus_obo", SYNONYM_PREDICATES),
        ("object_property_explicit_label", {RDFS.label, SKOS.prefLabel}),
        ("object_property_definition_standard_plus_iao", DEFINITION_PREDICATES),
        ("object_property_synonym_skos_plus_obo", SYNONYM_PREDICATES),
    ):
        entities = classes if key.startswith("class_") else properties
        count = sum(literal_annotated(graph, entity, predicates) for entity in entities)
        add_rate(row, key, count, len(entities))

    return row, failures, collision_review


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["ontology", "failure"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "ontology_manifest_extended.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "audit_20261001")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = []
    failures = []
    collision_rows = []
    for name, spec in manifest.items():
        row, parse_failures, collisions = audit(name, spec)
        rows.append(row)
        collision_rows.extend(collisions)
        failures.extend({"ontology": name, "failure": failure} for failure in parse_failures)
    write_csv(args.output / "ontology_quality_extra_metrics.csv", rows)
    write_csv(args.output / "extra_metrics_parse_failures.csv", failures)
    write_csv(args.output / "normalized_label_collision_review.csv", collision_rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
