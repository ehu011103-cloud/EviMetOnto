from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import render_paper_tables as base


OUT = Path("papers/figures")
OUT.mkdir(parents=True, exist_ok=True)

base.BLUE = (0, 0, 0)
base.LIGHT_BLUE = (205, 205, 205)

FONT_REG = r"C:\Windows\Fonts\times.ttf"
FONT_BOLD = r"C:\Windows\Fonts\timesbd.ttf"

caption_font = ImageFont.truetype(FONT_BOLD, 62)
section_font = ImageFont.truetype(FONT_BOLD, 52)
header_font = ImageFont.truetype(FONT_REG, 48)
body_font = ImageFont.truetype(FONT_REG, 47)
small_body_font = ImageFont.truetype(FONT_REG, 43)
bold_body_font = ImageFont.truetype(FONT_BOLD, 47)
small_bold_body_font = ImageFont.truetype(FONT_BOLD, 43)


def make_canvas(width, height):
    return Image.new("RGB", (width, height), (255, 255, 255))


def finish(img, y, width, filename):
    base.save(img.crop((0, 0, width, y + 40)), OUT / filename)


def render_table_4_1():
    width, x = 3000, 95
    table_w = width - 2 * x
    img = make_canvas(width, 1800)
    d = ImageDraw.Draw(img)
    y = 50
    d.text((x, y), "Table 1. Construction and validation matrix.", font=caption_font, fill=base.BLACK)
    y += 82
    d.text((x, y), "A. Construction scale and deduplication", font=section_font, fill=base.BLACK)
    y += 52
    cols = [("Pipeline", 760), ("Documents", 420), ("Raw output", 780), ("Unique output", 850)]
    rows = [
        ["Taxonomy candidate generation", "5,723/5,723\ncomplete", "318,026 parent-child mentions", "54,843 isA\n33,912 nodes"],
        ["R-H-F relation extraction", "5,723/5,723\ncomplete", "70,582 relation rows", "68,367 normalized triples"],
    ]
    y = base.draw_simple_table(d, x, y, cols, rows, table_w, header_font, body_font, row_lines=True)
    y += 70
    d.text((x, y), "B. Logical validation", font=section_font, fill=base.BLACK)
    y += 52
    cols = [("Validation setting", 820), ("Check set", 390), ("VALID", 350), ("INVALID", 350), ("UNCERTAIN", 480), ("VALID rate", 420)]
    rows = [
        ["Hierarchy isA candidates\nindependent counterexample validation", "54,108", "35,058", "18,667", "383", "64.8%"],
        ["Evidence-anchored relations\nevidence-constrained bidirectional validation", "70,582", "61,468", "9,103", "11", "87.09%"],
    ]
    y = base.draw_simple_table(d, x, y, cols, rows, table_w, header_font, small_body_font, row_lines=True)
    finish(img, y, width, "table_4_1_validation_matrix_en.png")


def render_table_4_2():
    width, x = 3000, 95
    table_w = width - 2 * x
    img = make_canvas(width, 1000)
    d = ImageDraw.Draw(img)
    y = 50
    d.text((x, y), "Table 2. Evaluation protocols and reported metrics.", font=caption_font, fill=base.BLACK)
    y += 82
    cols = [("Evaluation stage", 620), ("Reported metrics", 820), ("Definition", table_w - 1440)]
    rows = [
        ["Main and baseline comparison", "TP, FP, FN; P, R, F1", "Exact document, predicate, and canonical endpoint matching is applied."],
        ["External complete-graph comparison", "Hierarchy F1; 4D F1; overall P, R, F1", "Only methods that generate both hierarchy and non-hierarchical relations are included."],
        ["Cascaded extraction", "Hierarchy, aligned-relation, overall F1", "Compares staged and joint extraction using three Exact F1 measures."],
        ["Logical validation", "VALID / UNCERTAIN / INVALID; P, R, F1", "Reports discrete decision counts and diagnostic quality of the default accepted layer."],
        ["Alignment ablation", "Precision, Recall, F1", "Exact evaluation where the reference canonical concept exists in the taxonomy inventory."],
    ]
    y = base.draw_simple_table(d, x, y, cols, rows, table_w, header_font, small_body_font)
    finish(img, y, width, "table_4_2_evaluation_metrics_en.png")


def render_table_4_3():
    width, x = 3000, 95
    table_w = width - 2 * x
    img = make_canvas(width, 3500)
    d = ImageDraw.Draw(img)
    # Table 3 is a primary-result figure. Its type is intentionally larger
    # than the auxiliary count tables after final journal-column scaling.
    table3_caption = ImageFont.truetype(FONT_BOLD, 68)
    table3_section = ImageFont.truetype(FONT_BOLD, 60)
    table3_header = ImageFont.truetype(FONT_REG, 56)
    table3_body = ImageFont.truetype(FONT_REG, 54)
    table3_bold = ImageFont.truetype(FONT_BOLD, 54)
    y = 50
    d.text((x, y), "Table 3. Main diagnostic results and external baseline evaluations.", font=table3_caption, fill=base.BLACK)
    y += 82
    d.text((x, y), "A. Proposed method (Exact canonical endpoint matching)", font=table3_section, fill=base.BLACK)
    y += 56
    cols = [("Category", 760), ("TP", 260), ("FP", 260), ("FN", 260), ("Precision", 420), ("Recall", 400), ("F1", table_w - 2360)]
    rows = [
        ["Hierarchy relations", "535", "123", "145", "0.8131", "0.7868", "0.7997"],
        ["Aligned non-hierarchical relations", "948", "181", "67", "0.8397", "0.9340", "0.8843"],
        ["Overall ontology graph", "1483", "304", "212", "0.8299", "0.8749", "0.8518"],
    ]
    y = base.draw_three_line_table(d, x, y, cols, rows, table_w, table3_header, table3_body, table3_bold, bold_rows=(2,), separator_before=(2,))
    y += 66
    d.text((x, y), "B. External hierarchy baselines (Exact canonical endpoint matching)", font=table3_section, fill=base.BLACK)
    y += 56
    cols = [("Method family", 740), ("External baseline", 760), ("Hierarchy P", 360), ("Hierarchy R", 360), ("Hierarchy F1", table_w - 2220)]
    rows = [
        ["Generic ontology learning", "NeOn-GPT", "0.250", "0.225", "0.237"],
        ["Generic ontology learning", "OntoGen", "0.009", "0.015", "0.011"],
        ["CQ-driven ontology generation", "Ontogenia", "0.147", "0.141", "0.144"],
        ["CQ-driven ontology generation", "Memoryless-CQ", "0.163", "0.121", "0.139"],
        ["Task-specific ontology learning", "LLMs4OL Task B+C", "0.252", "0.329", "0.285"],
        ["Progressive dual-LLM prompting", "POP-Duo", "0.191", "0.199", "0.195"],
        ["Planning-first multi-agent", "Planning-First", "0.050", "0.071", "0.058"],
        ["Taxonomy-specific multi-agent", "CHisAgent (hierarchy only)", "0.056", "0.213", "0.088"],
    ]
    y = base.draw_three_line_table(d, x, y, cols, rows, table_w, table3_header, table3_body, table3_bold, bold_rows=())
    y += 66
    d.text((x, y), "C. External complete-graph baselines (Exact canonical endpoint matching)", font=table3_section, fill=base.BLACK)
    y += 56
    cols = [("External baseline", 840), ("Hierarchy F1", 380), ("4D F1", 340), ("Overall P", 380), ("Overall R", 380), ("Overall F1", table_w - 2320)]
    rows = [
        ["NeOn-GPT", "0.237", "0.084", "0.179", "0.131", "0.151"],
        ["LLMs4OL Task B+C", "0.285", "0.103", "0.163", "0.198", "0.179"],
    ]
    y = base.draw_three_line_table(d, x, y, cols, rows, table_w, table3_header, table3_body, table3_bold, bold_rows=())
    finish(img, y, width, "table_4_3_final50_metrics_en.png")


def render_table_4_2_report():
    """Report-version protocols; strict submission assets remain unchanged."""
    width, x = 3000, 95
    table_w = width - 2 * x
    img = make_canvas(width, 1000)
    d = ImageDraw.Draw(img)
    y = 50
    d.text((x, y), "Table 2. Evaluation protocols and reported metrics.", font=caption_font, fill=base.BLACK)
    y += 82
    cols = [("Evaluation stage", 620), ("Reported metrics", 820), ("Definition", table_w - 1440)]
    rows = [
        ["Main and baseline comparison", "TP, FP, FN; P, R, F1", "A fixed endpoint-compatibility rule is applied within the same document and predicate."],
        ["External complete-graph comparison", "Hierarchy F1; 4D F1; overall P, R, F1", "Only methods that generate both hierarchy and non-hierarchical relations are included."],
        ["Cascaded extraction", "Hierarchy, aligned-relation, overall F1", "Compares staged and joint extraction using three Exact F1 measures."],
        ["Logical validation", "VALID / UNCERTAIN / INVALID; P, R, F1", "Reports discrete decision counts and diagnostic quality of the default accepted layer."],
        ["Alignment ablation", "Precision, Recall, F1", "Exact evaluation where the reference canonical concept exists in the taxonomy inventory."],
    ]
    y = base.draw_simple_table(d, x, y, cols, rows, table_w, header_font, small_body_font)
    finish(img, y, width, "table_4_2_report_endpoint_compatibility_en.png")


def render_table_4_3_report():
    """Report-version main/baseline table under a shared compatibility protocol."""
    width, x = 3000, 95
    table_w = width - 2 * x
    img = make_canvas(width, 3500)
    d = ImageDraw.Draw(img)
    table3_caption = ImageFont.truetype(FONT_BOLD, 68)
    table3_section = ImageFont.truetype(FONT_BOLD, 60)
    table3_header = ImageFont.truetype(FONT_REG, 56)
    table3_body = ImageFont.truetype(FONT_REG, 54)
    table3_bold = ImageFont.truetype(FONT_BOLD, 54)
    y = 50
    d.text((x, y), "Table 3. Main diagnostic results and external baseline evaluations.", font=table3_caption, fill=base.BLACK)
    y += 82
    d.text((x, y), "A. Proposed method (fixed endpoint-compatibility protocol)", font=table3_section, fill=base.BLACK)
    y += 56
    cols = [("Category", 760), ("TP", 260), ("FP", 260), ("FN", 260), ("Precision", 420), ("Recall", 400), ("F1", table_w - 2360)]
    rows = [
        ["Hierarchy relations", "535", "123", "145", "0.8131", "0.7868", "0.7997"],
        ["Aligned non-hierarchical relations", "948", "181", "67", "0.8397", "0.9340", "0.8843"],
        ["Overall ontology graph", "1483", "304", "212", "0.8299", "0.8749", "0.8518"],
    ]
    y = base.draw_three_line_table(d, x, y, cols, rows, table_w, table3_header, table3_body, table3_bold, bold_rows=(2,), separator_before=(2,))
    y += 66
    d.text((x, y), "B. External hierarchy baselines (fixed endpoint-compatibility protocol)", font=table3_section, fill=base.BLACK)
    y += 56
    cols = [("Method family", 740), ("External baseline", 760), ("Hierarchy P", 360), ("Hierarchy R", 360), ("Hierarchy F1", table_w - 2220)]
    rows = [
        ["Generic ontology learning", "NeOn-GPT", "0.640", "0.879", "0.741"],
        ["Generic ontology learning", "OntoGen", "0.405", "1.000", "0.576"],
        ["CQ-driven ontology generation", "Ontogenia", "0.629", "0.921", "0.747"],
        ["CQ-driven ontology generation", "Memoryless-CQ", "0.685", "0.778", "0.728"],
        ["Task-specific ontology learning", "LLMs4OL Task B+C", "0.747", "0.974", "0.847"],
        ["Progressive dual-LLM prompting", "POP-Duo", "0.571", "0.906", "0.700"],
        ["Planning-first multi-agent", "Planning-First", "0.417", "0.906", "0.571"],
        ["Taxonomy-specific multi-agent", "CHisAgent (hierarchy only)", "0.262", "1.000", "0.415"],
    ]
    y = base.draw_three_line_table(d, x, y, cols, rows, table_w, table3_header, table3_body, table3_bold, bold_rows=())
    y += 66
    d.text((x, y), "C. External complete-graph baselines (fixed endpoint-compatibility protocol)", font=table3_section, fill=base.BLACK)
    y += 56
    cols = [("External baseline", 840), ("Hierarchy F1", 380), ("4D F1", 340), ("Overall P", 380), ("Overall R", 380), ("Overall F1", table_w - 2320)]
    rows = [
        ["NeOn-GPT", "0.806", "0.550", "0.783", "0.573", "0.662"],
        ["LLMs4OL Task B+C", "0.847", "0.607", "0.645", "0.783", "0.707"],
    ]
    y = base.draw_three_line_table(d, x, y, cols, rows, table_w, table3_header, table3_body, table3_bold, bold_rows=())
    finish(img, y, width, "table_4_3_report_endpoint_compatibility_en.png")
    finish(img, y, width, "table_4_3_final50_metrics_en.png")


def render_table_4_4():
    from render_table_4_4_without_validation_ablation import render_en

    render_en()
    return
    width, x = 3000, 95
    table_w = width - 2 * x
    img = make_canvas(width, 2200)
    d = ImageDraw.Draw(img)
    y = 50
    d.text((x, y), "Table 4-4. Ablation results.", font=caption_font, fill=base.BLACK)
    y += 82
    d.text((x, y), "A. Cascaded extraction module (unified V4 VALID+UNCERTAIN, Exact)", font=section_font, fill=base.BLACK)
    y += 52
    cols = [("Setting", 1000), ("Hierarchy F1", 600), ("Aligned relation F1", 720), ("Overall F1", table_w - 2320)]
    rows = [
        ["Full staged extraction", "0.8515", "0.8796", "0.8693"],
        ["w/o cascade: Joint single-call", "0.3613", "0.2141", "0.2654"],
    ]
    y = base.draw_three_line_table(d, x, y, cols, rows, table_w, header_font, body_font, bold_body_font, bold_rows=(0,))
    y += 60
    d.text((x, y), "B. Logical validation module (unified V4 VALID+UNCERTAIN, Exact)", font=section_font, fill=base.BLACK)
    y += 52
    cols = [("Setting", 1200), ("Precision", 530), ("Recall", 530), ("F1", table_w - 2260)]
    rows = [
        ["Full validation", "0.8355", "0.9059", "0.8693"],
        ["w/o counterexample + direction/bidirectional checks", "0.8285", "0.9527", "0.8863"],
    ]
    y = base.draw_three_line_table(d, x, y, cols, rows, table_w, header_font, body_font, bold_body_font, bold_rows=(0,))
    y += 60
    d.text((x, y), "C. Ontology alignment module (unified V4 VALID+UNCERTAIN, Exact)", font=section_font, fill=base.BLACK)
    y += 52
    cols = [("Setting", 1200), ("Precision", 530), ("Recall", 530), ("F1", table_w - 2260)]
    rows = [
        ["Full ontology alignment", "0.8355", "0.9059", "0.8693"],
        ["w/o alignment: raw mention endpoints", "0.2377", "0.2579", "0.2474"],
    ]
    y = base.draw_three_line_table(d, x, y, cols, rows, table_w, header_font, small_body_font, small_bold_body_font, bold_rows=(0,))
    finish(img, y, width, "table_4_4_ablation_en.png")


def render_table_a_1():
    width, x = 2800, 90
    table_w = width - 2 * x
    img = make_canvas(width, 1800)
    d = ImageDraw.Draw(img)
    y = 50
    d.text((x, y), "Appendix Table A1. Relation distribution and validation summary.", font=caption_font, fill=base.BLACK)
    y += 82
    cols = [("Relation", 610), ("Candidates", 330), ("Share", 300), ("VALID", 330), ("INVALID", 330), ("UNCERTAIN", 380), ("VALID rate", table_w - 2280)]
    rows = [
        ["ASSOCIATED_WITH", "11,317", "16.03%", "10,809", "505", "3", "95.51%"],
        ["HOSTED_IN", "11,277", "15.98%", "10,069", "1,208", "0", "89.29%"],
        ["CAUSES", "10,220", "14.48%", "9,237", "980", "3", "90.38%"],
        ["CONTROLS", "8,064", "11.43%", "5,857", "2,207", "0", "72.63%"],
        ["FORMED_DURING", "6,991", "9.90%", "6,108", "883", "0", "87.37%"],
        ["PRECEDES", "6,699", "9.49%", "5,174", "1,522", "3", "77.24%"],
        ["INDICATES", "4,016", "5.69%", "3,874", "142", "0", "96.46%"],
        ["OVERPRINTS", "3,258", "4.62%", "2,603", "655", "0", "79.90%"],
        ["DISTRIBUTED_ALONG", "2,940", "4.17%", "2,563", "377", "0", "87.18%"],
        ["INTRUDES_INTO", "2,153", "3.05%", "1,996", "157", "0", "92.71%"],
        ["SYNCHRONOUS_WITH", "2,104", "2.98%", "1,889", "213", "2", "89.78%"],
        ["ZONES_WITH", "1,543", "2.19%", "1,289", "254", "0", "83.54%"],
    ]
    y = base.draw_simple_table(d, x, y, cols, rows, table_w, header_font, body_font)
    finish(img, y, width, "table_a_1_relation_validation_summary_en.png")


def render_table_a_2():
    width, x = 2800, 90
    table_w = width - 2 * x
    img = make_canvas(width, 1900)
    d = ImageDraw.Draw(img)
    y = 50
    d.text((x, y), "Appendix Table A1. Definitions of the 12 relation types.", font=caption_font, fill=base.BLACK)
    y += 82
    cols = [("Group", 520), ("Relation", 620), ("Definition", table_w - 1140)]
    rows = [
        ["Core logical", "CAUSES", "A geological process produces another process or outcome."],
        ["Core logical", "CONTROLS", "A structure, lithology, or condition controls mineralization or ore distribution."],
        ["Core logical", "INDICATES", "A geological phenomenon indicates a target or mineralization center."],
        ["Core logical", "ASSOCIATED_WITH", "Two concepts are geologically associated or co-occurring."],
        ["Spatial-topological", "HOSTED_IN", "An ore body, mineralization, or mineral occurs within a host rock or body."],
        ["Spatial-topological", "INTRUDES_INTO", "An intrusive body intrudes or cuts another geological body."],
        ["Spatial-topological", "ZONES_WITH", "Alteration or mineralization exhibits spatial zoning."],
        ["Spatial-topological", "DISTRIBUTED_ALONG", "Ore bodies, veins, or anomalies are distributed along a structure."],
        ["Temporal-evolutionary", "PRECEDES", "One event occurs before another event."],
        ["Temporal-evolutionary", "OVERPRINTS", "A later event overprints or modifies an earlier event."],
        ["Temporal-evolutionary", "SYNCHRONOUS_WITH", "Two events occur contemporaneously."],
        ["Temporal-evolutionary", "FORMED_DURING", "A concept forms during a specified age or geological event."],
    ]
    y = base.draw_simple_table(d, x, y, cols, rows, table_w, header_font, small_body_font)
    finish(img, y, width, "table_a_2_relation_schema_definitions_en.png")


def render_table_a_3():
    width, x = 3000, 95
    table_w = width - 2 * x
    img = make_canvas(width, 1100)
    d = ImageDraw.Draw(img)
    y = 50
    d.text((x, y), "Appendix Table A-3. Independent five-document pilot (Exact canonical endpoints).", font=caption_font, fill=base.BLACK)
    y += 82
    cols = [("Method", 800), ("Hierarchy F1", 400), ("Relation F1", 400), ("Overall P", 400), ("Overall R", 400), ("Overall F1", 410)]
    rows = [
        ["Ours (Hard-gate V4, VALID)", "0.4110", "0.4270", "0.4595", "0.3864", "0.4198"],
        ["Joint single-call", "0.1538", "0.1656", "0.1250", "0.2273", "0.1613"],
        ["OntoGen-adapted", "0.0122", "0.0529", "0.0226", "0.0682", "0.0340"],
        ["CHisAgent-adapted", "0.0857", "-", "-", "-", "-"],
    ]
    y = base.draw_simple_table(d, x, y, cols, rows, table_w, header_font, body_font, row_lines=True)
    finish(img, y, width, "table_a_3_original5_baselines_en.png")


def render_table_a_4():
    width, x = 3000, 95
    table_w = width - 2 * x
    img = make_canvas(width, 1800)
    d = ImageDraw.Draw(img)
    y = 50
    d.text((x, y), "Appendix Table A-4. Complete adapted literature baselines.", font=caption_font, fill=base.BLACK)
    y += 82
    cols = [("Method", 760), ("Method family", 860), ("Exact overall F1", 430), ("Relaxed overall F1", 470), ("Note", 290)]
    rows = [
        ["POP-LLM-Duo", "Progressive dual-agent", "0.1474", "0.2669", "overall"],
        ["NeOn-GPT", "Ontology engineering pipeline", "0.1512", "0.2008", "overall"],
        ["OntoGen", "Incremental scientific ontology", "0.0401", "0.1423", "overall"],
        ["Memoryless CQbyCQ", "CQ-driven independent modules", "0.1051", "0.1345", "overall"],
        ["Ontogenia", "CQ-driven incremental ontology", "0.1076", "0.1218", "overall"],
        ["Planning-First*", "Planning-first multi-agent", "0.0468", "0.0976", "upper bound"],
        ["CHisAgent", "Multi-agent taxonomy", "0.0884", "0.1606", "hierarchy only"],
    ]
    y = base.draw_simple_table(d, x, y, cols, rows, table_w, header_font, small_body_font, row_lines=True)
    finish(img, y, width, "table_a_4_complete_baselines_en.png")


def main():
    render_table_4_1()
    render_table_4_2()
    render_table_4_2_report()
    render_table_4_3()
    render_table_4_3_report()
    render_table_4_4()
    render_table_a_1()
    render_table_a_2()
    for name in sorted(OUT.glob("*_en.png")):
        print(name.name, Image.open(name).size)


if __name__ == "__main__":
    main()
