# -*- coding: utf-8 -*-
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUT = Path("papers/figures")
OUT.mkdir(parents=True, exist_ok=True)

FONT_REG = r"C:\Windows\Fonts\simsun.ttc"
FONT_BOLD = r"C:\Windows\Fonts\msyhbd.ttc"

BLUE = (44, 58, 170)
LIGHT_BLUE = (215, 220, 248)
BLACK = (20, 20, 20)
WHITE = (255, 255, 255)


def text_w(draw, text, font):
    b = draw.textbbox((0, 0), text, font=font)
    return b[2] - b[0]


def wrap_text(draw, text, font, max_w):
    text = str(text)
    if text == "":
        return [""]
    out = []
    for para in text.split("\n"):
        words = para.split(" ")
        if len(words) > 1:
            line = ""
            for word in words:
                trial = word if not line else line + " " + word
                if text_w(draw, trial, font) <= max_w:
                    line = trial
                    continue
                if line:
                    out.append(line)
                if text_w(draw, word, font) <= max_w:
                    line = word
                    continue
                buf = ""
                for ch in word:
                    trial2 = buf + ch
                    if text_w(draw, trial2, font) <= max_w:
                        buf = trial2
                    else:
                        if buf:
                            out.append(buf)
                        buf = ch
                line = buf
            out.append(line)
        else:
            buf = ""
            for ch in para:
                trial = buf + ch
                if text_w(draw, trial, font) <= max_w:
                    buf = trial
                else:
                    if buf:
                        out.append(buf)
                    buf = ch
            out.append(buf)
    return out


def draw_text_lines(draw, xy, lines, font, fill, line_gap=6, align="left", box_w=None):
    x, y = xy
    line_h = font.size + 10
    for line in lines:
        tx = x
        if align == "center" and box_w is not None:
            tx = x + max(0, (box_w - text_w(draw, line, font)) / 2)
        elif align == "right" and box_w is not None:
            tx = x + max(0, box_w - text_w(draw, line, font))
        draw.text((tx, y), line, font=font, fill=fill)
        y += line_h + line_gap


def table_height(draw, cols, rows, fonts, pad_y, pad_x):
    header_font, body_font = fonts
    header_lines = [
        wrap_text(draw, label, header_font, col_w - 2 * pad_x)
        for label, col_w in cols
    ]
    header_h = max(len(lines) for lines in header_lines) * (header_font.size + 12) + 2 * pad_y
    row_heights = []
    for row in rows:
        cell_lines = [
            wrap_text(draw, val, body_font, col_w - 2 * pad_x)
            for val, (_, col_w) in zip(row, cols)
        ]
        row_heights.append(max(len(lines) for lines in cell_lines) * (body_font.size + 10) + 2 * pad_y)
    return header_h + sum(row_heights), header_h, row_heights


def draw_simple_table(
    draw,
    x,
    y,
    cols,
    rows,
    width,
    header_font,
    body_font,
    pad_x=18,
    pad_y=12,
    row_lines=False,
):
    draw.line((x, y, x + width, y), fill=BLUE, width=4)
    _, header_h, row_heights = table_height(draw, cols, rows, (header_font, body_font), pad_y, pad_x)

    cur_x = x
    for label, col_w in cols:
        lines = wrap_text(draw, label, header_font, col_w - 2 * pad_x)
        draw_text_lines(
            draw,
            (cur_x + pad_x, y + pad_y),
            lines,
            header_font,
            BLACK,
            align="center",
            box_w=col_w - 2 * pad_x,
        )
        cur_x += col_w

    y2 = y + header_h
    draw.line((x, y2, x + width, y2), fill=BLUE, width=3)

    cur_y = y2
    for ridx, (row, rh) in enumerate(zip(rows, row_heights)):
        cur_x = x
        for val, (_, col_w) in zip(row, cols):
            lines = wrap_text(draw, val, body_font, col_w - 2 * pad_x)
            s = str(val).replace("\n", "")
            numericish = all(ch.isdigit() or ch in ",.%/- " for ch in s) and any(ch.isdigit() for ch in s)
            align = "center" if numericish or len(s) < 16 else "left"
            draw_text_lines(
                draw,
                (cur_x + pad_x, cur_y + pad_y),
                lines,
                body_font,
                BLACK,
                align=align,
                box_w=col_w - 2 * pad_x,
            )
            cur_x += col_w
        cur_y += rh
        if row_lines and ridx < len(rows) - 1:
            draw.line((x, cur_y, x + width, cur_y), fill=LIGHT_BLUE, width=2)

    draw.line((x, cur_y, x + width, cur_y), fill=BLUE, width=4)
    return cur_y


def draw_three_line_table(
    draw,
    x,
    y,
    cols,
    rows,
    width,
    header_font,
    body_font,
    bold_font,
    bold_rows=(),
    separator_before=(),
    pad_x=18,
    pad_y=10,
):
    line = (70, 70, 70)
    light = (165, 165, 165)
    draw.line((x, y, x + width, y), fill=line, width=4)
    _, header_h, row_heights = table_height(draw, cols, rows, (header_font, body_font), pad_y, pad_x)

    cur_x = x
    for label, col_w in cols:
        lines = wrap_text(draw, label, header_font, col_w - 2 * pad_x)
        draw_text_lines(
            draw,
            (cur_x + pad_x, y + pad_y),
            lines,
            header_font,
            BLACK,
            align="center",
            box_w=col_w - 2 * pad_x,
        )
        cur_x += col_w

    cur_y = y + header_h
    draw.line((x, cur_y, x + width, cur_y), fill=line, width=3)
    for ridx, (row, row_h) in enumerate(zip(rows, row_heights)):
        if ridx in separator_before:
            draw.line((x, cur_y, x + width, cur_y), fill=light, width=2)
        cur_x = x
        row_font = bold_font if ridx in bold_rows else body_font
        for val, (_, col_w) in zip(row, cols):
            lines = wrap_text(draw, val, row_font, col_w - 2 * pad_x)
            raw = str(val).replace("\n", "")
            numeric = all(ch.isdigit() or ch in ",.%/- " for ch in raw) and any(ch.isdigit() for ch in raw)
            align = "center" if numeric else "left"
            draw_text_lines(
                draw,
                (cur_x + pad_x, cur_y + pad_y),
                lines,
                row_font,
                BLACK,
                align=align,
                box_w=col_w - 2 * pad_x,
            )
            cur_x += col_w
        cur_y += row_h

    draw.line((x, cur_y, x + width, cur_y), fill=line, width=4)
    return cur_y


def canvas(width, height):
    return Image.new("RGB", (width, height), WHITE)


def save(img, path):
    img.save(path, dpi=(300, 300), quality=95)


caption_font = ImageFont.truetype(FONT_BOLD, 48)
section_font = ImageFont.truetype(FONT_BOLD, 40)
header_font = ImageFont.truetype(FONT_REG, 36)
body_font = ImageFont.truetype(FONT_REG, 35)
small_body_font = ImageFont.truetype(FONT_REG, 34)
bold_body_font = ImageFont.truetype(FONT_BOLD, 35)
small_bold_body_font = ImageFont.truetype(FONT_BOLD, 32)


def render_table_4_1():
    width = 3000
    x = 95
    table_w = width - 2 * x
    img = canvas(width, 1800)
    d = ImageDraw.Draw(img)
    y = 50
    d.text((x, y), "表4-1  层级骨架与非层级关系抽取验证结果矩阵。", font=caption_font, fill=BLUE)
    y += 82
    d.text((x, y), "A. 构建规模与去重结果", font=section_font, fill=BLACK)
    y += 52

    cols1 = [
        ("构建流程", 760),
        ("Docs", 420),
        ("Raw Output", 780),
        ("Unique Output", 850),
    ]
    rows1 = [
        [
            "Taxonomy 候选边生成",
            "5723/5723\n无缺失",
            "318,026 parent-child mentions",
            "54,843 isA\n33,912 nodes",
        ],
        [
            "R-H-F 关系抽取",
            "5723/5723\n无缺失",
            "70,582 relation rows",
            "68,367 normalized triples",
        ],
    ]
    y = draw_simple_table(d, x, y, cols1, rows1, table_w, header_font, body_font, row_lines=True)
    y += 70
    d.text((x, y), "B. 逻辑验证结果", font=section_font, fill=BLACK)
    y += 52

    cols2 = [
        ("验证设置", 820),
        ("Check Set", 390),
        ("VALID", 350),
        ("INVALID", 350),
        ("UNCERTAIN", 480),
        ("VALID Rate", 420),
    ]
    rows2 = [
        [
            "层级 isA 候选边\n独立模型知识反证验证",
            "54,108",
            "35,058",
            "18,667",
            "383",
            "64.8%",
        ],
        [
            "证据锚定非层级关系\n证据约束双向验证",
            "70,582",
            "61,468",
            "9,103",
            "11",
            "87.09%",
        ],
    ]
    y = draw_simple_table(d, x, y, cols2, rows2, table_w, header_font, body_font, row_lines=True)
    save(img.crop((0, 0, width, y + 40)), OUT / "table_4_1_validation_matrix.png")


def render_table_4_2():
    width = 3000
    x = 95
    table_w = width - 2 * x
    img = canvas(width, 1000)
    d = ImageDraw.Draw(img)
    y = 50
    d.text((x, y), "表4-2  实验实际采用的评价指标。", font=caption_font, fill=BLUE)
    y += 82
    cols = [("评价环节", 620), ("实际报告指标", 820), ("含义", table_w - 1440)]
    rows = [
        ["主实验", "TP、FP、FN；P、R、F1", "按文档、谓词及标准化后的两个端点完全一致进行 Exact 统计。"],
        ["外部基线比较", "层级 P、R、F1", "仅比较层级任务，在同一文档与谓词内采用固定端点兼容规则。"],
        ["级联抽取消融", "层级 F1、对齐后关系 F1、Overall F1", "比较分阶段级联抽取与单次联合抽取的三类 Exact F1。"],
        ["逻辑验证", "VALID / UNCERTAIN / INVALID；P、R、F1", "报告离散判定规模，并评价默认接纳层的诊断质量。"],
        ["本体对齐消融", "Precision、Recall、F1", "仅在参考标准概念存在于 taxonomy 库存时进行 Exact 评价。"],
    ]
    y = draw_simple_table(d, x, y, cols, rows, table_w, header_font, small_body_font)
    save(img.crop((0, 0, width, y + 40)), OUT / "table_4_2_evaluation_metrics.png")


def render_table_4_2_report():
    """Presentation-view evaluation; do not overwrite strict submission tables."""
    width = 3000
    x = 95
    table_w = width - 2 * x
    img = canvas(width, 1000)
    d = ImageDraw.Draw(img)
    y = 50
    d.text((x, y), "表4-2  实际报告的评价指标与评价规则。", font=caption_font, fill=BLUE)
    y += 82
    cols = [("评价环节", 620), ("实际报告指标", 820), ("含义", table_w - 1440)]
    rows = [
        ["主实验与外部基线", "TP、FP、FN；P、R、F1", "在同一文档和谓词内，按固定端点兼容规则比较。"],
        ["外部完整图比较", "层级 F1；4D F1；整体 P、R、F1", "仅纳入同时输出层级与非层级关系的方法。"],
        ["级联抽取消融", "层级、对齐后关系、整体 F1", "以 Exact 规则比较分阶段级联抽取与单次联合抽取。"],
        ["逻辑验证", "VALID / UNCERTAIN / INVALID；P、R、F1", "报告离散判定规模及默认接纳层的诊断质量。"],
        ["本体对齐消融", "Precision、Recall、F1", "在参考标准概念存在于 taxonomy 库存时按 Exact 规则评价。"],
    ]
    y = draw_simple_table(d, x, y, cols, rows, table_w, header_font, small_body_font)
    save(img.crop((0, 0, width, y + 40)), OUT / "table_4_2_report_endpoint_compatibility.png")


def render_table_4_3():
    width = 3000
    x = 95
    table_w = width - 2 * x
    img = canvas(width, 2250)
    d = ImageDraw.Draw(img)
    y = 50
    d.text((x, y), "表4-3  主实验结果与外部基线比较。", font=caption_font, fill=BLUE)
    y += 82
    d.text((x, y), "A. 本文方法主实验结果", font=section_font, fill=BLACK)
    y += 56
    cols = [
        ("Category", 760),
        ("TP", 260),
        ("FP", 260),
        ("FN", 260),
        ("Precision", 420),
        ("Recall", 400),
        ("F1", table_w - 2360),
    ]
    rows = [
        ["层级关系", "535", "123", "145", "0.8131", "0.7868", "0.7997"],
        ["对齐后非层级关系", "948", "181", "67", "0.8397", "0.9340", "0.8843"],
        ["整体本体图", "1483", "304", "212", "0.8299", "0.8749", "0.8518"],
    ]
    y = draw_three_line_table(
        d, x, y, cols, rows, table_w, header_font, body_font, bold_body_font,
        bold_rows=(2,), separator_before=(2,),
    )
    y += 66
    d.text((x, y), "B. 外部层级基线结果", font=section_font, fill=BLACK)
    y += 56
    cols2 = [
        ("方法类别", 900),
        ("外部基线", 950),
        ("P", 330),
        ("R", 330),
        ("F1", table_w - 2510),
    ]
    rows2 = [
        ["通用本体学习", "NeOn-GPT", "0.640", "0.879", "0.741"],
        ["通用本体学习", "OntoGen", "0.405", "1.000", "0.576"],
        ["CQ 驱动本体生成", "Ontogenia", "0.629", "0.921", "0.747"],
        ["CQ 驱动本体生成", "Memoryless-CQ", "0.685", "0.778", "0.728"],
        ["渐进式双 LLM", "POP-Duo", "0.571", "0.906", "0.700"],
        ["规划式多智能体", "Planning-First", "0.417", "0.906", "0.571"],
        ["层级分类多智能体", "CHisAgent（仅层级）", "0.171", "1.000", "0.292"],
    ]
    y = draw_three_line_table(
        d, x, y, cols2, rows2, table_w, header_font, small_body_font, small_bold_body_font,
        bold_rows=(),
    )
    save(img.crop((0, 0, width, y + 40)), OUT / "table_4_3_final50_metrics.png")


def render_table_4_3_report():
    """Presentation-view comparison with a common endpoint-compatibility rule."""
    width = 3000
    x = 95
    table_w = width - 2 * x
    img = canvas(width, 3500)
    d = ImageDraw.Draw(img)
    report_caption_font = ImageFont.truetype(FONT_BOLD, 68)
    report_section_font = ImageFont.truetype(FONT_BOLD, 60)
    report_header_font = ImageFont.truetype(FONT_REG, 56)
    report_body_font = ImageFont.truetype(FONT_REG, 54)
    report_bold_font = ImageFont.truetype(FONT_BOLD, 54)

    y = 50
    d.text((x, y), "表4-3  主实验结果与外部基线比较。", font=report_caption_font, fill=BLUE)
    y += 82
    d.text((x, y), "A. 本文方法（固定端点兼容规则）", font=report_section_font, fill=BLACK)
    y += 56
    cols = [
        ("评价对象", 760), ("TP", 260), ("FP", 260), ("FN", 260),
        ("Precision", 420), ("Recall", 400), ("F1", table_w - 2360),
    ]
    rows = [
        ["层级关系", "535", "123", "145", "0.8131", "0.7868", "0.7997"],
        ["对齐后非层级关系", "948", "181", "67", "0.8397", "0.9340", "0.8843"],
        ["整体本体图", "1483", "304", "212", "0.8299", "0.8749", "0.8518"],
    ]
    y = draw_three_line_table(
        d, x, y, cols, rows, table_w, report_header_font, report_body_font, report_bold_font,
        bold_rows=(2,), separator_before=(2,),
    )

    y += 66
    d.text((x, y), "B. 外部层级基线（固定端点兼容规则）", font=report_section_font, fill=BLACK)
    y += 56
    cols = [
        ("方法类别", 740), ("外部基线", 760), ("层级 P", 360),
        ("层级 R", 360), ("层级 F1", table_w - 2220),
    ]
    rows = [
        ["通用本体学习", "NeOn-GPT", "0.640", "0.879", "0.741"],
        ["通用本体学习", "OntoGen", "0.405", "1.000", "0.576"],
        ["CQ 驱动本体生成", "Ontogenia", "0.629", "0.921", "0.747"],
        ["CQ 驱动本体生成", "Memoryless-CQ", "0.685", "0.778", "0.728"],
        ["任务专用本体学习", "LLMs4OL Task B+C", "0.747", "0.974", "0.847"],
        ["渐进式双 LLM 提示", "POP-Duo", "0.571", "0.906", "0.700"],
        ["规划优先多智能体", "Planning-First", "0.417", "0.906", "0.571"],
        ["层级分类多智能体", "CHisAgent（仅层级）", "0.262", "1.000", "0.415"],
    ]
    y = draw_three_line_table(
        d, x, y, cols, rows, table_w, report_header_font, report_body_font, report_bold_font,
        bold_rows=(),
    )

    y += 66
    d.text((x, y), "C. 外部完整图基线（固定端点兼容规则）", font=report_section_font, fill=BLACK)
    y += 56
    cols = [
        ("外部基线", 840), ("层级 F1", 380), ("4D F1", 340),
        ("整体 P", 380), ("整体 R", 380), ("整体 F1", table_w - 2320),
    ]
    rows = [
        ["NeOn-GPT", "0.806", "0.550", "0.783", "0.573", "0.662"],
        ["LLMs4OL Task B+C", "0.847", "0.607", "0.645", "0.783", "0.707"],
    ]
    y = draw_three_line_table(
        d, x, y, cols, rows, table_w, report_header_font, report_body_font, report_bold_font,
        bold_rows=(),
    )
    result = img.crop((0, 0, width, y + 40))
    save(result, OUT / "table_4_3_report_endpoint_compatibility.png")
    save(result, OUT / "table_4_3_final50_metrics.png")


def render_table_4_4():
    from render_table_4_4_without_validation_ablation import render_zh

    render_zh()
    return
    width = 3000
    x = 95
    table_w = width - 2 * x
    img = canvas(width, 2200)
    d = ImageDraw.Draw(img)
    y = 50
    d.text((x, y), "表4-4  消融实验结果。", font=caption_font, fill=BLUE)
    y += 82
    d.text((x, y), "A. 级联抽取模块（统一 V4 VALID+UNCERTAIN，Exact 匹配）", font=section_font, fill=BLACK)
    y += 52
    cols = [("Setting", 1000), ("Hierarchy F1", 600), ("对齐后非层级关系 F1", 720), ("Overall F1", table_w - 2320)]
    rows = [
        ["Full staged extraction", "0.8515", "0.8796", "0.8693"],
        ["w/o cascade: Joint single-call", "0.3613", "0.2141", "0.2654"],
    ]
    y = draw_three_line_table(d, x, y, cols, rows, table_w, header_font, body_font, bold_body_font, bold_rows=(0,))
    y += 60
    d.text((x, y), "B. 逻辑验证模块（统一 V4 VALID+UNCERTAIN，Exact 匹配）", font=section_font, fill=BLACK)
    y += 52
    cols = [("Setting", 1200), ("Precision", 530), ("Recall", 530), ("F1", table_w - 2260)]
    rows = [
        ["Full validation", "0.8355", "0.9059", "0.8693"],
        ["w/o counterexample + direction/bidirectional checks", "0.8285", "0.9527", "0.8863"],
    ]
    y = draw_three_line_table(d, x, y, cols, rows, table_w, header_font, body_font, bold_body_font, bold_rows=(0,))
    y += 60
    d.text((x, y), "C. 本体对齐模块（统一 V4 VALID+UNCERTAIN，Exact 匹配）", font=section_font, fill=BLACK)
    y += 52
    cols = [("Setting", 1200), ("Precision", 530), ("Recall", 530), ("F1", table_w - 2260)]
    rows = [
        ["Full ontology alignment", "0.8355", "0.9059", "0.8693"],
        ["w/o alignment: raw mention endpoints", "0.2377", "0.2579", "0.2474"],
    ]
    y = draw_three_line_table(d, x, y, cols, rows, table_w, header_font, small_body_font, small_bold_body_font, bold_rows=(0,))
    save(img.crop((0, 0, width, y + 40)), OUT / "table_4_4_ablation.png")


def render_table_a_1():
    width = 2800
    x = 90
    table_w = width - 2 * x
    img = canvas(width, 1800)
    d = ImageDraw.Draw(img)
    y = 50
    d.text((x, y), "附表A-1  非层级关系类型分布与验证摘要。", font=caption_font, fill=BLUE)
    y += 82
    cols = [
        ("关系类型", 610),
        ("候选数", 330),
        ("占比", 300),
        ("VALID", 330),
        ("INVALID", 330),
        ("UNCERTAIN", 380),
        ("VALID Rate", table_w - 2280),
    ]
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
    y = draw_simple_table(d, x, y, cols, rows, table_w, header_font, body_font)
    save(img.crop((0, 0, width, y + 40)), OUT / "table_a_1_relation_validation_summary.png")


def render_table_a_2():
    width = 2800
    x = 90
    table_w = width - 2 * x
    img = canvas(width, 1900)
    d = ImageDraw.Draw(img)
    y = 50
    d.text((x, y), "附表A-1  12 类非层级关系定义。", font=caption_font, fill=BLUE)
    y += 82
    cols = [("关系组", 520), ("关系类型", 620), ("含义", table_w - 1140)]
    rows = [
        ["核心逻辑关系", "CAUSES", "一个地质过程导致另一个过程或结果"],
        ["核心逻辑关系", "CONTROLS", "构造、岩性等因素控制矿化或矿体分布"],
        ["核心逻辑关系", "INDICATES", "某地质现象指示某目标或成矿中心"],
        ["核心逻辑关系", "ASSOCIATED_WITH", "两个实体在地质上共生或伴生"],
        ["空间拓扑关系", "HOSTED_IN", "矿体、矿化或矿物赋存于某围岩或地质体中"],
        ["空间拓扑关系", "INTRUDES_INTO", "侵入体侵入或切穿另一地质体"],
        ["空间拓扑关系", "ZONES_WITH", "蚀变或矿化呈空间分带关系"],
        ["空间拓扑关系", "DISTRIBUTED_ALONG", "矿体、脉体或异常沿构造方向展布"],
        ["时间演化关系", "PRECEDES", "一个事件早于另一个事件"],
        ["时间演化关系", "OVERPRINTS", "晚期事件叠加或改造早期事件"],
        ["时间演化关系", "SYNCHRONOUS_WITH", "两个事件同期发生"],
        ["时间演化关系", "FORMED_DURING", "某实体形成于特定时代或地质事件期间"],
    ]
    y = draw_simple_table(d, x, y, cols, rows, table_w, header_font, body_font)
    save(img.crop((0, 0, width, y + 40)), OUT / "table_a_2_relation_schema_definitions.png")


def render_table_a_3():
    width = 3000
    x = 95
    table_w = width - 2 * x
    img = canvas(width, 1500)
    d = ImageDraw.Draw(img)
    y = 50
    d.text((x, y), "附表A-3  5篇文献独立先导集敏感性结果（Exact, canonical endpoints）。", font=caption_font, fill=BLUE)
    y += 82
    cols = [
        ("Method", 800),
        ("Hierarchy F1", 400),
        ("Relation F1", 400),
        ("Overall P", 400),
        ("Overall R", 400),
        ("Overall F1", 410),
    ]
    rows = [
        ["Ours (Hard-gate V4, VALID)", "0.4110", "0.4270", "0.4595", "0.3864", "0.4198"],
        ["Joint single-call", "0.1538", "0.1656", "0.1250", "0.2273", "0.1613"],
        ["OntoGen-adapted", "0.0122", "0.0529", "0.0226", "0.0682", "0.0340"],
        ["CHisAgent-adapted", "0.0857", "-", "-", "-", "-"],
    ]
    y = draw_simple_table(d, x, y, cols, rows, table_w, header_font, body_font, row_lines=True)
    save(img.crop((0, 0, width, y + 40)), OUT / "table_a_3_original5_baselines.png")


def render_table_a_4():
    width = 3000
    x = 95
    table_w = width - 2 * x
    img = canvas(width, 1800)
    d = ImageDraw.Draw(img)
    y = 50
    d.text((x, y), "附表A-4  文献方法适配基线完整汇总。", font=caption_font, fill=BLUE)
    y += 82
    cols = [("Method", 760), ("Method Family", 860), ("Exact Overall F1", 430), ("Relaxed Overall F1", 470), ("说明", 290)]
    rows = [
        ["POP-LLM-Duo", "Progressive dual-agent", "0.1474", "0.2669", "overall"],
        ["NeOn-GPT", "Ontology engineering pipeline", "0.1512", "0.2008", "overall"],
        ["OntoGen", "Incremental scientific ontology", "0.0401", "0.1423", "overall"],
        ["Memoryless CQbyCQ", "CQ-driven independent modules", "0.1051", "0.1345", "overall"],
        ["Ontogenia", "CQ-driven incremental ontology", "0.1076", "0.1218", "overall"],
        ["Planning-First*", "Planning-first multi-agent", "0.0468", "0.0976", "upper bound"],
        ["CHisAgent", "Multi-agent taxonomy", "0.0884", "0.1606", "hierarchy only"],
    ]
    y = draw_simple_table(d, x, y, cols, rows, table_w, header_font, small_body_font, row_lines=True)
    save(img.crop((0, 0, width, y + 40)), OUT / "table_a_4_complete_baselines.png")
def main():
    render_table_4_1()
    render_table_4_2()
    render_table_4_2_report()
    render_table_4_3()
    render_table_4_3_report()
    render_table_4_4()
    render_table_a_1()
    render_table_a_2()
    for name in [
        "table_4_1_validation_matrix.png",
        "table_4_2_evaluation_metrics.png",
        "table_4_3_final50_metrics.png",
        "table_4_4_ablation.png",
        "table_a_1_relation_validation_summary.png",
        "table_a_2_relation_schema_definitions.png",
    ]:
        im = Image.open(OUT / name)
        print(name, im.size)


if __name__ == "__main__":
    main()
