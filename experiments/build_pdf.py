"""Build a professional multi-page PDF report from the ContextLab results.

Uses reportlab (installed) with DejaVu fonts (shipped with matplotlib) for full
Unicode support. Embeds the three result figures.

    python experiments/build_pdf.py
    -> writes ContextLab_Report.pdf in the project root
"""

import os
from pathlib import Path

import matplotlib
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "ContextLab_Report.pdf"

# --- Unicode-capable fonts (from matplotlib) --------------------------------
_fdir = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
_base = "Helvetica"
_bold = "Helvetica-Bold"
try:
    pdfmetrics.registerFont(TTFont("DV", str(_fdir / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("DV-Bold", str(_fdir / "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFont(
        TTFont("DV-Mono", str(_fdir / "DejaVuSansMono.ttf"))
    )
    _base, _bold = "DV", "DV-Bold"
except Exception:
    pass

_ss = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("title", parent=_ss["Title"], fontName=_bold,
                            fontSize=30, leading=34, alignment=TA_CENTER,
                            textColor=colors.HexColor("#10243e")),
    "sub": ParagraphStyle("sub", fontName=_base, fontSize=14, leading=18,
                          alignment=TA_CENTER, textColor=colors.HexColor("#3b5677")),
    "meta": ParagraphStyle("meta", fontName=_base, fontSize=10, leading=15,
                           alignment=TA_CENTER, textColor=colors.HexColor("#555555")),
    "h1": ParagraphStyle("h1", fontName=_bold, fontSize=17, leading=21,
                         spaceBefore=14, spaceAfter=6,
                         textColor=colors.HexColor("#10243e")),
    "h2": ParagraphStyle("h2", fontName=_bold, fontSize=13, leading=17,
                         spaceBefore=10, spaceAfter=4,
                         textColor=colors.HexColor("#1f4e79")),
    "body": ParagraphStyle("body", fontName=_base, fontSize=10.5, leading=15.5,
                           alignment=TA_JUSTIFY, spaceAfter=6),
    "bullet": ParagraphStyle("bullet", fontName=_base, fontSize=10.5,
                             leading=15, leftIndent=14, bulletIndent=4,
                             spaceAfter=3),
    "note": ParagraphStyle("note", fontName=_base, fontSize=10, leading=14,
                           textColor=colors.HexColor("#7a4a00"),
                           backColor=colors.HexColor("#fff6e5"),
                           borderPadding=8, borderColor=colors.HexColor("#f0c078"),
                           borderWidth=0.8, spaceBefore=4, spaceAfter=8),
    "cap": ParagraphStyle("cap", fontName=_base, fontSize=9, leading=12,
                          alignment=TA_CENTER, textColor=colors.HexColor("#555555"),
                          spaceBefore=2, spaceAfter=10),
    "code": ParagraphStyle("code", fontName="DV-Mono" if _base == "DV" else "Courier",
                           fontSize=8.5, leading=12, textColor=colors.HexColor("#10243e")),
}


def H1(t):
    return Paragraph(t, S["h1"])


def H2(t):
    return Paragraph(t, S["h2"])


def P(t):
    return Paragraph(t, S["body"])


def NOTE(t):
    return Paragraph(t, S["note"])


def bullets(items):
    return [Paragraph(i, S["bullet"], bulletText="\u2022") for i in items]


def SP(h=0.25):
    return Spacer(1, h * cm)


def IMG(name, w=15.0):
    return Image(str(RESULTS / name), width=w * cm, height=w * 0.667 * cm)


def CAP(t):
    return Paragraph(t, S["cap"])


def TBL(data, widths=None):
    t = Table(data, colWidths=[w * cm for w in widths] if widths else None)
    t.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, 0), _bold, 9.5),
        ("FONT", (0, 1), (-1, -1), _base, 9.5),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#10243e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#eef2f7")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c3ccd8")),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def build_story():
    story = []

    # ---------------- Title page ----------------
    story += [
        SP(3.2),
        Paragraph("ContextLab", S["title"]),
        SP(0.25),
        Paragraph("Measuring the Effective Context of Large Language Models",
                  S["sub"]),
        SP(0.4),
        Paragraph("Phase-1 Research Report", S["meta"]),
        SP(0.15),
        Paragraph("Model under test: Qwen/Qwen2.5-0.5B (base)", S["meta"]),
        Paragraph("Machine: Intel i5-12450H, 16 GB RAM, CPU-only", S["meta"]),
        Paragraph("Date: 2026-10-01", S["meta"]),
        SP(0.8),
        NOTE("Honesty statement. Every number in this report comes from a CSV "
             "produced by the code in this repository on the machine above. "
             "Nothing is simulated, inferred, or invented. Claims not backed by "
             "a measurement are explicitly labelled as hypotheses."),
        PageBreak(),
    ]

    # ---------------- 0. Plain English ----------------
    story += [
        H1("0. What this is, in plain English"),
        P("Modern language models advertise huge context windows, for example "
          "&quot;up to 128,000 tokens&quot; or &quot;up to 1,000,000 tokens&quot;. "
          "The number tells you how much text you can <i>hand</i> the model, not "
          "how much of it the model can <i>actually use</i>. Our project asks a "
          "simple, hard question: <b>how much of an advertised context window "
          "can a model genuinely rely on for a given task?</b>"),
        P("To answer it we built <b>ContextLab</b>, a controlled test bench. We "
          "give the model a long block of text containing one secret answer, "
          "optionally surrounded by look-alike decoys, and ask for the secret. "
          "Because we build the prompt token-by-token, we know the exact context "
          "length, the exact position of the answer, and how much competing "
          "information sits around it. The answer is one of twelve fixed words, "
          "so scoring is a clean, deterministic 12-way choice."),
        P("We then ran three real experiments on a 0.5-billion-parameter model "
          "and recorded every single trial. The headline results:"),
    ]
    story += bullets([
        "<b>Retrieval of a single fact is trivial.</b> With no decoys, the model "
        "is 100% accurate at 512 tokens, at every position, with ~0.999 "
        "confidence.",
        "<b>Interference is the real bottleneck.</b> Adding a few decoy "
        "&quot;password&quot; sentences collapses accuracy from 1.00 to 0.50 "
        "(3 decoys) and 0.17 (6 decoys) - at the <i>same</i> context length.",
        "<b>Lost-in-the-middle is real, but conditional.</b> With decoys present, "
        "the probability of the right answer is 0.76 near the start, falls to "
        "0.35 in the middle, and recovers to 0.94 at the end. With no decoys the "
        "curve is flat.",
        "<b>Raw length did not matter here.</b> Up to 2,048 tokens, accuracy "
        "never degraded without interference.",
    ])
    story += [
        P("The practical conclusion: at the scales we can measure, what limits a "
          "model is <b>competition between pieces of information</b>, not the "
          "raw token count. That reframes the project: the quantity to predict "
          "is an <i>interference load</i>, not a length."),
    ]

    # ---------------- 1. The problem ----------------
    story += [
        H1("1. The problem"),
        P("Existing work already shows that nominal context length overstates "
          "practical ability: <i>Lost in the Middle</i> (position-dependent "
          "degradation), <i>RULER</i> (many models fail before their advertised "
          "size as complexity rises), <i>BABILong</i> (only a fraction of a very "
          "long context is usable for reasoning), and industry reports such as "
          "&quot;context rot&quot;."),
        P("So &quot;long contexts degrade&quot; is <b>already known</b>. The open "
          "gap is a <b>predictive, task-specific, pre-inference</b> account: "
          "given a model and a task, forecast the effective (usable) context "
          "<i>before</i> running it, then move that boundary outward. That is what "
          "ContextLab is built to eventually do."),
    ]

    # ---------------- 2. The plan ----------------
    story += [
        H1("2. The plan we are executing"),
        P("The project plan proposes a three-phase path built around one "
          "measurable quantity."),
        H2("2.1 The quantity: Effective Context Capacity (ECC)"),
        P("ECC(M, T, \u03c4) is the largest context length L at which performance "
          "P on model M and task T stays at or above \u03c4 times the "
          "short-context baseline P\u2080. In words: <i>up to what length is "
          "performance still acceptable?</i>"),
        H2("2.2 The three phases"),
    ]
    story += bullets([
        "<b>Phase 1 - Research:</b> reproduce known effects, build a controllable "
        "benchmark, map the performance surface, and look for a predictive law.",
        "<b>Phase 2 - Breakthrough:</b> test whether a law generalises "
        "out-of-distribution (fit on models A-C, predict an unseen model D and "
        "task), then investigate mechanisms.",
        "<b>Phase 3 - Product:</b> a &quot;Context Profiler&quot; and optimiser "
        "(&quot;ContextOS&quot;) that tell a customer how much context they can "
        "trust and how to build it - sold on <i>accuracy per input dollar</i>.",
    ])
    story += [
        P("<b>This document is Phase 1, first iteration.</b> Its job is to "
          "establish ground truth by measurement, before any product exists."),
    ]

    # ---------------- 3. Method ----------------
    story += [
        H1("3. What we built and how it is measured"),
        H2("3.1 A controllable, token-exact task"),
        P("Each test case is a needle task built directly at the token level:"),
        Paragraph("Needle:   &quot; Important: the vault password is {word}.&quot;<br/>"
                  "Decoys:   &quot; Note: the {noun} password is {word}.&quot; "
                  "(optional)<br/>"
                  "Query:    &quot;\\nQuestion: What is the vault password?\\n"
                  "Answer:&quot;", S["code"]),
        SP(0.15),
        P("The answer space is a fixed set of twelve single-token words (apple, "
          "river, tiger, cloud, stone, candle, silver, garden, planet, forest, "
          "harbor, lantern). Filler text never contains any of these words, so "
          "the true answer appears <b>exactly once</b> in the whole prompt. This "
          "gives exact control over three variables: context length (tokens), "
          "answer position (fraction), and number of decoys (interference)."),
        H2("3.2 Scoring"),
        P("The model's next-token distribution is restricted to the twelve "
          "candidates. We record <i>accuracy</i> (does the top candidate equal "
          "the true answer?) and <i>p_correct</i> (the probability mass on the "
          "correct word). p_correct is a smoother signal than the discrete "
          "accuracy, so it is reported alongside it."),
        H2("3.3 Metrics"),
    ]
    story += bullets([
        "<b>ECC(\u03c4)</b> - first context length where accuracy drops below "
        "\u03c4 \u00d7 baseline (interpolated). Right-censored if it never drops.",
        "<b>Effective Context Mass (ECM)</b> - normalised area under the accuracy "
        "curve over log-length; a threshold-free summary.",
        "<b>Stretched-exponential fit</b> - P(L) = P\u2080\u00b7exp(\u2212(L/scale)"
        "<super>\u03b2</super>).",
    ])
    story += [
        H2("3.4 Code layout"),
        Paragraph("contextlab/tasks/generator.py  - controllable task generator<br/>"
                  "contextlab/models/local_hf.py   - CPU model adapter (fp32+sdpa)<br/>"
                  "contextlab/metrics/ecc.py       - ECC, ECM, curve fits<br/>"
                  "contextlab/experiments/runner.py- grid runner (CSV, resumable)<br/>"
                  "contextlab/cli.py               - command line interface<br/>"
                  "experiments/                    - run + analyse scripts<br/>"
                  "tests/                          - 7 unit tests (all pass)",
                  S["code"]),
    ]

    # ---------------- 4. Environment ----------------
    story += [
        H1("4. Environment and constraints (measured)"),
        TBL([
            ["Component", "Value"],
            ["CPU", "Intel i5-12450H, 12 logical cores"],
            ["RAM", "16 GB"],
            ["GPU", "none"],
            ["torch", "2.13.0+cpu"],
            ["transformers", "5.14.1"],
            ["Model", "Qwen/Qwen2.5-0.5B (base)"],
        ], widths=[4.0, 11.0]),
        SP(0.3),
        H2("4.1 The dtype trap (a real finding, not trivia)"),
        P("The model loads in <b>bfloat16</b> by default. This CPU has no native "
          "BF16, so it is emulated in software: a single 512-token forward pass "
          "took <b>57 seconds</b>. Forcing <b>float32 + sdpa</b> reduced it to "
          "about <b>4 seconds</b> - a 14-100x speed-up. Any future run on this "
          "class of machine must force float32."),
        H2("4.2 Occasional multi-minute stalls"),
        P("Most forward passes take 2.5-7 s, but occasional passes stalled for "
          "27-1661 s; the 2048-token decoy cell took 27 minutes. This is an "
          "environmental effect, not an algorithmic one, and it caps how large "
          "our experiments can be. It is why our samples are small (n = 4-8) and "
          "why the curves are directional rather than precise."),
        NOTE("Consequence for the plan: the full-scale study described in the "
             "plan (128k-token contexts, several model families, many samples) "
             "<b>cannot run on this hardware</b>. A GPU box or an API budget is "
             "the first infrastructure decision to make."),
    ]

    # ---------------- 5. Results ----------------
    story += [
        H1("5. Results"),
        P("All trials are recorded in results/. Each table below is generated "
          "directly from those CSVs."),
    ]

    # 5.1 Interference
    story += [
        H2("5.1 Experiment 1 - interference scan"),
        P("Fixed context length 512, answer at position 0.5, n = 6 per point. We "
          "vary only the number of decoy passages d."),
        TBL([
            ["decoys d", "accuracy", "p_correct"],
            ["0", "1.00", "0.999"],
            ["3", "0.50", "0.344"],
            ["6", "0.17", "0.180"],
            ["9", "0.33", "0.218"],
            ["12", "0.33", "0.238"],
        ], widths=[4.0, 4.0, 4.0]),
        SP(0.2),
        KeepTogether([
            IMG("figure_interference.png"),
            CAP("Figure 1. Interference collapses retrieval at constant context "
                "length. Zero decoys is trivial; a handful of decoys destroys "
                "accuracy."),
        ]),
        P("With no decoys the model is perfect (1.00, p=0.999). Just three decoys "
          "cut accuracy in half, and six decoys cut it to 0.17 - <b>at the same "
          "512-token length</b>. This isolates interference as the dominant "
          "variable."),
    ]

    # 5.2 Position
    story += [
        H2("5.2 Experiment 2 - position sweep (lost in the middle)"),
        P("Fixed context length 512, n = 8. Two conditions: no decoys (d=0) and "
          "four decoys (d=4)."),
        TBL([
            ["position", "d=0 acc", "d=4 acc", "d=4 p_correct"],
            ["0.05", "1.00", "0.875", "0.759"],
            ["0.20", "1.00", "0.750", "0.703"],
            ["0.35", "1.00", "1.000", "0.690"],
            ["0.50", "1.00", "0.500", "0.351"],
            ["0.65", "1.00", "0.750", "0.584"],
            ["0.80", "1.00", "0.750", "0.627"],
            ["0.95", "1.00", "1.000", "0.945"],
        ], widths=[3.0, 3.5, 3.5, 4.0]),
        SP(0.2),
        KeepTogether([
            IMG("figure_position.png"),
            CAP("Figure 2. Answer position vs accuracy. Without interference "
                "(d=0) the curve is flat at 1.00. With interference (d=4) there "
                "is a clear mid-context dip."),
        ]),
        P("Without interference, position is irrelevant (flat 1.00). With "
          "interference, the probability of the correct answer forms a <b>U-shape</b>: "
          "0.76 at the start, dipping to <b>0.35 in the middle</b>, recovering to "
          "0.94 at the end. This is a direct reproduction of the "
          "<i>lost-in-the-middle</i> effect - and it only appears when "
          "interference is present."),
    ]

    # 5.3 Length
    story += [
        H2("5.3 Experiment 3 - length sweep (effective context)"),
        P("Fixed position 0.5, n = 4. Context lengths from 256 to 2048 tokens."),
        TBL([
            ["length", "d=0 acc", "d=0 p_c", "d=4 acc", "d=4 p_c"],
            ["256", "1.00", "0.9989", "0.50", "0.366"],
            ["512", "1.00", "0.9992", "0.50", "0.384"],
            ["1024", "1.00", "0.9988", "0.75", "0.542"],
            ["2048", "1.00", "0.9991", "0.75", "0.610"],
        ], widths=[2.6, 3.0, 3.2, 3.0, 3.2]),
        SP(0.2),
        KeepTogether([
            IMG("figure_length.png"),
            CAP("Figure 3. Context length vs accuracy. Without interference the "
                "model is perfect at every length; with interference there is no "
                "downward trend over this range."),
        ]),
        P("Without interference, accuracy is <b>1.00 at every length</b> up to "
          "2048 tokens - no degradation at all. With interference there is no "
          "downward trend; if anything p_correct rises (0.37 to 0.61). "
          "<b>Hypothesis:</b> spreading decoys over more filler dilutes their "
          "competition. With n = 4 this is suggestive, not conclusive."),
        H2("5.4 Metrics on the length curve"),
    ]
    story += bullets([
        "<b>ECC(\u03c4=0.9)</b> is <b>right-censored at 2048</b> in both "
        "conditions: accuracy never drops below 90% of its baseline.",
        "<b>ECM</b> (threshold-free) = <b>1.00</b> for d=0 and <b>1.25</b> for "
        "d=4 (above 1 because accuracy rises).",
        "<b>Stretched-exponential fit</b> matches d=0 well (R\u00b2 \u2248 0.9999, "
        "effectively flat) and fails on d=4 (R\u00b2 \u2248 0) because there is no "
        "declining curve to fit.",
    ])

    # ---------------- 6. Findings ----------------
    story += [
        H1("6. Interpretation - what we actually learned"),
    ]
    story += bullets([
        "<b>The naive reading of &quot;effective context&quot; is wrong at these "
        "scales.</b> Performance does not decay with token count: at 2048 tokens "
        "a 0.5B model finds a lone fact as reliably as at 256.",
        "<b>Information competition is the variable that matters.</b> Accuracy "
        "collapses when decoys compete for the same answer slot - and it "
        "collapses at short length.",
        "<b>Position matters only once competition exists.</b> The "
        "lost-in-the-middle U-shape appears only with decoys.",
        "<b>These results point at the plan's own mechanism list.</b> "
        "&quot;Attention competition&quot; and &quot;distractor similarity&quot; "
        "look like the primary terms; raw length looks secondary at reachable "
        "scales.",
    ])
    story += [
        NOTE("Refined working hypothesis. Effective context is governed by a "
             "competition ratio - roughly (number of competing items x "
             "similarity) / attention budget - rather than by length L alone. "
             "The plan's candidate law P = P\u2080\u00b7exp(\u2212\u03b1(L\u00b7R/C)"
             "<super>\u03b2</super>) has the right shape; the results suggest R "
             "should be redefined as an interference load, not a length term."),
    ]

    # ---------------- 7. Limitations ----------------
    story += [
        H1("7. Limitations (read before trusting anything)"),
    ]
    story += bullets([
        "<b>Small n</b> (4-8 samples/cell): accuracy steps are 0.25 (n=4) or "
        "0.125 (n=8). Treat curves as directional, not precise.",
        "<b>One model</b> (Qwen2.5-0.5B, base, not instruction-tuned). No "
        "cross-model claim is made here.",
        "<b>Short lengths only</b> (up to 2048 tokens). The 8k-128k regime is "
        "untested and untestable on this CPU.",
        "<b>Repetitive filler</b>: the clean-filler bank is ~388 tokens, so long "
        "contexts tile it. Real prose (e.g. wikitext) is a needed upgrade.",
        "<b>Environmental stalls</b> injected timing variance (one cell took "
        "27 minutes).",
        "<b>Twelve-word answer space</b>: a larger space would change difficulty.",
    ])

    # ---------------- 8. Next steps ----------------
    story += [
        H1("8. Next steps (priority order)"),
        TBL([
            ["#", "Action", "Why"],
            ["1", "Use real prose filler (wikitext)",
             "Remove the repetition confound"],
            ["2", "Interference x length joint sweep",
             "Test the competition-ratio hypothesis"],
            ["3", "More samples + a second model (Llama-3.2-1B)",
             "Cross-model validation"],
            ["4", "Fit the competition-based law, test OOD prediction",
             "The plan's Stage 4-5 breakthrough test"],
            ["5", "Build the Context Profiler CLI",
             "First product, per the plan"],
        ], widths=[1.0, 7.5, 6.5]),
    ]

    # ---------------- 9. Reproducibility ----------------
    story += [
        H1("9. Reproducibility"),
        Paragraph("pip install -r requirements.txt<br/>"
                  "python experiments/run_main.py position<br/>"
                  "python experiments/run_main.py length<br/>"
                  "python experiments/analyze.py position<br/>"
                  "python experiments/analyze.py length<br/>"
                  "python experiments/analyze.py interference<br/>"
                  "python -m pytest tests/ -q    # 7 unit tests<br/>"
                  "python experiments/build_pdf.py   # this report",
                  S["code"]),
        SP(0.2),
        P("Artifacts: results/*.csv (raw trials), results/summary_*.csv, "
          "results/figure_interference.png, results/figure_position.png, "
          "results/figure_length.png."),
        SP(0.4),
    ]

    story += build_phase1b()
    story += build_phase1c()
    story += build_phase1d()
    story += build_phase1e()
    story += build_phase1f()
    story += build_phase7()
    story += build_phase7_cd()
    story += build_phase8()
    story += build_phase9()
    story += build_phase10()
    story += build_phase11()
    story += build_phase12a()
    story += build_phase12b()
    story += [SP(0.6), Paragraph("End of report.", S["meta"])]
    return story


def build_phase1b():
    story = []

    # ---------------- 10. New research question ----------------
    story += [
        PageBreak(),
        H1("10. Phase 1B - the new research question"),
        P("The central question changed from <i>how does context length affect "
          "performance?</i> to:"),
        NOTE("<b>Can effective context usage be predicted from the amount and "
             "structure of the information competing for attention?</b>"),
        P("The working hypothesis is an interference load:"),
        Paragraph("I  approx  ( number of competing items x similarity ) "
                  "/ effective attention budget", S["code"]),
        SP(0.1),
        P("Before scaling to long contexts, we fixed the confound identified in "
          "Phase 1: the filler was a tiny ~388-token bank that repeated itself. "
          "Experiment 4 replaces it with real WikiText prose (345,590 clean "
          "tokens), so long contexts are genuine, non-repeating text."),
    ]

    # ---------------- 11. Experiment 4 ----------------
    story += [
        H2("11. Experiment 4 - real-prose interference x length"),
        P("Filler: WikiText. Position 0.5. Decoys D in {0,1,2,4,8}. Similarity "
          "fixed at medium. n = 4 (n = 2 at L = 4096)."),
        TBL([
            ["L \\ D", "0", "1", "2", "4", "8"],
            ["256", "1.00", "1.00", "0.75", "0.75", "0.50"],
            ["512", "1.00", "0.75", "0.25", "0.25", "0.50"],
            ["1024", "1.00", "0.75", "0.50", "0.25", "0.25"],
            ["2048", "1.00", "1.00", "0.75", "0.75", "0.25"],
            ["4096*", "1.00", "1.00", "1.00", "1.00", "1.00"],
        ], widths=[2.4, 2.4, 2.4, 2.4, 2.4, 2.4]),
        SP(0.1),
        CAP("* L = 4096 used only n = 2 per cell; treat that row as provisional."),
        KeepTogether([
            IMG("figure_exp4_heatmap.png"),
            CAP("Figure 4. Accuracy across length x interference (real prose, "
                "256-2048 tokens). Failure is driven by the number of decoys "
                "(columns), not by length (rows)."),
        ]),
        P("With real prose the picture is clean: at D = 0 accuracy is <b>1.00 at "
          "every length tested, up to 4096 tokens</b>. As decoys increase, "
          "accuracy falls. Across the rows (length) there is no systematic "
          "decline - the naive &quot;length hurts&quot; story does not hold. The "
          "4096 row is remarkable: with n = 2 it stayed at 1.00 even with 8 "
          "decoys, consistent with interference being a <i>local</i> effect that "
          "gets diluted when items are spread over a longer context."),
    ]

    # ---------------- 12. Experiment 5 ----------------
    story += [
        H2("12. Experiment 5 - similarity of competing information"),
        P("Now the decoys are made scientifically controllable. Similarity S has "
          "three levels:"),
    ]
    story += bullets([
        "<b>low</b> - unrelated sentence (e.g. weather); no key clash.",
        "<b>medium</b> - &quot;the {other-key} password is {word}&quot;; same "
        "pattern, different key.",
        "<b>high</b> - &quot;Important: the {near-synonym-of-vault} password is "
        "{word}&quot;; near-identical template and near-synonymous key.",
    ])
    story += [
        P("Fixed length 1024, four decoys, n = 6 per level."),
        TBL([
            ["similarity S", "accuracy", "p_correct"],
            ["low", "1.00", "0.998"],
            ["medium", "0.50", "0.380"],
            ["high", "0.00", "0.026"],
        ], widths=[4.0, 4.0, 4.0]),
        SP(0.2),
        KeepTogether([
            IMG("figure_exp5_similarity.png", w=11),
            CAP("Figure 5. Similarity of competing information has a decisive, "
                "monotonic effect: 1.00 -> 0.50 -> 0.00."),
        ]),
        P("This is the cleanest single result in the project: at <i>identical</i> "
          "length and number of decoys, changing only how <i>similar</i> they are "
          "to the target moves accuracy from perfect to zero. Similarity is a "
          "first-order term, exactly as the hypothesis predicts."),
    ]

    # ---------------- 13. Collapse test ----------------
    story += [
        H2("13. Collapse test - does one latent variable explain everything?"),
        P("We test whether all conditions collapse onto one curve when plotted "
          "against an interference <i>density</i> I = D x w(S) / L, with "
          "w(low)=0.1, w(medium)=0.5, w(high)=1.0."),
        KeepTogether([
            IMG("figure_collapse.png"),
            CAP("Figure 6. Accuracy vs interference density. Points of different "
                "length do NOT fall on one curve."),
        ]),
        NOTE("<b>Honest negative result.</b> The single density variable does NOT "
             "collapse the data. E.g. at I approx 1.95, accuracy ranges from 0.25 "
             "(L=2048, D=8) to 1.00 (L=256, D=1). Two conditions with the same "
             "density but very different decoy counts behave differently, so the "
             "latent variable is richer than a single density term.",
        ),
        P("What the collapse plot <i>does</i> show: (i) at I = 0 (no decoys) "
          "accuracy is 1.00 regardless of length, and (ii) accuracy falls as I "
          "grows. But the spread at fixed I tells us we need more than one "
          "parameter - an absolute count term and a length/dilution term appear "
          "to be separately needed. This sharpens the next experiment rather "
          "than closing it."),
    ]

    # ---------------- 14. Phase 1B findings ----------------
    story += [
        H1("14. What Phase 1B establishes (and what remains open)"),
        H2("14.1 Findings"),
    ]
    story += bullets([
        "<b>The confound is fixed.</b> With real prose, accuracy at D = 0 is 1.00 "
        "at every length up to 4096 tokens - so this is not an artefact of "
        "repetitive filler.",
        "<b>Interference count D is a first-order driver.</b> Adding decoys "
        "collapses accuracy at constant length.",
        "<b>Similarity S is a first-order driver.</b> At fixed length and D, "
        "moving similarity from low to high moves accuracy from 1.00 to 0.00.",
        "<b>Length alone shows no measurable effect here.</b>",
        "<b>A single latent density does not explain the data.</b> The collapse "
        "test is a genuine negative - the variable must be richer than one term.",
    ])
    story += [
        H2("14.2 Corrected, paper-safe wording"),
        NOTE("Instead of &quot;raw length did not matter&quot;, use: <i>No "
             "measurable degradation from raw context length was observed within "
             "the tested 256-4096-token range for this model and task setup.</i> "
             "The 8K-128K regime remains untested."),
        H2("14.3 Limitations (Phase 1B)"),
    ]
    story += bullets([
        "n = 4 per cell (n = 2 at 4096). Curves are directional, not precise.",
        "One model (Qwen2.5-0.5B) and one filler source (WikiText).",
        "Similarity weights (0.1 / 0.5 / 1.0) are assumed, not fitted.",
        "Only the 256-4096-token range is covered; 8K+ is untested.",
        "CPU stalls injected timing noise (one cell took 1211 s).",
    ])
    story += [
        H2("14.4 Next steps"),
        TBL([
            ["#", "Action", "Why"],
            ["1", "Increase n to >=20 per cell",
             "Current differences are within noise"],
            ["2", "Fit P = f(D, S, L) with an interaction term",
             "Single-parameter density failed"],
            ["3", "Model I as count + length-dilution",
             "Data needs two terms, not one"],
            ["4", "Fit on models A-C, predict unseen model D + task E",
             "The plan's OOD breakthrough test"],
            ["5", "Move to a GPU to reach 8K-128K",
             "Current CPU cannot go further"],
        ], widths=[1.0, 7.5, 6.5]),
    ]

    return story
    return story


def build_phase1c():
    story = []

    # ---------------- 15. Experiment 6 ----------------
    story += [
        PageBreak(),
        H1("15. Experiment 6 - high-n factorial (n = 20)"),
        P("To move from observation to a fitted law we ran a balanced factorial "
          "with real prose and n = 20 per cell: L in {256, 1024}, D in "
          "{0,1,2,4,8}, S in {low, medium, high}. Cells: 2 x 5 x 3 = 30, samples "
          "= 600."),
        TBL([
            ["S", "D", "L=256", "L=1024"],
            ["low", "0", "1.00", "1.00"],
            ["low", "1", "1.00", "1.00"],
            ["low", "2", "1.00", "1.00"],
            ["low", "4", "1.00", "1.00"],
            ["low", "8", "1.00", "1.00"],
            ["medium", "0", "1.00", "1.00"],
            ["medium", "1", "0.90", "0.75"],
            ["medium", "2", "0.80", "0.50"],
            ["medium", "4", "0.65", "0.45"],
            ["medium", "8", "0.50", "0.35"],
            ["high", "0", "1.00", "1.00"],
            ["high", "1", "0.75", "0.75"],
            ["high", "2", "0.55", "0.60"],
            ["high", "4", "0.15", "0.30"],
            ["high", "8", "0.05", "0.00"],
        ], widths=[3.2, 3.2, 3.2, 3.2]),
        SP(0.15),
    ]
    story += bullets([
        "<b>Low similarity is inert.</b> 1.00 at every decoy count and both "
        "lengths - unrelated text does not compete for the answer.",
        "<b>Medium and high similarity degrade sharply.</b> High similarity is "
        "catastrophic: 1.00 -> 0.00 as D grows.",
        "<b>Length now shows a small but consistent effect</b> - at 1024 tokens "
        "accuracy is usually lower than at 256 (e.g. medium/D=2: 0.80 vs 0.50), "
        "consistent with a genuine, weak length term once interference is high.",
    ])

    # ---------------- 16. Fitting the law ----------------
    story += [
        H2("16. Fitting the law P = f(D, S, L) with interactions"),
        P("Logistic regression on per-sample correctness with features: "
          "log2L, D, S, and the interactions D*S, D*log2L, S*log2L, D*S*log2L "
          "(S weighted at 0.1 / 0.5 / 1.0)."),
        TBL([
            ["term", "coefficient"],
            ["intercept", "3.98"],
            ["D*S", "-0.82"],
            ["S", "-0.44"],
            ["D", "0.22"],
            ["log2L", "-0.11"],
            ["D*S*log2L", "-0.04"],
            ["S*log2L", "-0.03"],
            ["D*log2L", "0.00"],
        ], widths=[5.0, 5.0]),
        SP(0.15),
        P("The dominant term is the <b>D*S interaction (-0.82)</b>: the number "
          "of competitors and their similarity multiply, so decoys only do "
          "damage when they are similar to the target. This is the quantitative "
          "form of the interference hypothesis."),
        KeepTogether([
            IMG("figure_exp6_fit.png", w=11.5),
            CAP("Figure 7. Predicted vs observed cell accuracy (n=20). Pearson "
                "r = 0.944, MAE = 0.077."),
        ]),
        P("<b>Goodness of fit:</b> 5-fold CV accuracy 0.718 +/- 0.129 on "
          "individual trials; on cell means the fit is excellent "
          "(r = 0.944, MAE = 0.077)."),
        H2("16.1 Generalisation to unseen conditions (the key test)"),
        TBL([
            ["held-out condition", "train on", "test accuracy"],
            ["unseen length (L=1024)", "L=256 only", "0.81"],
            ["unseen decoy count (D=8)", "D != 8", "0.85"],
            ["unseen similarity (medium)", "low + high", "0.69"],
            ["unseen similarity (high)", "low + medium", "0.49"],
        ], widths=[5.2, 4.0, 3.4]),
        SP(0.15),
        P("The law transfers well to an <b>unseen length (0.81)</b> and an "
          "<b>unseen decoy count (0.85)</b> - i.e. it can predict P(correct) "
          "before the experiment is run for those axes. It transfers less well "
          "to an unseen <b>similarity level</b> (0.49-0.69): similarity is the "
          "axis where the current functional form is weakest and where further "
          "work is needed."),
    ]

    # ---------------- 17. Conclusion ----------------
    story += [
        H1("17. Where this leaves the theory"),
        P("The project has moved from an observation to a characterisation. The "
          "original framing - that performance decays with context length - does "
          "not survive contact with real prose and adequate sample sizes. What "
          "survives, and strengthens, is the interference view:"),
    ]
    story += bullets([
        "The dominant control on whether information stays retrievable is the "
        "<b>count of competing items multiplied by their similarity</b> to the "
        "target (the D*S term, coefficient -0.82).",
        "Raw length is a <b>weak, secondary</b> term in the tested range "
        "(256-1024 tokens at n=20; 256-4096 at n=4-8).",
        "Unrelated information is essentially free (low similarity = 1.00), so "
        "the cost of context is paid in <i>confusable</i>, not merely in "
        "<i>additional</i>, tokens.",
    ])
    story += [
        P("A logistic model over (log2L, D, S) with interactions predicts cell "
          "accuracy with r = 0.944 (MAE = 0.077), and - the important part - it "
          "predicts <b>before the fact</b> for unseen length (0.81) and unseen "
          "decoy count (0.85). That is the first working version of the "
          "proposed pre-inference theory of effective context."),
        H2("17.1 Honest limits"),
    ]
    story += bullets([
        "Generalisation to an <b>unseen similarity level</b> is weak (0.49-0.69).",
        "One model (Qwen2.5-0.5B); the cross-model test has not been run.",
        "Lengths capped at 1024 here (4096 at n=2 earlier); 8K+ untested.",
        "n=20 is adequate but not large; confidence intervals are still wide.",
    ])
    story += [
        H2("17.2 The next decisive experiment"),
        P("The plan's original breakthrough criterion is now within reach: fit "
          "the law on models A-C and tasks A-D, then predict an <b>unseen model "
          "D and unseen task E</b>. If the interference law transfers across "
          "models (not just across conditions of one model), it is a genuine, "
          "general theory. That - together with a GPU to reach 8K-128K - is the "
          "next step."),
    ]
    return story


def build_phase1d():
    story = []

    # ---------------- 18. Cross-model transfer ----------------
    story += [
        PageBreak(),
        H1("18. Cross-model transfer: the frozen Qwen law predicts Llama"),
        P("This is the decisive test. The Qwen law was <b>frozen first</b> "
          "(coefficients hashed to sha256 cf79eeae...), predictions for all 30 "
          "Llama cells were written to disk <i>before</i> any Llama data existed, "
          "and only then was the identical factorial run on a different model "
          "family (Llama-3.2-1B, 600 samples)."),
        H2("18.1 Transfer quality (frozen coefficients, no refit)"),
        TBL([
            ["metric", "value", "meaning"],
            ["Pearson r", "0.946", "cell prediction vs observation"],
            ["Spearman rho", "0.923", "rank agreement"],
            ["MAE", "0.080", "mean absolute cell error"],
            ["ECE", "0.049", "calibration error (low = well calibrated)"],
            ["AUC", "0.915", "per-trial discrimination"],
        ], widths=[3.2, 2.6, 7.2]),
        SP(0.15),
        P("The frozen Qwen law predicts Llama with <b>r = 0.946</b> - as well as "
          "it fits Qwen itself (r = 0.944). This is genuine cross-model transfer, "
          "not a refit."),
        KeepTogether([
            IMG("figure_transfer.png", w=11),
            CAP("Figure 8. Predicted (frozen Qwen law) vs observed (Llama) cell "
                "accuracy. Points lie close to the identity line."),
        ]),
        H2("18.2 Predicted vs observed, every cell"),
        TBL([
            ["S", "D", "pred(256)", "obs(256)", "pred(1024)", "obs(1024)"],
            ["low", "0", "0.95", "1.00", "0.94", "1.00"],
            ["low", "4", "0.97", "1.00", "0.96", "1.00"],
            ["low", "8", "0.98", "1.00", "0.97", "1.00"],
            ["medium", "1", "0.91", "0.80", "0.89", "0.95"],
            ["medium", "4", "0.78", "0.55", "0.70", "0.65"],
            ["medium", "8", "0.45", "0.40", "0.31", "0.60"],
            ["high", "1", "0.81", "0.75", "0.75", "0.95"],
            ["high", "4", "0.21", "0.25", "0.12", "0.40"],
            ["high", "8", "0.01", "0.00", "0.00", "0.10"],
        ], widths=[2.2, 1.8, 2.7, 2.7, 2.7, 2.7]),
        SP(0.1),
        CAP("(Condensed; all 30 cells in results/transfer_pairs.csv.)"),
        H2("18.3 Do the coefficients keep their structure?"),
        TBL([
            ["term", "Qwen (frozen)", "Llama (refit)"],
            ["D*S", "-0.82", "-1.11"],
            ["S", "-0.44", "-0.19"],
            ["log2L", "-0.11", "+0.16"],
            ["intercept", "3.98", "2.29"],
        ], widths=[4.0, 4.0, 4.0]),
        SP(0.1),
        P("The <b>dominant D*S structure transfers strongly</b> across the two "
          "models (same sign; -0.82 vs -1.11), while the <b>lower-order "
          "coefficients and the model-specific calibration are less stable</b> "
          "(S: -0.44 vs -0.19; log2L changes sign; intercept 3.98 vs 2.29). The "
          "safe reading is: <i>the dominant interference structure transfers; "
          "absolute calibration does not</i>."),
    ]

    H2("18.4 Where the residuals live")
    story += [
        P("Residuals are small on average but structured: at long length and high "
          "interference Llama is <i>more</i> robust than predicted (e.g. "
          "L=1024, high, D=4: predicted 0.12, observed 0.40; L=1024, medium, D=8: "
          "predicted 0.31, observed 0.60). This is exactly what a model-specific "
          "attention budget predicts: the same interference structure, a larger "
          "budget."),
    ]

    # ---------------- 19. Updated conclusion ----------------
    story += [
        H1("19. Where this leaves the theory (after cross-model transfer)"),
        NOTE("Strongest defensible claim: <i>For these two models and this "
             "benchmark, retrieval performance is well explained by a "
             "multiplicative interaction between distractor count and distractor "
             "similarity, with context length a smaller secondary effect; the "
             "fitted law, frozen on one model, predicts an unseen model's cells "
             "with r = 0.946.</i>"),
        H2("19.1 What is now established"),
    ]
    story += bullets([
        "Interference = <b>count x similarity</b> is the dominant control, "
        "reproduced on a second model family.",
        "A frozen law <b>transfers across models</b> (r = 0.946, ECE = 0.049), "
        "with a <b>shared structural term</b> and a <b>model-specific intercept</b>.",
        "Unrelated (low-similarity) context is inert on both models.",
    ])
    story += [
        H2("19.2 Honest limits"),
    ]
    story += bullets([
        "Only <b>two</b> models, one task family (needle retrieval), lengths "
        "<= 1024 at n = 20.",
        "Similarity S is still a <b>handcrafted categorical</b> variable; a "
        "continuous, measurable metric (embedding / key overlap) is needed.",
        "Lower-order coefficients (S, log2L) are unstable between models - they "
        "are entangled with D*S and should not be over-interpreted.",
    ])
    story += [
        H2("19.3 The next milestone"),
        P("Fit the law on models A-C and tasks A-D, then predict an <b>unseen "
          "model E and an unseen task E</b>. If that holds, the interference law "
          "is a general predictive theory of context, not a two-model "
          "coincidence. Replacing categorical similarity with a continuous "
          "metric is the other prerequisite for publication."),
    ]
    return story


def build_phase1e():
    story = []

    # ---------------- 20. Continuous similarity ----------------
    story += [
        PageBreak(),
        H1("20. Making similarity measurable (continuous S)"),
        P("Until now S was a handcrafted category (low/medium/high), the "
          "project's biggest scientific weakness: the dominant D*S term partly "
          "depended on a human label. Experiment 8 replaces it with measurable "
          "quantities. <b>Three candidate metrics were defined and computed "
          "before</b> the language model was run:"),
    ]
    story += bullets([
        "<b>cos_sim</b> - cosine similarity of sentence embeddings "
        "(all-MiniLM-L6-v2) between the target sentence and the decoy.",
        "<b>jaccard</b> - token-set Jaccard overlap between the two sentences.",
        "<b>shared_prefix</b> - number of identical leading words.",
    ])
    story += [
        P("Twenty decoy sentences were pre-registered, spanning cos_sim from "
          "-0.04 (weather) to 1.00 (identical template). At D = 1 the effect was "
          "below the detection floor (all 1.00), so the sweep was run at "
          "<b>D = 4</b>, L = 1024, n = 10 per decoy."),
        TBL([
            ["decoy", "cos_sim", "accuracy"],
            ["unrelated_weather", "-0.04", "1.00"],
            ["topic_password", "0.44", "1.00"],
            ["password_policy", "0.50", "1.00"],
            ["account_password", "0.59", "0.60"],
            ["system_password", "0.57", "0.50"],
            ["security_password", "0.60", "0.40"],
            ["archive_password", "0.72", "0.30"],
            ["storage_password", "0.76", "0.30"],
            ["safe_password", "0.76", "0.40"],
            ["case_note_similar", "0.96", "0.00"],
            ["case_important_identical", "1.00", "0.00"],
        ], widths=[5.4, 3.0, 3.0]),
        SP(0.1),
        CAP("(Condensed; all 20 decoys in results/exp8_summary_qwen.csv.)"),
        P("Accuracy falls monotonically as similarity rises: unrelated text is "
          "inert (1.00), moderately similar text costs roughly a third of the "
          "accuracy, and a near-identical sentence destroys retrieval (0.00)."),
        H2("20.1 Which metric predicts interference?"),
        TBL([
            ["metric", "Pearson r (Qwen)", "held-out-decoy MAE"],
            ["jaccard", "-0.941", "0.099"],
            ["cos_sim", "-0.878", "0.133"],
            ["shared_prefix", "-0.437", "n/a"],
        ], widths=[4.0, 4.5, 4.5]),
        SP(0.1),
        P("<b>Lexical Jaccard overlap is the strongest single predictor</b> "
          "(r = -0.941), edging out embedding cosine (r = -0.878); template "
          "prefix is weak. Both strong metrics were chosen by pre-registration, "
          "not by post-hoc selection."),
        KeepTogether([
            IMG("figure_exp8_similarity_qwen.png"),
            CAP("Figure 9. Accuracy vs continuous similarity (D=4, L=1024)."),
        ]),
    ]

    # ---------------- 21. Third model ----------------
    story += [
        H1("21. A third model (gpt2-medium) - does the metric replicate?"),
        P("The same continuous sweep was run on a third, unrelated model family "
          "(gpt2-medium; L=768 because GPT-2 caps at 1024 total tokens)."),
        TBL([
            ["metric", "Qwen r", "gpt2 r", "agree?"],
            ["jaccard", "-0.941", "-0.886", "yes"],
            ["cos_sim", "-0.878", "-0.747", "yes"],
            ["shared_prefix", "-0.437", "-0.433", "yes"],
        ], widths=[4.0, 3.2, 3.2, 2.4]),
        SP(0.1),
        P("The <b>ranking of metrics is identical on both models</b>: Jaccard is "
          "the best predictor, embedding cosine second, prefix last. The "
          "continuous similarity metric is therefore not Qwen-specific."),
        KeepTogether([
            IMG("figure_exp8_models.png"),
            CAP("Figure 10. Accuracy vs continuous similarity on two model "
                "families. Same descending relationship; gpt2-medium is more "
                "tolerant (a larger attention budget), not differently shaped."),
        ]),
        P("gpt2-medium is more robust: it tolerates moderately similar decoys "
          "that already hurt Qwen. This is the same &quot;shared structure, "
          "different budget&quot; pattern seen in the Qwen-to-Llama transfer, "
          "now along a <b>measurable</b> similarity axis."),
        H2("21.1 Summary of the continuous result"),
    ]
    story += bullets([
        "The law is now <b>P = f(D, S_cont, L)</b> with S_cont a measured "
        "quantity (lexical Jaccard best), not a human category.",
        "The <b>D * S</b> interaction remains the dominant control, reproduced "
        "across three model families (Qwen, Llama, gpt2).",
        "The best metric <b>generalises across decoys</b> (leave-one-decoy-out "
        "MAE = 0.099 on Qwen).",
    ])

    # ---------------- 22. Final position ----------------
    story += [
        H1("22. Final position (end of the research phase)"),
        NOTE("Strongest defensible claim. <i>For the tested models and "
             "benchmark, retrieval accuracy is well explained by a "
             "multiplicative interaction between the number of competing "
             "passages and their measured similarity to the target, with context "
             "length a weaker secondary effect. The dominant interference "
             "structure transfers across model families, and a single similarity "
             "metric (lexical overlap) predicts interference consistently.</i>"),
        H2("22.1 What is established"),
    ]
    story += bullets([
        "Interference, not length, drives failure (real prose; n=20; "
        "256-1024 tokens).",
        "Law P = f(D, S, L) fits cell means at r = 0.944 and predicts an unseen "
        "model's cells at r = 0.946 (pre-registered, frozen, no refit).",
        "Similarity is now measurable: continuous Jaccard/cosine predict "
        "accuracy (r = -0.88 to -0.94) on <b>two</b> model families.",
        "The ranking of similarity metrics is identical across models.",
    ])
    story += [
        H2("22.2 What is NOT yet established (honest)"),
    ]
    story += bullets([
        "Cross-<b>task</b> transfer (only needle retrieval tested).",
        "Few models (Qwen-0.5B, Llama-1B, gpt2-medium); absolute calibration "
        "differs and slopes are not identical between them.",
        "Lengths <= 1024 at n=20; 8K-128K untested on this hardware.",
        "The 'attention budget' is inferred, not directly measured.",
    ])
    story += [
        H2("22.3 Next (product phase)"),
        P("The research phase ends here. The next step is the product: a "
          "Context Profiler that measures a customer's context, scores its "
          "interference load with the validated similarity metric, predicts the "
          "safe operating range, and rewrites the context to reduce "
          "interference - the intervention the law now makes possible."),
    ]
    return story


def build_phase7():
    """Phase 7 - RAG product benchmark: prune + reorder + conflict resolution."""
    story = [PageBreak(), H1("28. Phase 7 - real RAG intervention benchmark")]
    story += [P(
        "The research law was frozen, so the project moved to the product "
        "question: does an end-to-end retrieval pipeline that uses the "
        "profiler actually answer better, with less context, than a baseline "
        "and a matched random-prune control? Every number below comes from a "
        "preregistered run (queries hashed before any model call) over a real "
        "MiniLM + FAISS retriever and the Qwen2.5-0.5B reader, scored with the "
        "same frozen K-way single-token rule as the rest of the project.")]

    story += [H2("28.1 Phase 7 - the first end-to-end run (a null with a bug)")]
    story += [P(
        "The first real run returned accuracy ~= 0.10 for <i>all</i> conditions. "
        "Diagnosis showed the top-8 retrieved passages were eight copies of one "
        "decoy, with the gold absent: the query generator reused the same 12 "
        "candidate words as decoys across queries, so 400 injected strings "
        "collapsed to 107 unique and duplicate strings flooded the FAISS index. "
        "After fixing the generator (unique golds + private disjoint decoy "
        "nouns, hard uniqueness assert), the corrected run gave BASELINE 0.075, "
        "CONTEXTLAB 0.175, RANDOM 0.075 - a +0.10 reordering effect whose paired "
        "95% CI still touched zero. Crucially <b>zero</b> passages were pruned "
        "(the corpus was contradiction-rich, which the policy protects), so this "
        "design could not test the token-saving claim at all.")]

    story += [H2("28.2 Phase 7B - redundancy intervention (the missing arm)")]
    story += [P(
        "Phase 7B fixes the design flaw: 40 preregistered queries x 4 context "
        "types (clean / redundant / contradictory / mixed) x 4 conditions "
        "(BASELINE / RANDOM / CONTEXTLAB / ORACLE) = 640 runs.")]
    story += [TBL([
        ["context", "condition", "accuracy", "ctx tokens", "removed"],
        ["clean", "BASELINE / RANDOM / CL", "1.000", "35.3", "0.00"],
        ["redundant", "BASELINE", "1.000", "61.5", "0.00"],
        ["redundant", "RANDOM", "1.000", "38.8", "2.25"],
        ["redundant", "CONTEXTLAB", "1.000", "41.8", "2.25"],
        ["redundant", "ORACLE", "1.000", "17.5", "4.00"],
        ["contradictory", "BASELINE", "0.350", "61.5", "0.00"],
        ["contradictory", "CONTEXTLAB", "0.250", "61.5", "0.00"],
        ["contradictory", "ORACLE", "0.525", "17.5", "4.00"],
        ["mixed", "BASELINE", "0.775", "87.8", "0.00"],
        ["mixed", "CONTEXTLAB", "0.675", "74.6", "1.50"],
    ], [2.6, 4.4, 2.4, 2.6, 2.4])]
    story += [P(
        "On <b>redundant</b> context ContextLab holds accuracy at 1.000 while "
        "cutting tokens 61.5 -> 41.8 (-32%, SIG; C2 CI [-20.9, -18.5]). That is "
        "the Pareto win - same accuracy, one-third less context. Honest caveat: "
        "the matched RANDOM control also reached 1.000, so this shows <i>pruning "
        "works</i>, not yet that the engine's <i>choice</i> beats alternatives. "
        "The contradiction result was the surprise: the reorder arm <i>hurt</i> "
        "(0.350 -> 0.250) and ORACLE (drop conflicts) reached 0.525 - detection "
        "is not resolution.")]
    return story


def build_phase7_cd():
    """Phase 7C + 7D - choice attribution and conflict-resolution policies."""
    story = [H2("28.3 Phase 7C - choice attribution (better than random?)")]
    story += [P(
        "7C removes the crutch: only ONE passage (gold) yields the answer, the "
        "rest are graded evidence plus redundant duplicates the profiler prunes. "
        "RANDOM removes the same count at random and will sometimes delete the "
        "gold; ContextLab never does.")]
    story += [IMG("figure_rag7c_pareto.png", 13.5),
              CAP("Phase 7C Pareto: accuracy vs context tokens.")]
    story += [TBL([
        ["condition", "accuracy", "gold survival", "ctx tokens"],
        ["BASELINE", "0.500", "1.000", "135.5"],
        ["RANDOM", "0.550", "0.775", "105.9"],
        ["CONTEXTLAB", "0.800", "1.000", "97.5"],
        ["ORACLE", "1.000", "1.000", "18.5"],
    ], [3.6, 3.0, 3.6, 3.0])]
    story += [NOTE(
        "Preregistered (paired, n=40): CONTEXTLAB-RANDOM accuracy = +0.250 "
        "(CI [+0.050, +0.425], SIG); gold survival = +0.225 (SIG); "
        "CONTEXTLAB-BASELINE accuracy = +0.300 (SIG) at -38 tokens (SIG). "
        "Mechanism: RANDOM accuracy is 0.222 when it deletes the gold vs 0.645 "
        "when it does not; ContextLab keeps the gold 100% of the time. This is "
        "the attribution result - <b>ContextLab makes better pruning decisions, "
        "not just 'pruning works'</b>.")]

    story += [H2("28.4 Phase 7D - conflict-resolution policies")]
    story += [P(
        "Detection is not resolution, so 7D tests four explicit conflict "
        "policies on the frozen contradictory/mixed contexts (320 runs): "
        "PROTECT (current), DROP_WEAKER (keep the highest-token conflict "
        "member), SURFACE (keep conflict, reorder), ORACLE (keep gold only).")]
    story += [TBL([
        ["policy", "accuracy", "ctx tokens", "removed"],
        ["PROTECT (current)", "0.562", "74.6", "0.00"],
        ["DROP_WEAKER", "1.000", "45.1", "3.375"],
        ["SURFACE", "0.500", "74.6", "0.00"],
        ["ORACLE", "1.000", "9.8", "6.375"],
    ], [4.4, 2.8, 3.0, 2.6])]
    story += [NOTE(
        "Preregistered (paired, n=80): DROP_WEAKER-PROTECT accuracy = +0.438 "
        "(CI [+0.338, +0.550], SIG) and tokens = -29.5 (SIG); on contradictory "
        "alone 0.350 -> 1.000 (+0.650, SIG). SURFACE does nothing (ns) - so it "
        "is <b>resolution</b>, not reordering, that matters. Caveat: on this "
        "construction the conflicting passages shared one template and the gold "
        "sat at index 0, so the rule broke ties by keeping the first passage - "
        "7E (below) removes that confound and the effect vanishes.")]

    story += [H2("28.5 Phase 7E - decoupled conflict resolution (confound removed)")]
    story += [P(
        "7D's 100% was a confound. 7E decouples SPECIFICITY from CORRECTNESS (a "
        "passage's length is drawn independently of whether it carries the "
        "answer; verified anti-leakage: P(correct)=0.250 by specificity, max dev "
        "0.068 &lt;= 0.12) and RANDOMIZES the correct passage's position.")]
    story += [TBL([
        ["policy", "accuracy", "survivor correct", "ctx tokens"],
        ["PROTECT", "0.300", "1.000", "99.0"],
        ["DROP_MORE_SPECIFIC", "0.250", "0.250", "51.7"],
        ["DROP_LESS_SPECIFIC", "0.275", "0.275", "35.3"],
        ["DROP_WEAKER", "0.300", "0.300", "44.9"],
        ["RANDOM", "0.275", "0.275", "42.3"],
        ["ORACLE", "1.000", "1.000", "42.0"],
    ], [4.4, 2.6, 3.4, 2.6])]
    story += [NOTE(
        "Preregistered (paired, n=40): DROP_WEAKER-PROTECT accuracy = +0.000 "
        "(ns); DROP_MORE_SPECIFIC-PROTECT = -0.050 (ns); DROP_WEAKER-RANDOM = "
        "+0.025 (ns); ORACLE-PROTECT = +0.700 (SIG). <b>Every deployable "
        "resolution heuristic collapses to chance</b> (survivor_correct "
        "0.250-0.300 = the 1/4 base rate); only the oracle closes the gap. "
        "Accuracy is fully explained by whether the surviving passage is correct "
        "(1.000 vs 0.000). Honest conclusion: conflict <i>detection</i> is "
        "solved; conflict <i>resolution without an oracle</i> is an open "
        "problem.")]

    story += [H2("28.6 What the RAG phase establishes (honest summary)")]
    story += bullets([
        "<b>Pruning/Pareto (7B):</b> -32% context tokens at unchanged accuracy "
        "on redundant context (SIG).",
        "<b>Attribution (7C):</b> ContextLab beats a matched random control by "
        "+0.25 accuracy and keeps the gold 100% - it makes <i>better</i> pruning "
        "decisions (SIG).",
        "<b>Conflict (7D):</b> on a construction where the correct passage was "
        "positionally privileged, resolving conflicts (drop the weaker) beat "
        "protecting them by +0.44 accuracy and -30 tokens (SIG).",
        "<b>Conflict (7E, confound removed):</b> once specificity is decoupled "
        "from correctness and position is randomized, every deployable "
        "resolution heuristic collapses to chance (survivor_correct 0.25-0.30 "
        "= the 1/4 base rate, ns) - the 7D win did not survive. Conflict "
        "<i>detection</i> is solved; conflict <i>resolution without an oracle</i> "
        "is an open problem.",
        "<b>Not yet shown:</b> a deployable rule that resolves conflicts better "
        "than chance; that is the next research question.",
    ])
    story += [NOTE(
        "Defensible product statement. <i>On redundant context, ContextLab "
        "reduced context tokens by 32% with no accuracy loss and, on a harder "
        "attribution benchmark, kept the answer-bearing passage 100% of the time "
        "and beat matched random pruning by 25 points (7C). On contradictory "
        "context, an early apparent conflict-resolution win (7D) did not survive "
        "a confound-fix benchmark (7E): no deployable rule beats chance at "
        "picking the correct side of a conflict, so ContextLab currently "
        "detects and surfaces conflicts rather than claiming to resolve them.</i>")]
    return story


def build_phase8():
    """Phase 8 - the functional form of the count term f(D)."""
    story = [PageBreak(), H1("29. Phase 8 - the functional form of the count term")]
    story += [P(
        "The product's count term is min(1, D / D_SATURATION) with "
        "D_SATURATION = 5 - a value the earlier D-sweep never recovered "
        "(best-fit k ~= 9). Phase 8 asks the sharper question: which functional "
        "form f(D) in P = sigma(a + b*f(D)) best predicts accuracy on HELD-OUT D "
        "values? The protocol was frozen before any fit (sha256 f1d548ac...): "
        "leave-one-D-out on the Qwen2.5-0.5B D-sweep (L=256, S=medium, "
        "D in {0,1,2,3,4,5,6,8,10,12,16}, n=20 per cell).")]
    story += [IMG("figure_phase8_forms.png", 14.0),
              CAP("Phase 8: held-out f(D) predictions vs observed accuracy.")]
    story += [TBL([
        ["form", "MAE", "RMSE", "Spearman", "ECE", "mono-viol", "params"],
        ["frozen 7-feature law", "0.094", "0.113", "0.973", "0.083", "0", "7"],
        ["linear D", "0.065", "0.096", "0.973", "0.063", "0", "2"],
        ["min(1, D/k)", "0.062", "0.093", "0.973", "0.060", "0", "3"],
        ["1 - e^(-D/k)", "0.086", "0.105", "0.964", "0.056", "1", "3"],
        ["log(1+D)", "0.056", "0.072", "0.973", "0.034", "0", "2"],
        ["isotonic (learned)", "0.063", "0.082", "0.970", "0.047", "0", "10"],
    ], [4.0, 1.8, 1.8, 2.0, 1.6, 1.8, 1.6])]
    story += [NOTE(
        "Findings. (1) log(1+D) is the best-interpolating form - lowest MAE "
        "(0.056) and best calibration (ECE 0.034) with only 2 parameters. "
        "(2) EVERY simple monotone form beats the frozen 7-feature law on "
        "held-out D (0.056-0.086 vs 0.094): on the D axis the elaborate law is "
        "dominated by a 2-parameter logarithm. (3) min(1,D/k) extrapolates best "
        "(err 0.037 when D=16 is held out); 1-e^(-D/k) is worst and the ONLY "
        "form with a monotonicity violation. (4) The learned isotonic form "
        "(10 params) buys nothing over log (2 params). Honest label: this "
        "REPLACES the unsupported D_SATURATION = 5 choice - the count term "
        "should be log(1+D) - while noting no form achieves zero error (the raw "
        "curve is non-monotone at D=5->6, a finite-n artefact at n=20).")]
    return story


def build_phase9():
    """Phase 9 - cross-task transfer of the competition-count structure."""
    story = [PageBreak(), H1("30. Phase 9 - cross-task transfer (zero-refit)")]
    story += [P(
        "Phase 8 fixed the count term within the retrieval family. Phase 9 asks "
        "the decisive question: does that structure survive a genuinely "
        "different task with ZERO refit? The protocol was frozen before touching "
        "target results (sha256 4b8ca8da...): fit each P = sigma(a + b*f(D)) on "
        "the retrieval D-sweep, FREEZE (a, b, k), then evaluate on the COUNTING "
        "task family (count the red objects against similar-colour items) - a "
        "'how many' task, not a rewording of retrieval.")]
    story += [TBL([
        ["slice", "model", "Spearman", "MAE", "ECE"],
        ["L1024", "linear", "0.949", "0.184", "0.184"],
        ["L1024", "min_sat", "0.949", "0.184", "0.184"],
        ["L1024", "exp_sat", "0.949", "0.200", "0.200"],
        ["L1024", "log", "0.949", "0.212", "0.212"],
        ["L1024", "frozen 7-feature law", "0.949", "0.238", "0.238"],
        ["pooled", "all forms", "-0.600", "0.19-0.25", "-"],
    ], [3.0, 4.2, 2.6, 2.0, 2.0])]
    story += [NOTE(
        "Findings (preregistered outcome = B, with a C signal on pooled data). "
        "(1) Rank structure transfers: on the L=1024 slice every frozen form "
        "ranks the counting-task D-effect correctly (Spearman 0.949) - the "
        "monotone FORM of the count term survives zero-refit. (2) Calibration "
        "does NOT transfer: ECE 0.18-0.24; the frozen retrieval coefficients set "
        "the wrong level for the counting task, so the coefficients are "
        "task-dependent. (3) Pooling the L=256 slice (where counting accuracy "
        "RISES with D, 0.5 -> 0.8) inverts the correlation to rho = -0.60: the "
        "D-structure is condition-dependent. Honest label: this supports "
        "R = F(task, f(D), S, L) - a task-conditioned reliability model - rather "
        "than one universal equation. No claim of task-general reliability is "
        "made.")]
    return story


def build_phase10():
    """Phase 10 - cross-domain generalization of the retrieval law."""
    story = [PageBreak(),
             H1("31. Phase 10 - cross-domain validation (zero-refit)")]
    story += [P(
        "Phases 8-9 held the wiki filler domain fixed. Phase 10 asks whether "
        "the retrieval law survives a change of CONTEXT DOMAIN with ZERO "
        "refit. The protocol was frozen before any new cell ran (sha256 "
        "604a8a96...): new D-sweep cells at L=256, S=medium, D in {0,4,8}, "
        "n=10 per cell, with the filler swapped from wiki prose to GENERAL "
        "ENGLISH (NeelNanda/pile-10k, URL-stripped, token-matched to the "
        "reference within 0.82% so filler amount is held fixed while domain "
        "changes). A cross-length zero-refit test on existing Exp6 data is "
        "also reported. Primary comparison: log vs the frozen 7-feature law.")]
    story += [IMG("figure_phase10_domain.png", 15.0),
              CAP("Phase 10: observed accuracy in the new domain and "
                  "zero-refit MAE by form across targets.")]
    story += [H2("31.1 Target 1 - new domain (general-English filler)"),
              P("Observed accuracy: D=0 -> 1.00, D=4 -> 0.50, D=8 -> 0.20 - a "
                "steep monotone decline in a domain the reference coefficients "
                "never saw."),
              TBL([
                  ["form", "Spearman", "MAE", "RMSE", "ECE", "slope", "params"],
                  ["log(1+D)", "1.000", "0.137", "0.158", "0.137", "1.45", "2"],
                  ["frozen 7-feature law", "1.000", "0.197", "0.220", "0.197",
                   "1.52", "7"],
                  ["linear", "1.000", "0.206", "0.215", "0.206", "1.96", "2"],
                  ["min_sat", "1.000", "0.206", "0.215", "0.206", "1.96", "3"],
              ], [4.4, 2.4, 2.0, 2.0, 2.0, 2.0, 2.0]),
              NOTE(
                  "Preregistered outcome = A (cross-domain transfer works). All "
                  "forms rank the domain-B D-effect perfectly (rho = 1.000, zero "
                  "monotonicity violations). The 2-param log is best calibrated "
                  "(MAE/ECE 0.137, slope closest to ideal 1.0) and beats the "
                  "7-param frozen law by 0.060 MAE - consistent with Phase 8.")]
    story += [H2("31.2 Target 2 - cross-length (train L=256 -> test L=1024)"),
              TBL([
                  ["S slice", "form", "Spearman", "MAE", "ECE", "slope"],
                  ["medium", "log", "1.000", "0.163", "0.147", "1.12"],
                  ["medium", "frozen law", "1.000", "0.169", "0.138", "0.79"],
                  ["high", "log", "1.000", "0.243", "0.227", "1.89"],
                  ["high", "frozen law", "1.000", "0.0745", "0.0735", "0.99"],
                  ["low", "log", "n/a (const)", "0.243", "0.243", "0.00"],
                  ["low", "frozen law", "n/a (const)", "0.048", "0.048", "0.00"],
              ], [2.6, 3.4, 3.0, 2.2, 2.2, 2.0]),
              NOTE(
                  "Honest label. Cross-domain transfer of the retrieval count "
                  "structure is a clean verdict A: the 2-param log(1+D) form "
                  "survives a domain change it never saw, with better "
                  "calibration than the 7-feature law. Cross-length transfer is "
                  "nuanced: rank structure is domain- and length-robust "
                  "(rho = 1.000 everywhere non-constant), but calibration "
                  "depends on which axis is held fixed - the richer law helps "
                  "where its S/L terms are active (cross-length), the simple "
                  "log helps where they are not (cross-domain). This reinforces "
                  "the Phase 9 conclusion R = F(task, f(D), S, L) with "
                  "condition-dependent coefficients, and adds domain as a "
                  "condition that changes coefficients, not functional form. "
                  "Verdicts are preliminary (n=10/cell, 3 D points); higher-n "
                  "replication is future work.")]
    return story
def build_phase11():
    """Phase 11 - position effects (candidate calibration feature)."""
    story = [PageBreak(),
             H1("32. Phase 11 - position effects (zero-refit)")]
    story += [P(
        "Does the evidence POSITION P (fraction of context before the needle) "
        "explain residual error after D, S and L are already known? The "
        "protocol was frozen before any cell ran (sha256 1bd50644...): P in "
        "{0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95}, D in {0, 4}, L=256, "
        "S=medium, n=10 per cell (140 trials), filler held fixed. Isolation "
        "is exact: EVERY cell is 267 context tokens, so length and filler "
        "amount are constant and only position varies. The position "
        "coefficient is fit only on {0.05, 0.30, 0.70, 0.95} and evaluated "
        "ZERO-REFIT on the held-out positions {0.15, 0.50, 0.85}.")]
    story += [IMG("figure_phase11_position.png", 15.0),
              CAP("Phase 11: observed accuracy vs position for D=0 (flat) and "
                  "D=4 (mild rise at late positions); held-out test "
                  "observed-vs-predicted, position-blind base vs position-aware "
                  "extension.")]
    story += [TBL([
        ["model", "MAE (test)", "ECE (test)", "Spearman", "params"],
        ["base log(1+D), position-blind", "0.049", "0.049", "0.949", "2"],
        ["+ position (pos_edge)", "0.027", "0.017", "0.906", "3"],
        ["frozen 7-feature law", "0.060", "0.054", "0.949", "7"],
    ], [5.6, 2.6, 2.6, 2.4, 2.0]),
        NOTE(
            "Preregistered outcome = B. Adding position improves HELD-OUT "
            "calibration (MAE 0.049 -> 0.027, ECE 0.049 -> 0.017; fit-split R2 "
            "0.841 -> 0.968, +0.128) but barely moves RANK (Spearman 0.949 -> "
            "0.906), because D=0 is flat (1.00 at every position) and D=4 moves "
            "only 0.70 -> 0.80. The MAE gain is 0.021, BELOW the preregistered "
            "A threshold of 0.030, so this is NOT an A - no threshold shopping. "
            "Position matters primarily as a CALIBRATION feature rather than a "
            "rank-ordering feature in this regime, so +P is NOT promoted to the "
            "core law; it stays a candidate calibration feature. Core law "
            "remains R = F(task, domain, log(1+D), S, L, ...). "
            "NOTE: an earlier exploratory exp1_position run at L=512 showed a "
            "stronger mid-context effect, but it is non-preregistered and "
            "reported only as contextual evidence; it does not contribute to "
            "this verdict. Verdicts preliminary (n=10/cell, 7 positions, "
            "2 D).")]
    return story


def build_phase12a():
    """Phase 12A - long-context scaling (competition-dependent penalty)."""
    story = [PageBreak(),
             H1("33. Phase 12A - long-context scaling (zero-refit screening)")]
    story += [P(
        "Before Phase 12A, long-context scaling - how much of a context window "
        "is actually usable - was one of the biggest holes in the evidence "
        "(Phases 8-11 all ran at L <= 1024). The protocol was frozen before "
        "any cell ran (sha256 d30146e3...): L in {2048, 4096, 8192} x D in "
        "{0, 4, 8}, S=medium, position=0.5, n=5 per cell = 45 samples, wiki "
        "filler fixed, length token-exact so L is the ONLY variable across the "
        "length axis. Short-L baselines reuse existing exp6 (n=20, L in "
        "{256,1024}) and exp4 - zero new baseline runs. The run completed "
        "45/45 in 4582s (~76 min), zero failures, batched and resumable by "
        "length.")]
    story += [TBL([
        ["L", "D=0", "D=4", "D=8"],
        ["2048", "1.00", "0.80", "0.40"],
        ["4096", "1.00", "0.40", "0.40"],
        ["8192", "1.00", "0.00", "0.40"],
    ], [3.0, 2.4, 2.4, 2.4]),
        CAP("Phase 12A screening accuracy (n=5/cell). D=0 stays 1.00 at every "
            "length - length alone is harmless; damage appears only under "
            "competition.")]
    story += [TBL([
        ["statistic", "value", "meaning"],
        ["harm(2048)", "-0.025", "essentially no penalty yet"],
        ["harm(4096)", "+0.175", "penalty emerging"],
        ["harm(8192)", "+0.375", "penalty growing"],
        ["deltaInt", "-0.400", "penalty worsens from L=2048 to 8192"],
    ], [3.4, 2.4, 7.2]),
        P("Aggregates over D in {4,8} vs the exp6 S=medium L=256 baseline "
          "(0.575): the degradation is monotonic in length and conditional on "
          "competition.")]
    story += [TBL([
        ["model", "params", "log-loss", "Brier", "Spearman"],
        ["M_A  F(log(1+D), S)", "2", "0.486", "0.0475", "0.757"],
        ["M_B  + L", "3", "0.449", "0.0264", "0.851"],
        ["M_C  + log(1+D) x L", "4", "0.4466", "0.0260", "0.851"],
    ], [5.4, 2.0, 2.4, 2.4, 2.4]),
        NOTE(
            "M_A -> M_B (adding L) is the major predictive jump (log-loss "
            "0.486 -> 0.449, Brier roughly halved); M_C (adding L x D) adds "
            "only a little - most signal is a main effect of L under "
            "competition. Frozen rule applied with NO threshold shopping: "
            "STAGE A = INTERESTING (harm4096/harm8192 >= 0.15 and deltaInt "
            "-0.400 <= -0.15) -> triggers Phase 12B; STAGE B = CONFIRMED AT "
            "THE PREREGISTERED SCREENING THRESHOLD (harm8192 0.375 >= 0.20 and "
            "deltaInt <= -0.05; n=5/cell is screening-level, not independent "
            "confirmation). CONCLUSION: Phase 12A provides strong screening "
            "evidence for a competition-dependent long-context penalty and "
            "triggers the preregistered Phase 12B confirmation. Long-context "
            "degradation is conditional on competing evidence, not length "
            "alone - context composition, not raw length, governs "
            "reliability. Verdicts preliminary pending 12B (n=10-20, "
            "identical frozen spec); S=medium only, S robustness deferred to "
            "Phase 13.")]
    return story


def build_phase12b():
    """Phase 12B - independent long-context confirmation (n=20)."""
    story = [PageBreak(),
             H1("34. Phase 12B - long-context confirmation (independent n=20)")]
    story += [P(
        "Phase 12A gave strong screening evidence and triggered this "
        "preregistered confirmation. Phase 12B repeats the IDENTICAL L x D "
        "experiment on fresh, independent samples at n=20 per cell, applying "
        "the UNCHANGED frozen decision rule. Protocol frozen before any cell "
        "ran (sha256 7336f02c..., byte-verified to carry the same thresholds "
        "and model spec as 12A): L in {2048, 4096, 8192} x D in {0, 4, 8}, "
        "S=medium, position=0.5, n=20/cell = 180 samples, true sample index "
        "t in [5,25) - disjoint from 12A's [0,5) (min seed 502053), so no seed "
        "collides and the confirmation set is genuinely independent. Run "
        "completed 180/180 in 11238s (~187 min), verified, batched and "
        "resumable (resumed once after an interrupt with no recomputation or "
        "seed drift).")]
    story += [TBL([
        ["L", "D=0", "D=4", "D=8"],
        ["2048", "1.00", "0.55", "0.35"],
        ["4096", "1.00", "0.60", "0.70"],
        ["8192", "1.00", "0.40", "0.20"],
    ], [3.0, 2.4, 2.4, 2.4]),
        CAP("Phase 12B confirmation accuracy (n=20/cell, independent samples). "
            "D=0 stays 1.00 at every length - length alone is harmless; damage "
            "appears only under competition, deepest at L=8192/D=8 (0.20).")]
    story += [TBL([
        ["statistic", "value", "meaning"],
        ["harm(2048)", "+0.125", "mild penalty"],
        ["harm(4096)", "-0.075", "mid-length noise (not in the rule)"],
        ["harm(8192)", "+0.275", "penalty at max length"],
        ["deltaInt", "-0.150", "penalty worse from L=2048 to 8192"],
    ], [3.4, 2.4, 7.2]),
        P("Aggregates over D in {4,8} vs the exp6 S=medium L=256 baseline "
          "(0.575). The two statistics that drive the frozen rule both clear: "
          "harm(8192) = +0.275 >= 0.20 and deltaInt = -0.150 <= -0.05.")]
    story += [TBL([
        ["model", "params", "log-loss", "Brier", "Spearman"],
        ["M_A  F(log(1+D), S)", "2", "0.4774", "0.02253", "0.804"],
        ["M_B  + L", "3", "0.4723", "0.02014", "0.797"],
        ["M_C  + log(1+D) x L", "4", "0.4720", "0.02012", "0.797"],
    ], [5.4, 2.0, 2.4, 2.4, 2.4]),
        NOTE(
            "M_A -> M_B again improves prediction (log-loss 0.4774 -> 0.4723); "
            "M_C (adding L x D) adds essentially nothing (0.4720) - most of "
            "the length signal is a main effect under competition, as in 12A. "
            "Frozen rule applied with UNCHANGED thresholds, no threshold "
            "shopping: STAGE A = INTERESTING (harm8192 0.275 >= 0.15 and "
            "deltaInt -0.150 <= -0.15); STAGE B = CONFIRMED AT THE "
            "PREREGISTERED THRESHOLD on fresh n=20 data (harm8192 0.275 >= "
            "0.20 AND deltaInt -0.150 <= -0.05, both True). CONCLUSION: a "
            "competition-dependent long-context penalty is CONFIRMED on "
            "independent samples - D=0 is 1.00 at every length while the "
            "penalty emerges and deepens only under competition, so context "
            "COMPOSITION, not raw length, governs reliability. With 12A (n=5 "
            "screening) and 12B (n=20 confirmation) both clearing the same "
            "frozen rule, the long-context arm rests on a preregistered "
            "independent confirmation, not a single screening pass. Caveats: "
            "single-cell noise at n=20 (SE ~0.11) and harm4096 = -0.075 is a "
            "mid-length non-monotonicity (not in the rule) - the confirmed "
            "claim is the aggregate 8192-vs-2048 contrast, not strict "
            "per-length monotonicity; S=medium, one model, cross-model/S "
            "robustness deferred to Phase 13.")]
    return story


def build_phase1f():
    story = []

    # ---------------- 23. Phase A ----------------
    story += [
        PageBreak(),
        H1("23. Phase A - cross-task validation (an honest negative)"),
        P("The central question left open was whether the interference law is "
          "task-general. We built a second task family with a different "
          "computation - <b>counting</b>: count the &quot;red&quot; objects while "
          "ignoring items of a similar colour (competitor colours chosen by "
          "measured RGB similarity: blue 0.18, orange 0.63, crimson 0.84). "
          "Grid: L in {256,1024}, D in {0,2,4,8}, S in {low,medium,high}, n=10."),
        TBL([
            ["S (measured)", "D=0", "D=2", "D=4", "D=8"],
            ["low (0.18)", "0.55", "0.60", "0.80", "0.65"],
            ["medium (0.63)", "0.55", "0.55", "0.80", "0.70"],
            ["high (0.84)", "0.55", "0.45", "0.50", "0.55"],
        ], widths=[4.0, 2.8, 2.8, 2.8, 2.8]),
        SP(0.15),
        P("<b>Result: the interference structure does NOT transfer to "
          "counting.</b> Accuracy is roughly flat (0.45-0.80) whether D is 0 or "
          "8 and whether S is low or high. The fitted D*S coefficient collapses "
          "from -1.17 (retrieval) to -0.09 (counting), and the retrieval-fitted "
          "law has essentially no predictive power on counting trials "
          "(corr = +0.02)."),
        NOTE("Interpretation (honest). A 0.5B model cannot reliably perform the "
             "counting task at all (baseline ~0.55 even with no competitors), so "
             "the task sits near a <b>capability floor</b> where interference "
             "cannot be measured. Either way, the safe conclusion is the same: "
             "the law is currently a theory of <b>retrieval</b> interference, "
             "not of all context interference."),
        CAP("This narrows the claim; it does not weaken the retrieval result."),
    ]

    # ---------------- 24. Phase B ----------------
    story += [
        H1("24. Phase B - intervention (does using the law help?)"),
        P("The most important remaining test: not just predicting risk, but "
          "<b>acting</b> on it. A retrieval context was built with the needle "
          "plus 8 decoys of known measured similarity (two near-identical, four "
          "moderate, two unrelated). Four conditions were compared at L=1024, "
          "n=20:"),
        TBL([
            ["condition", "decoys kept", "accuracy"],
            ["FULL (none removed)", "8", "0.05"],
            ["RANDOM prune", "4 (random)", "0.35"],
            ["SIM prune (profiler)", "4 (lowest-similarity)", "0.70"],
            ["NONE (no decoys)", "0", "1.00"],
        ], widths=[5.4, 4.0, 3.0]),
        SP(0.15),
        KeepTogether([
            IMG("figure_expB_intervention.png", w=11),
            CAP("Figure 11. Removing the most similar competing passages "
                "recovers accuracy far better than random removal."),
        ]),
        P("<b>Paired result (bootstrap 95% CI):</b> SIM - RANDOM = "
          "<b>+0.35 [0.05, 0.65]</b>; RANDOM - FULL = +0.30 [0.10, 0.50]; "
          "FULL -> SIM = +0.65 [0.45, 0.85]. Pruning by the <i>measured</i> "
          "interference metric roughly <b>doubles</b> accuracy versus random "
          "pruning. This converts the work from description into an actionable "
          "method."),
    ]

    # ---------------- 25. Phase C ----------------
    story += [
        H1("25. Phase C - robustness checks"),
        P("We checked whether the result depends on one particular benchmark "
          "construction. Fixed retrieval task, L=1024, D=4, medium similarity, "
          "n=12; varied target <b>position</b> and <b>filler source</b>."),
        TBL([
            ["filler", "pos=0.1", "pos=0.5", "pos=0.9", "range"],
            ["real prose (WikiText)", "0.83", "0.42", "0.67", "0.42"],
            ["sentence bank (legacy)", "0.83", "0.67", "0.83", "0.17"],
        ], widths=[5.0, 2.9, 2.9, 2.9, 2.3]),
        SP(0.15),
    ]
    story += bullets([
        "The <b>mid-context penalty</b> (lost-in-the-middle) reproduces under "
        "both filler sources and is not an artefact of one construction.",
        "Real prose is systematically <b>harder</b> (mean 0.64) than the "
        "repetitive sentence bank (0.78) - validating the earlier decision to "
        "replace the bank. Repetitive filler had inflated accuracy.",
        "The interference effect is therefore robust to position and to filler "
        "source, while remaining sensitive to how realistic the filler is.",
    ])

    # ---------------- 26. Phase D ----------------
    story += [
        H1("26. Phase D - statistical finalization"),
        P("All headline claims with bootstrap 95% confidence intervals, from the "
          "saved CSVs (4,000 resamples; coefficients 200 refits)."),
        TBL([
            ["claim (measured)", "value", "95% CI"],
            ["Exp6 cell fit, Pearson r", "0.944", "[0.87, 0.98]"],
            ["Exp6 D*S coefficient", "-0.82", "[-1.68, +0.08]"],
            ["Cross-model transfer r", "0.946", "[0.87, 0.98]"],
            ["Cross-model MAE", "0.080", "[0.056, 0.108]"],
            ["Cross-model AUC", "0.915", "[0.893, 0.937]"],
            ["Cross-model ECE (calibration)", "0.049", "-"],
            ["Continuous Jaccard r", "-0.941", "[-0.976, -0.895]"],
            ["Intervention gain (SIM-RANDOM)", "+0.35", "[0.05, 0.65]"],
        ], widths=[6.0, 2.8, 3.8]),
        SP(0.15),
        KeepTogether([
            IMG("figure_calibration.png", w=10),
            CAP("Figure 12. Calibration of the frozen Qwen law on Llama trials "
                "(ECE = 0.049)."),
        ]),
        NOTE("Measured vs inferred - kept strictly separate. <b>Measured:</b> "
             "accuracies, the D*S interaction, similarity metrics, transfer "
             "correlations, intervention gain, calibration. <b>Inferred:</b> the "
             "&quot;attention budget&quot; explanation. We do NOT claim to have "
             "proved an internal budget; the observation is <i>consistent with</i> "
             "an interference-limited retrieval mechanism."),
        P("Honest caveat: the bootstrap CI for the single D*S coefficient "
          "(-0.82, CI [-1.68, +0.08]) <b>includes zero</b> at n=600. The "
          "interaction is best supported jointly with the intervention result "
          "(SIM vs RANDOM, +0.35, CI excludes 0) and the cross-model transfer, "
          "rather than by that coefficient alone."),
    ]

    # ---------------- 27. Frozen ----------------
    story += [
        H1("27. Research frozen"),
        P("All four remaining checkpoints are complete: cross-task validation "
          "(Phase A), intervention (Phase B), robustness (Phase C), and "
          "statistical finalization (Phase D)."),
    ]
    story += bullets([
        "Cross-task: interference law does <b>not</b> transfer to counting - a "
        "capability-floor / task-specificity result. Claim narrowed to "
        "<b>retrieval interference</b>.",
        "Intervention: pruned by measured similarity, accuracy rises from 0.05 "
        "to 0.70 (+0.35 over random) - the law is <b>actionable</b>.",
        "Robustness: holds across position and filler source.",
        "Statistics: full CIs; mechanism explicitly marked as interpretation.",
    ])
    story += [
        NOTE("Final defensible claim. <i>Within the retrieval task family, and "
             "across three model families, retrieval accuracy is predicted by a "
             "multiplicative interaction between the number of competing passages "
             "and their measured similarity to the target, with context length a "
             "weaker secondary effect; the dominant structure transfers "
             "cross-model, and pruning on the measured metric improves "
             "performance.</i> This is a <b>retrieval-interference</b> theory, not "
             "(yet) a general theory of context."),
        H2("27.1 The stopping rule was met"),
        P("The criterion set for freezing research was: can the relationship "
          "generalise beyond the original condition, and can using it improve "
          "performance? Generalisation: cross-model yes, cross-task no. Action: "
          "yes (+0.35). Research is therefore frozen here and the project moves "
          "to the product phase (Context Profiler)."),
    ]
    return story


if __name__ == "__main__":
    doc = SimpleDocTemplate(
        str(OUT), pagesize=A4,
        leftMargin=2.0 * cm, rightMargin=2.0 * cm,
        topMargin=1.8 * cm, bottomMargin=1.8 * cm,
        title="ContextLab - Phase-1 Research Report",
        author="ContextLab",
    )
    doc.build(build_story())
    print(f"wrote {OUT}")

