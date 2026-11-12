#!/usr/bin/env python3
"""
Generates a publication-grade Presentation Slide Deck PDF for the LLM Agent Failure Research Project.
Covers:
1. Title Slide
2. Introduction
3. Objectives
4. Proposed Methodology
5. Work Done So Far - Part 1 (Data & Features)
6. Work Done So Far - Part 2 (Model Benchmark & Calibration)
7. Work Done So Far - Part 3 (Early Detection & Decision Regret)
8. Work Done So Far - Part 4 (Generalization & Latency)
9. Summary & Future Scope
"""

import os
import sys
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.pdfgen import canvas

PDF_PATH = r"c:\Users\aparn\Desktop\Minor\llm-agent-failure-research\PROJECT_PRESENTATION_SLIDES.pdf"

# Palette
PRIMARY = colors.HexColor("#0F172A")     # Slate 900
SECONDARY = colors.HexColor("#1E40AF")   # Blue 800
ACCENT = colors.HexColor("#2563EB")      # Blue 600
ACCENT_LIGHT = colors.HexColor("#EFF6FF")# Blue 50
SUCCESS = colors.HexColor("#166534")     # Green 800
SUCCESS_BG = colors.HexColor("#DCFCE7")  # Green 100
DARK_TEXT = colors.HexColor("#1E293B")   # Slate 800
MUTED_TEXT = colors.HexColor("#64748B")  # Slate 500
BORDER_COLOR = colors.HexColor("#CBD5E1")# Slate 300
BG_CARD = colors.HexColor("#F8FAFC")     # Slate 50

class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and print total page count."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_decorations(self, total_pages):
        width, height = self._pagesize
        if self._pageNumber == 1:
            # Title slide decorative bottom bar
            self.setFillColor(SECONDARY)
            self.rect(0, 0, width, 8, fill=True, stroke=False)
            return

        # Top decorative banner
        self.setFillColor(SECONDARY)
        self.rect(0, height - 6, width, 6, fill=True, stroke=False)

        # Bottom footer bar
        self.setStrokeColor(BORDER_COLOR)
        self.setLineWidth(0.75)
        self.line(36, 30, width - 36, 30)

        # Footer text
        self.setFont("Helvetica", 8)
        self.setFillColor(MUTED_TEXT)
        self.drawString(36, 18, "Online Telemetry-Based Failure Detection in LLM Tool-Using Agents | Minor Project")
        
        page_str = f"Slide {self._pageNumber} of {total_pages}"
        self.drawRightString(width - 36, 18, page_str)

def build_pdf():
    # Page dimensions: Landscape Letter (11 x 8.5 in -> 792 x 612 pt)
    page_width, page_height = landscape(letter)
    
    doc = SimpleDocTemplate(
        PDF_PATH,
        pagesize=(page_width, page_height),
        leftMargin=40,
        rightMargin=40,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_main = ParagraphStyle(
        "TitleMain",
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=30,
        textColor=PRIMARY,
        alignment=TA_LEFT,
    )
    title_sub = ParagraphStyle(
        "TitleSub",
        fontName="Helvetica",
        fontSize=12,
        leading=16,
        textColor=SECONDARY,
        alignment=TA_LEFT,
    )
    slide_header = ParagraphStyle(
        "SlideHeader",
        fontName="Helvetica-Bold",
        fontSize=17,
        leading=21,
        textColor=PRIMARY,
    )
    slide_subheader = ParagraphStyle(
        "SlideSubHeader",
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=14,
        textColor=SECONDARY,
    )
    body_text = ParagraphStyle(
        "BodyTextCustom",
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=DARK_TEXT,
    )
    bullet_text = ParagraphStyle(
        "BulletTextCustom",
        fontName="Helvetica",
        fontSize=9.5,
        leading=14,
        textColor=DARK_TEXT,
        leftIndent=12,
    )
    bullet_bold = ParagraphStyle(
        "BulletBoldCustom",
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=14,
        textColor=PRIMARY,
        leftIndent=12,
    )
    table_cell = ParagraphStyle(
        "TableCell",
        fontName="Helvetica",
        fontSize=8.5,
        leading=11.5,
        textColor=DARK_TEXT,
        alignment=TA_CENTER,
    )
    table_cell_bold = ParagraphStyle(
        "TableCellBold",
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11.5,
        textColor=PRIMARY,
        alignment=TA_CENTER,
    )
    table_cell_left = ParagraphStyle(
        "TableCellLeft",
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11.5,
        textColor=PRIMARY,
        alignment=TA_LEFT,
    )
    card_title = ParagraphStyle(
        "CardTitle",
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=13,
        textColor=SECONDARY,
    )
    speaker_note = ParagraphStyle(
        "SpeakerNote",
        fontName="Helvetica-Oblique",
        fontSize=8.5,
        leading=12,
        textColor=MUTED_TEXT,
        alignment=TA_LEFT,
    )

    story = []

    # =========================================================================
    # SLIDE 1: TITLE SLIDE
    # =========================================================================
    story.append(Spacer(1, 40))
    # Category tag
    tag_data = [[Paragraph("<font color='#1E40AF'><b>RESEARCH & PROJECT PROGRESS PRESENTATION</b></font>", body_text)]]
    tag_table = Table(tag_data, colWidths=[320])
    tag_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), ACCENT_LIGHT),
        ('BOX', (0,0), (-1,-1), 1, ACCENT),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(tag_table)
    story.append(Spacer(1, 16))

    story.append(Paragraph("Online Telemetry-Based Failure Detection<br/>in LLM Tool-Using Agents", title_main))
    story.append(Spacer(1, 10))
    story.append(Paragraph("An Empirical Study of Causal Runtime Signals, Multi-Model Calibration & Early Warning Dynamics", title_sub))
    story.append(Spacer(1, 28))

    meta_content = [
        [
            Paragraph("<b>Presentation Outline:</b>", body_text),
            Paragraph("<b>Target Frameworks:</b>", body_text),
        ],
        [
            Paragraph("• 1. Introduction & Problem Statement<br/>• 2. Research Objectives<br/>• 3. Proposed Methodology & Architecture<br/>• 4. Work Done So Far (Detailed Experiments)", body_text),
            Paragraph("• ToolBench (4,000 Real Trajectories)<br/>• LangGraph Orchestration + Ollama (qwen3:8b)<br/>• 14 Causal Telemetry Signals (19,233 steps)<br/>• XGBoost, Calibration & Decision Regret", body_text),
        ]
    ]
    meta_table = Table(meta_content, colWidths=[340, 360])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), BG_CARD),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING', (0,0), (-1,-1), 16),
        ('RIGHTPADDING', (0,0), (-1,-1), 16),
        ('TOPPADDING', (0,0), (-1,-1), 12),
        ('BOTTOMPADDING', (0,0), (-1,-1), 12),
    ]))
    story.append(meta_table)
    story.append(PageBreak())

    # =========================================================================
    # SLIDE 2: INTRODUCTION
    # =========================================================================
    story.append(Paragraph("1. Introduction", slide_subheader))
    story.append(Spacer(1, 2))
    story.append(Paragraph("The Latent Failure Challenge in Autonomous LLM Agents", slide_header))
    story.append(Spacer(1, 14))

    intro_cols = [
        [
            Paragraph("<b>The Rise of Tool-Using Agents</b>", card_title),
            Paragraph("<b>The Latent Failure Problem</b>", card_title),
        ],
        [
            Paragraph("• Autonomous LLM agents (LangGraph, ReAct) plan, reason, and invoke external APIs to execute complex multi-step workflows.<br/>• Unlike traditional deterministic software, LLMs operate stochastically and encounter unhandled edge cases.", body_text),
            Paragraph("• Failures manifest <b>latently</b>: agents enter recursive tool loops, hallucinate tool parameters, or diverge from goals while still emitting fluent natural text.<br/>• Uncaught errors lead to compute waste, rate-limit burn, and corrupted downstream tasks.", body_text),
        ],
        [
            Paragraph("<b>Why Existing Defenses Fall Short</b>", card_title),
            Paragraph("<b>Our Core Innovation</b>", card_title),
        ],
        [
            Paragraph("• <i>Post-Hoc Evaluators:</i> Identify errors only after termination—the failure has already occurred.<br/>• <i>LLM-as-a-Judge:</i> Querying a secondary LLM at every step adds ~650 ms latency and substantial token costs, rendering real-time safety impractical.", body_text),
            Paragraph("• <b>Lightweight Causal Telemetry:</b> Extracting 14 scalar runtime features (latencies, error streaks, lexical entropy) to predict failure <b>2 to 3 steps in advance</b>.<br/>• Runs in <b>0.004 ms at $0 token cost</b>—enabling real-time monitoring.", body_text),
        ]
    ]
    intro_table = Table(intro_cols, colWidths=[345, 345])
    intro_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,0), ACCENT_LIGHT),
        ('BACKGROUND', (1,0), (1,0), ACCENT_LIGHT),
        ('BACKGROUND', (0,2), (0,2), ACCENT_LIGHT),
        ('BACKGROUND', (1,2), (1,2), ACCENT_LIGHT),
        ('BACKGROUND', (0,1), (0,1), BG_CARD),
        ('BACKGROUND', (1,1), (1,1), BG_CARD),
        ('BACKGROUND', (0,3), (0,3), BG_CARD),
        ('BACKGROUND', (1,3), (1,3), BG_CARD),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
        ('RIGHTPADDING', (0,0), (-1,-1), 12),
        ('TOPPADDING', (0,0), (-1,-1), 7),
        ('BOTTOMPADDING', (0,0), (-1,-1), 7),
    ]))
    story.append(intro_table)
    story.append(Spacer(1, 14))
    story.append(Paragraph("<b>Key Takeaway:</b> Before building an active intervention guardrail, we must scientifically prove that lightweight telemetry contains sufficient signal to forecast failures ahead of time.", speaker_note))
    story.append(PageBreak())

    # =========================================================================
    # SLIDE 3: OBJECTIVES
    # =========================================================================
    story.append(Paragraph("2. Objectives", slide_subheader))
    story.append(Spacer(1, 2))
    story.append(Paragraph("Research Objectives & Experimental Scope", slide_header))
    story.append(Spacer(1, 14))

    obj_cards = [
        [
            Paragraph("<b>Primary Research Objective</b>", card_title),
        ],
        [
            Paragraph("To develop, validate, and benchmark an <b>online, strictly causal failure prediction methodology</b> for tool-using LLM agents that forecasts failure within a 2-step horizon (H=2) using lightweight telemetry—eliminating lookahead leakage and heavy LLM judge overhead.", body_text),
        ],
        [
            Paragraph("<b>Specific Scientific & Engineering Objectives</b>", card_title),
        ],
        [
            Paragraph("<b>1. Multi-Class Data Generation:</b> Execute 4,000 real-world ToolBench tasks across <i>Healthy</i>, <i>Injected</i>, and natural <i>Organic</i> LLM execution modes.<br/>"
                      "<b>2. Causal Feature Engineering:</b> Formulate 14 telemetry signals strictly observable at step ≤ t (temporal, tool errors, context lengths, lexical diversity).<br/>"
                      "<b>3. Multi-Model Benchmarking & Calibration:</b> Train and evaluate 5 classifier families with 1,000 bootstrap iterations and calibrate probabilities (ECE target &lt; 0.05).<br/>"
                      "<b>4. Early Warning & Lead Time Profiling:</b> Quantify empirical advance lead time (L = t_fail - t_alarm) prior to terminal crash.<br/>"
                      "<b>5. Decision Regret Optimization:</b> Formulate a cost-sensitive Bayes-optimal intervention policy to mathematically balance false alarms vs. missed failures.<br/>"
                      "<b>6. Cross-Domain Generalization:</b> Verify zero-shot transfer from controlled synthetic faults to natural organic LLM hallucinations (Hypothesis H6).", body_text),
        ]
    ]
    obj_table = Table(obj_cards, colWidths=[700])
    obj_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,0), ACCENT_LIGHT),
        ('BACKGROUND', (0,1), (0,1), BG_CARD),
        ('BACKGROUND', (0,2), (0,2), ACCENT_LIGHT),
        ('BACKGROUND', (0,3), (0,3), BG_CARD),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('LEFTPADDING', (0,0), (-1,-1), 14),
        ('RIGHTPADDING', (0,0), (-1,-1), 14),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(obj_table)
    story.append(PageBreak())

    # =========================================================================
    # SLIDE 4: PROPOSED METHODOLOGY
    # =========================================================================
    story.append(Paragraph("3. Proposed Methodology", slide_subheader))
    story.append(Spacer(1, 2))
    story.append(Paragraph("End-to-End System Architecture & Execution Pipeline", slide_header))
    story.append(Spacer(1, 14))

    meth_cols = [
        [
            Paragraph("<b>Stage 1: Execution Layer</b>", card_title),
            Paragraph("<b>Stage 2: Causal Telemetry Layer</b>", card_title),
        ],
        [
            Paragraph("• <b>ToolBench Streaming:</b> Real-world multi-step tasks streamed directly via Hugging Face API.<br/>"
                      "• <b>LangGraph Orchestration:</b> Agent state graph executing with local <code>qwen3:8b</code> via Ollama.<br/>"
                      "• <b>3 Execution Modes:</b> Healthy runs, injected fault runs (parameters, timeouts, loops), and organic error runs.", body_text),
            Paragraph("• <b>Strict Causality (≤ t):</b> Only past events are visible to feature computation—zero future lookahead.<br/>"
                      "• <b>14 Scalar Signals:</b> Latency, error streak, token ratio, lexical diversity.<br/>"
                      "• <b>Horizon Labeling:</b> Ground truth target <code>imminent_failure_H2</code> flagged if failure occurs in [t+1, t+2].", body_text),
        ],
        [
            Paragraph("<b>Stage 3: Predictive ML Layer</b>", card_title),
            Paragraph("<b>Stage 4: Decision Policy Layer</b>", card_title),
        ],
        [
            Paragraph("• <b>Trajectory-Disjoint Splitting:</b> 70% Train, 15% Val, 15% Test partitioned at trajectory level.<br/>"
                      "• <b>Classifiers Trained:</b> XGBoost, Gradient Boosting, Random Forest, Logistic Regression, MLP.<br/>"
                      "• <b>Risk Calibration:</b> Validated honest output probability <code>p ∈ [0, 1]</code> (ECE = 0.023).", body_text),
            Paragraph("• <b>Cost-Sensitive Decision Theory:</b> Asymmetric penalties (C_FP = 1.0 vs C_FN = 5.0).<br/>"
                      "• <b>Bayes Optimal Threshold:</b> θ* = C_FP / (C_FN - C_fix + C_FP) ≈ 0.18.<br/>"
                      "• <b>Simulated Regret Minimization:</b> Benchmarked vs Passive and Static policies.", body_text),
        ]
    ]
    meth_table = Table(meth_cols, colWidths=[345, 345])
    meth_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,0), ACCENT_LIGHT),
        ('BACKGROUND', (1,0), (1,0), ACCENT_LIGHT),
        ('BACKGROUND', (0,2), (0,2), ACCENT_LIGHT),
        ('BACKGROUND', (1,2), (1,2), ACCENT_LIGHT),
        ('BACKGROUND', (0,1), (0,1), BG_CARD),
        ('BACKGROUND', (1,1), (1,1), BG_CARD),
        ('BACKGROUND', (0,3), (0,3), BG_CARD),
        ('BACKGROUND', (1,3), (1,3), BG_CARD),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
        ('RIGHTPADDING', (0,0), (-1,-1), 12),
        ('TOPPADDING', (0,0), (-1,-1), 7),
        ('BOTTOMPADDING', (0,0), (-1,-1), 7),
    ]))
    story.append(meth_table)
    story.append(PageBreak())

    # =========================================================================
    # SLIDE 5: WORK DONE SO FAR - PART 1 (DATA & FEATURES)
    # =========================================================================
    story.append(Paragraph("4. Work Done So Far", slide_subheader))
    story.append(Spacer(1, 2))
    story.append(Paragraph("Step 1 & 2: Trajectory Execution & Feature Engineering", slide_header))
    story.append(Spacer(1, 14))

    step1_cards = [
        [
            Paragraph("<b>Step 1: Benchmark Execution (4,000 Tasks Completed)</b>", card_title),
            Paragraph("<b>Step 2: 14 Causal Telemetry Features (19,233 Steps)</b>", card_title),
        ],
        [
            Paragraph("• Successfully completed all 4,000 scheduled ToolBench tasks using LangGraph + local <code>qwen3:8b</code>.<br/>"
                      "• <b>Healthy Runs (1,438 runs):</b> Normal execution demonstrating baseline task completion.<br/>"
                      "• <b>Injected Runs (1,980 runs):</b> Deliberate system, tool, parameter, and loop injections.<br/>"
                      "• <b>Organic Runs (582 runs):</b> Natural LLM reasoning failures, hallucinations, and drift.<br/>"
                      "• Output: High-fidelity step logs in <code>data/generated/</code>.", body_text),
            Paragraph("• Extracted 14 strictly causal telemetry signals at every step:<br/>"
                      "  - <i>Temporal:</i> <code>step</code>, <code>step_ratio</code>, <code>step_latency</code>, <code>cum_latency</code><br/>"
                      "  - <i>Tool Dynamics:</i> <code>tool_calls_count</code>, <code>tool_error_count</code>, <code>tool_error_rate</code>, <code>consecutive_tool_errors</code>, <code>repeat_tool_ratio</code><br/>"
                      "  - <i>Context Dynamics:</i> <code>latest_message_len</code>, <code>avg_message_len</code>, <code>token_expansion_ratio</code><br/>"
                      "  - <i>Semantic Health:</i> <code>observation_error_flag</code>, <code>lexical_diversity</code><br/>"
                      "• Verified zero temporal leakage via automated audit script.", body_text),
        ],
        [
            Paragraph("<b>Dataset Partitions (Trajectory-Disjoint)</b>", card_title),
            Paragraph("<b>Dataset Curation Artifacts</b>", card_title),
        ],
        [
            Paragraph("• <b>Train Set (70%):</b> 13,463 steps (2,800 trajectories) for classifier training.<br/>"
                      "• <b>Validation Set (15%):</b> 2,885 steps (600 trajectories) for probability calibration.<br/>"
                      "• <b>Test Set (15%):</b> 2,885 steps (600 trajectories) strictly held out for final evaluation.", body_text),
            Paragraph("• Produced <code>telemetry_dataset.csv</code> (19,233 rows × 23 columns).<br/>"
                      "• Produced clean <code>refined_training_dataset.csv</code> (15 columns: 14 features + 1 target).<br/>"
                      "• Verified complete integrity with zero missing values or NaN entries.", body_text),
        ]
    ]
    step1_table = Table(step1_cards, colWidths=[345, 345])
    step1_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,0), ACCENT_LIGHT),
        ('BACKGROUND', (1,0), (1,0), ACCENT_LIGHT),
        ('BACKGROUND', (0,2), (0,2), ACCENT_LIGHT),
        ('BACKGROUND', (1,2), (1,2), ACCENT_LIGHT),
        ('BACKGROUND', (0,1), (0,1), BG_CARD),
        ('BACKGROUND', (1,1), (1,1), BG_CARD),
        ('BACKGROUND', (0,3), (0,3), BG_CARD),
        ('BACKGROUND', (1,3), (1,3), BG_CARD),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
        ('RIGHTPADDING', (0,0), (-1,-1), 12),
        ('TOPPADDING', (0,0), (-1,-1), 7),
        ('BOTTOMPADDING', (0,0), (-1,-1), 7),
    ]))
    story.append(step1_table)
    story.append(PageBreak())

    # =========================================================================
    # SLIDE 6: WORK DONE SO FAR - PART 2 (MODELS & CALIBRATION)
    # =========================================================================
    story.append(Paragraph("4. Work Done So Far", slide_subheader))
    story.append(Spacer(1, 2))
    story.append(Paragraph("Step 3 & 4: Multi-Model Benchmark & Probability Calibration", slide_header))
    story.append(Spacer(1, 12))

    # Benchmark Table
    bench_data = [
        [
            Paragraph("<b>Model Family</b>", table_cell_left),
            Paragraph("<b>ROC-AUC</b>", table_cell_bold),
            Paragraph("<b>PR-AUC</b>", table_cell_bold),
            Paragraph("<b>F1-Score</b>", table_cell_bold),
            Paragraph("<b>Precision</b>", table_cell_bold),
            Paragraph("<b>Recall</b>", table_cell_bold),
            Paragraph("<b>Brier Score</b>", table_cell_bold),
            Paragraph("<b>ECE (Error)</b>", table_cell_bold),
        ],
        [
            Paragraph("<b>XGBoost Classifier (Primary)</b>", table_cell_left),
            Paragraph("<b>0.908</b>", table_cell_bold),
            Paragraph("<b>0.831</b>", table_cell_bold),
            Paragraph("<b>0.733</b>", table_cell_bold),
            Paragraph("0.761", table_cell),
            Paragraph("0.707", table_cell),
            Paragraph("0.116", table_cell),
            Paragraph("<b>0.023</b>", table_cell_bold),
        ],
        [
            Paragraph("Gradient Boosting", table_cell_left),
            Paragraph("0.909", table_cell),
            Paragraph("0.834", table_cell),
            Paragraph("0.732", table_cell),
            Paragraph("0.760", table_cell),
            Paragraph("0.705", table_cell),
            Paragraph("0.115", table_cell),
            Paragraph("0.028", table_cell),
        ],
        [
            Paragraph("MLP Classifier (Neural Net)", table_cell_left),
            Paragraph("0.898", table_cell),
            Paragraph("0.814", table_cell),
            Paragraph("0.712", table_cell),
            Paragraph("0.758", table_cell),
            Paragraph("0.672", table_cell),
            Paragraph("0.122", table_cell),
            Paragraph("0.028", table_cell),
        ],
        [
            Paragraph("Random Forest", table_cell_left),
            Paragraph("0.888", table_cell),
            Paragraph("0.791", table_cell),
            Paragraph("0.693", table_cell),
            Paragraph("0.780", table_cell),
            Paragraph("0.623", table_cell),
            Paragraph("0.130", table_cell),
            Paragraph("0.053", table_cell),
        ],
        [
            Paragraph("Logistic Regression", table_cell_left),
            Paragraph("0.859", table_cell),
            Paragraph("0.751", table_cell),
            Paragraph("0.644", table_cell),
            Paragraph("0.725", table_cell),
            Paragraph("0.579", table_cell),
            Paragraph("0.143", table_cell),
            Paragraph("0.030", table_cell),
        ],
    ]
    bench_table = Table(bench_data, colWidths=[160, 75, 75, 75, 75, 75, 80, 85])
    bench_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), SECONDARY),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('BACKGROUND', (0,1), (-1,1), SUCCESS_BG),
        ('BACKGROUND', (0,2), (-1,-1), BG_CARD),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(bench_table)
    story.append(Spacer(1, 14))

    findings_data = [
        [
            Paragraph("<b>Key Benchmark Findings</b>", card_title),
            Paragraph("<b>Probability Calibration Findings</b>", card_title),
        ],
        [
            Paragraph("• <b>Tree-Boosting Superiority:</b> Non-linear telemetry spikes (error streaks, token expansion) are captured far better by tree boosting than linear models.<br/>"
                      "• <b>Statistical Significance:</b> McNemar's test confirmed XGBoost significantly outperforms Logistic Regression (χ² = 105.52, p &lt; 10⁻²⁴) and Random Forest (χ² = 18.41, p = 1.7×10⁻⁵).", body_text),
            Paragraph("• <b>Honest Probabilities:</b> XGBoost naturally achieved an Expected Calibration Error of <b>0.023 (2.3% error)</b> right out of the box.<br/>"
                      "• <b>Post-Hoc Distortions:</b> Platt Scaling (0.035) and Isotonic Regression (0.030) slightly degraded calibration—confirming raw tree outputs are already reliable.", body_text),
        ]
    ]
    findings_table = Table(findings_data, colWidths=[345, 345])
    findings_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), ACCENT_LIGHT),
        ('BACKGROUND', (0,1), (-1,1), BG_CARD),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
        ('RIGHTPADDING', (0,0), (-1,-1), 12),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(findings_table)
    story.append(PageBreak())

    # =========================================================================
    # SLIDE 7: WORK DONE SO FAR - PART 3 (EARLY DETECTION & REGRET)
    # =========================================================================
    story.append(Paragraph("4. Work Done So Far", slide_subheader))
    story.append(Spacer(1, 2))
    story.append(Paragraph("Step 5 & 6: Early Detection Lead Time & Policy Regret", slide_header))
    story.append(Spacer(1, 12))

    # Early detection & regret metrics
    step3_cards = [
        [
            Paragraph("<b>Early Detection Dynamics (600 Test Tasks)</b>", card_title),
            Paragraph("<b>Decision Regret Optimization (Bayes Policy)</b>", card_title),
        ],
        [
            Paragraph("• <b>94.75% Overall Detection Rate:</b> 289 of 305 failing trajectories successfully flagged before terminal crash.<br/>"
                      "• <b>2.66 Steps Mean Lead Time:</b> Alarms trigger an average of 2.66 steps ahead of fatal failure.<br/>"
                      "• <b>Organic Lead Time:</b> Reached <b>6.3 steps advance notice</b> on natural cognitive errors.<br/>"
                      "• <i>Significance:</i> Proves a 2-to-3 step window exists for future guardrails to execute corrective re-prompting.", body_text),
            Paragraph("• <b>Cost Asymmetry:</b> False Alarm penalty C_FP = 1.0 vs. Uncaught Failure penalty C_FN = 5.0.<br/>"
                      "• <b>Bayes Optimal Threshold:</b> θ* = C_FP / (C_FN - C_fix + C_FP) ≈ 0.18.<br/>"
                      "• <b>Regret Reduction:</b><br/>"
                      "  - Passive (No Guardrail): Mean Regret = 1.689 (Success: 66.2%)<br/>"
                      "  - Static Threshold (p ≥ 0.5): Mean Regret = 0.571 (66.2% cut)<br/>"
                      "  - <b>Adaptive Policy: Mean Regret = 0.280 (83.4% Regret Cut! Success: 94.4%)</b>", body_text),
        ]
    ]
    step3_table = Table(step3_cards, colWidths=[345, 345])
    step3_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), ACCENT_LIGHT),
        ('BACKGROUND', (0,1), (-1,1), BG_CARD),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
        ('RIGHTPADDING', (0,0), (-1,-1), 12),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(step3_table)
    story.append(Spacer(1, 14))

    # Multi-tier action table
    tier_data = [
        [Paragraph("<b>Risk Band</b>", table_cell_bold), Paragraph("<b>Threshold Rule</b>", table_cell_bold), Paragraph("<b>Simulated Guardrail Action</b>", table_cell_bold), Paragraph("<b>Operational Impact</b>", table_cell_bold)],
        [Paragraph("Safe Zone", table_cell), Paragraph("p &lt; 0.18", table_cell), Paragraph("No Intervention (Allow)", table_cell_left), Paragraph("Zero latency or token overhead added", table_cell_left)],
        [Paragraph("Warning Zone", table_cell), Paragraph("0.18 ≤ p &lt; 0.50", table_cell), Paragraph("Inject Soft Warning", table_cell_left), Paragraph("Prompt injection alerting agent to inspect parameters", table_cell_left)],
        [Paragraph("Danger Zone", table_cell), Paragraph("0.50 ≤ p &lt; 0.80", table_cell), Paragraph("Force Re-Planning Directive", table_cell_left), Paragraph("Instructs agent to abort current branch and re-plan", table_cell_left)],
        [Paragraph("Critical Zone", table_cell), Paragraph("p ≥ 0.80", table_cell), Paragraph("Tool Repair / Hard Halt", table_cell_left), Paragraph("Direct argument recovery or human escalation", table_cell_left)],
    ]
    tier_table = Table(tier_data, colWidths=[120, 120, 200, 260])
    tier_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), SECONDARY),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('BACKGROUND', (0,1), (-1,-1), BG_CARD),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(tier_table)
    story.append(PageBreak())

    # =========================================================================
    # SLIDE 8: WORK DONE SO FAR - PART 4 (GENERALIZATION & LATENCY)
    # =========================================================================
    story.append(Paragraph("4. Work Done So Far", slide_subheader))
    story.append(Spacer(1, 2))
    story.append(Paragraph("Step 7, 8 & 9: Generalization, Taxonomy & Latency Benchmarks", slide_header))
    story.append(Spacer(1, 14))

    step4_cards = [
        [
            Paragraph("<b>Injected-to-Organic Transfer (Hypothesis H6)</b>", card_title),
            Paragraph("<b>Efficiency vs. LLM-as-a-Judge Baseline</b>", card_title),
        ],
        [
            Paragraph("• Model trained strictly on artificial fault injections evaluated zero-shot on natural organic AI errors:<br/>"
                      "  - <i>In-Distribution (Injected):</i> ROC-AUC = <b>0.899</b><br/>"
                      "  - <i>Out-of-Distribution (Organic):</i> ROC-AUC = <b>0.915</b><br/>"
                      "• <b>Key Discovery:</b> Natural reasoning errors exhibit stronger vocabulary decay than abrupt system faults, making them even easier for telemetry to detect.", body_text),
            Paragraph("• <b>Inference Latency:</b> Telemetry Guardrail = <b>0.004 ms</b> vs. LLM Judge = <b>643.9 ms</b> (<b>156,219× faster</b>).<br/>"
                      "• <b>Token Consumption:</b> Telemetry Guardrail = <b>0 tokens</b> vs. LLM Judge = 171 tokens per step.<br/>"
                      "• <b>Cost per 1M Steps:</b> Telemetry = <b>$0.00</b> vs. LLM Judge = $342.00.<br/>"
                      "• Proves telemetry can operate as an invisible inline safety net.", body_text),
        ],
        [
            Paragraph("<b>21-Category Failure Taxonomy Breakdown</b>", card_title),
            Paragraph("<b>Feature Ablation Insights</b>", card_title),
        ],
        [
            Paragraph("• Evaluated 21 granular failure types (16 cognitive, 5 temporal).<br/>"
                      "• Tool execution errors flagged by <code>tool_error_rate</code> (35%).<br/>"
                      "• Hallucinations and loops flagged by <code>lexical_diversity</code> (35%).<br/>"
                      "• Global top feature: <code>lexical_diversity</code> (importance weight: <b>0.307</b>).", body_text),
            Paragraph("• Systematic Leave-One-Group-Out ablation study:<br/>"
                      "  - Removing Context features caused the largest drop (ΔAUC = -0.028).<br/>"
                      "  - No single group alone achieved > 0.77 ROC-AUC.<br/>"
                      "• Proves multi-modal telemetry signals are strictly complementary.", body_text),
        ]
    ]
    step4_table = Table(step4_cards, colWidths=[345, 345])
    step4_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,0), ACCENT_LIGHT),
        ('BACKGROUND', (1,0), (1,0), ACCENT_LIGHT),
        ('BACKGROUND', (0,2), (0,2), ACCENT_LIGHT),
        ('BACKGROUND', (1,2), (1,2), ACCENT_LIGHT),
        ('BACKGROUND', (0,1), (0,1), BG_CARD),
        ('BACKGROUND', (1,1), (1,1), BG_CARD),
        ('BACKGROUND', (0,3), (0,3), BG_CARD),
        ('BACKGROUND', (1,3), (1,3), BG_CARD),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
        ('RIGHTPADDING', (0,0), (-1,-1), 12),
        ('TOPPADDING', (0,0), (-1,-1), 7),
        ('BOTTOMPADDING', (0,0), (-1,-1), 7),
    ]))
    story.append(step4_table)
    story.append(PageBreak())

    # =========================================================================
    # SLIDE 9: SUMMARY & FUTURE SCOPE
    # =========================================================================
    story.append(Paragraph("5. Summary & Next Steps", slide_subheader))
    story.append(Spacer(1, 2))
    story.append(Paragraph("Project Status & Roadmap to Deployment", slide_header))
    story.append(Spacer(1, 14))

    sum_cols = [
        [
            Paragraph("<b>Completed Project Deliverables</b>", card_title),
            Paragraph("<b>Future Scope & Practical Roadmap</b>", card_title),
        ],
        [
            Paragraph("• ✅ <b>4,000-Trajectory Benchmark Curated:</b> Balanced across healthy, injected, and natural organic execution modes.<br/>"
                      "• ✅ <b>14 Causal Telemetry Signals Extracted:</b> Complete causal verification with zero temporal data leakage.<br/>"
                      "• ✅ <b>Calibrated Predictive Model Established:</b> XGBoost achieved 90.8% ROC-AUC with an optimal ECE of 0.023.<br/>"
                      "• ✅ <b>Early Detection Window Proved:</b> Demonstrated 94.75% detection rate with a 2.66-step lead time.<br/>"
                      "• ✅ <b>Decision Regret Minimized:</b> Adaptive Bayes policy cut simulated decision regret by 83.4%.<br/>"
                      "• ✅ <b>Zero-Shot Transfer Confirmed:</b> Proven generalization to natural organic LLM reasoning failures.<br/>"
                      "• ✅ <b>Full Documentation:</b> Completed 7,400-word research paper draft and step-by-step pipeline guides.", body_text),
            Paragraph("• <b>1. Live Inline Guardrail Middleware:</b> Deploy the trained XGBoost detector as an active node interceptor in LangGraph runtime loops.<br/>"
                      "• <b>2. Closed-Loop Autonomous Recovery:</b> Implement active remediation handlers (prompt reflection, tool parameter repair, re-planning directives) triggered upon early risk alarms.<br/>"
                      "• <b>3. Empirical Counterfactual Testing:</b> Re-run agent branches from early warning points to empirically measure live task recovery rates.<br/>"
                      "• <b>4. Multi-Agent Telemetry Expansion:</b> Extend causal tracking to multi-agent collaborative workflows where cascading failures cross agent boundaries.<br/>"
                      "• <b>5. Streaming Online Adaptation:</b> Implement incremental model updates from observed production outcomes without full offline retraining.", body_text),
        ]
    ]
    sum_table = Table(sum_cols, colWidths=[345, 345])
    sum_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), ACCENT_LIGHT),
        ('BACKGROUND', (0,1), (-1,1), BG_CARD),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('LEFTPADDING', (0,0), (-1,-1), 14),
        ('RIGHTPADDING', (0,0), (-1,-1), 14),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(sum_table)

    # Build document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated PDF: {PDF_PATH}")

if __name__ == "__main__":
    build_pdf()
