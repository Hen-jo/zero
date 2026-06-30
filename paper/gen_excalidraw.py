import json, random, time

random.seed(42)
ts = int(time.time() * 1000)

def uid():
    return ''.join(random.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=20))

def rect(x, y, w, h, bg="#ffffff", stroke="#000000", dash=False, opacity=100):
    return {
        "id": uid(), "type": "rectangle",
        "x": x, "y": y, "width": w, "height": h, "angle": 0,
        "strokeColor": stroke, "backgroundColor": bg,
        "fillStyle": "solid", "strokeWidth": 2,
        "strokeStyle": "dashed" if dash else "solid",
        "roughness": 0, "opacity": opacity,
        "groupIds": [], "frameId": None,
        "roundness": {"type": 3},
        "seed": random.randint(0, 999999), "version": 1,
        "versionNonce": random.randint(0, 999999),
        "isDeleted": False, "boundElements": [], "updated": ts,
        "link": None, "locked": False
    }

def text(x, y, w, h, txt, size=14, align="center", color="#000000", bold=False, italic=False):
    return {
        "id": uid(), "type": "text",
        "x": x, "y": y, "width": w, "height": h, "angle": 0,
        "strokeColor": color, "backgroundColor": "transparent",
        "fillStyle": "solid", "strokeWidth": 1,
        "strokeStyle": "solid", "roughness": 0, "opacity": 100,
        "groupIds": [], "frameId": None, "roundness": None,
        "seed": random.randint(0, 999999), "version": 1,
        "versionNonce": random.randint(0, 999999),
        "isDeleted": False, "boundElements": [], "updated": ts,
        "link": None, "locked": False,
        "text": txt, "fontSize": size, "fontFamily": 2,
        "textAlign": align, "verticalAlign": "middle",
        "containerId": None, "originalText": txt,
        "autoResize": True, "lineHeight": 1.25,
    }

def arrow(x1, y1, x2, y2, color="#000000", dash=False, bend=None):
    if bend:
        pts = [[0, 0]] + [[bx - x1, by - y1] for bx, by in bend] + [[x2 - x1, y2 - y1]]
    else:
        pts = [[0, 0], [x2 - x1, y2 - y1]]
    return {
        "id": uid(), "type": "arrow",
        "x": x1, "y": y1,
        "width": abs(x2 - x1), "height": abs(y2 - y1), "angle": 0,
        "strokeColor": color, "backgroundColor": "transparent",
        "fillStyle": "solid", "strokeWidth": 2,
        "strokeStyle": "dashed" if dash else "solid",
        "roughness": 0, "opacity": 100,
        "groupIds": [], "frameId": None,
        "roundness": {"type": 2},
        "seed": random.randint(0, 999999), "version": 1,
        "versionNonce": random.randint(0, 999999),
        "isDeleted": False, "boundElements": [], "updated": ts,
        "link": None, "locked": False,
        "points": pts,
        "lastCommittedPoint": None,
        "startBinding": None, "endBinding": None,
        "startArrowhead": None, "endArrowhead": "arrow",
    }

els = []

# Iteration background
els.append(rect(20, 60, 1640, 760, bg="#f9f9f9", stroke="#999999", dash=True, opacity=40))
els.append(text(830, 72, 200, 28, "Iteration  k", size=15, color="#666666", italic=True))

# Seed Data
els.append(rect(35, 205, 148, 90, bg="#fff9db", stroke="#d6a800"))
els.append(text(35, 208, 148, 22, "Seed QA Data", size=12, bold=True))
els.append(text(35, 232, 148, 58, "NQ  TQA  PopQA\nHotpotQA  2Wiki ...\nhop ratio 4:3:2:1", size=10, color="#555555"))

# Challenger
CX, CY, CW, CH = 215, 95, 330, 300
els.append(rect(CX, CY, CW, CH, bg="#fff0f0", stroke="#c0392b"))
els.append(text(CX, CY + 8, CW, 26, "Challenger (Proposer)", size=14, bold=True, color="#c0392b"))
els.append(rect(CX + 20, CY + 44, CW - 40, 40, bg="#fce4e4", stroke="#c0392b"))
els.append(text(CX + 20, CY + 44, CW - 40, 40, "LLM  (Qwen2.5-3B)", size=12))
els.append(text(CX + 10, CY + 96, CW - 20, 38, "RL: GRPO-Batch\ngroup=1  reward_n=5", size=11, color="#333333"))
els.append(rect(CX + 15, CY + 144, CW - 30, 50, bg="#ffeaa7", stroke="#d6a800"))
els.append(text(CX + 15, CY + 144, CW - 30, 50, "Reward: solver difficulty\n(hard Q  high reward)", size=10, color="#5d4037"))
els.append(text(CX + 10, CY + 205, CW - 20, 50, "Generates multi-hop questions\nby decomposing seed queries\nwith iterative search", size=10, color="#555555"))

# Solver
SX, SY, SW, SH = 700, 95, 330, 300
els.append(rect(SX, SY, SW, SH, bg="#f0fff4", stroke="#27ae60"))
els.append(text(SX, SY + 8, SW, 26, "Solver", size=14, bold=True, color="#27ae60"))
els.append(rect(SX + 20, SY + 44, SW - 40, 40, bg="#c8f7d8", stroke="#27ae60"))
els.append(text(SX + 20, SY + 44, SW - 40, 40, "LLM  (Qwen2.5-3B)", size=12))
els.append(text(SX + 10, SY + 96, SW - 20, 38, "RL: GRPO  (group=5)\nMulti-turn retrieval", size=11, color="#333333"))
els.append(rect(SX + 15, SY + 144, SW - 30, 50, bg="#ffeaa7", stroke="#d6a800"))
els.append(text(SX + 15, SY + 144, SW - 30, 50, "Reward: Verifier judgment\n+ answer correctness (EM)", size=10, color="#5d4037"))
els.append(text(SX + 10, SY + 205, SW - 20, 50, "Answers via iterative search\n+ chain-of-thought reasoning\n<think>...<answer>", size=10, color="#555555"))

# Verifier
VX, VY, VW, VH = 1185, 95, 350, 300
els.append(rect(VX, VY, VW, VH, bg="#f8f0ff", stroke="#8e44ad"))
els.append(text(VX, VY + 8, VW, 26, "Verifier", size=14, bold=True, color="#8e44ad"))
els.append(rect(VX + 20, VY + 44, VW - 40, 40, bg="#e8d5f5", stroke="#8e44ad"))
els.append(text(VX + 20, VY + 44, VW - 40, 40, "LLM  (Qwen2.5-3B)", size=12))
els.append(text(VX + 10, VY + 96, VW - 20, 38, "RL: GRPO  (group=5)\nMulti-turn retrieval", size=11, color="#333333"))
els.append(rect(VX + 15, VY + 144, VW - 30, 50, bg="#ffeaa7", stroke="#d6a800"))
els.append(text(VX + 15, VY + 144, VW - 30, 50, "Reward: verification accuracy\n(yes/no vs. ground truth)", size=10, color="#5d4037"))
els.append(text(VX + 10, VY + 205, VW - 20, 50, "Judges solver answer:\nis the proposed answer correct?\n<answer>yes/no</answer>", size=10, color="#555555"))

# Generated Questions artifact
GQX, GQY = 225, 450
els.append(rect(GQX, GQY, 290, 65, bg="#fff9db", stroke="#d6a800"))
els.append(text(GQX, GQY + 2, 290, 24, "Generated Questions (D_k)", size=11, bold=True))
els.append(text(GQX, GQY + 28, 290, 32, "~50k multi-hop questions / iter\nstratified by hop count", size=10, color="#555555"))

# Solver Predictions artifact
SAX, SAY = 700, 450
els.append(rect(SAX, SAY, 290, 65, bg="#e8f8ee", stroke="#27ae60"))
els.append(text(SAX, SAY + 2, 290, 24, "Solver Predictions", size=11, bold=True, color="#27ae60"))
els.append(text(SAX, SAY + 28, 290, 32, "answer + reasoning chain\n12,800 samples", size=10, color="#555555"))

# Verifier Training Data
VDX, VDY = 1010, 440
els.append(rect(VDX, VDY, 340, 85, bg="#f0e8ff", stroke="#8e44ad", dash=True))
els.append(text(VDX, VDY + 4, 340, 22, "Verifier Training Data", size=11, bold=True, color="#8e44ad"))
els.append(text(VDX, VDY + 28, 340, 55, "(Q, proposed ans) -> yes / no\nPos: solver correct\nNeg: solver wrong  |  12,800 pairs", size=10, color="#444444"))

# Retrieval Module
RX, RY = 490, 620
els.append(rect(RX, RY, 500, 80, bg="#e8f4fd", stroke="#2980b9"))
els.append(text(RX, RY + 6, 500, 22, "Retrieval Module  (shared)", size=12, bold=True, color="#2980b9"))
els.append(text(RX, RY + 32, 500, 44, "Wikipedia Corpus (wiki-18)  |  FAISS Index\nE5-base-v2 Retriever  |  Top-3 passages", size=10, color="#333333"))

# Next iteration
NX, NY = 440, 548
els.append(rect(NX, NY, 600, 48, bg="#f5f5f5", stroke="#555555"))
els.append(text(NX, NY + 2, 600, 44, "C_(k+1)   S_(k+1)   V_(k+1)     Iteration k+1", size=13, bold=True, color="#333333"))

# ── Arrows ────────────────────────────────────────────────────────────────────

# Seed -> Challenger
els.append(arrow(183, 250, CX, 250))

# Challenger -> Gen Questions
els.append(arrow(CX + CW // 2, CY + CH, CX + CW // 2, GQY, color="#c0392b"))

# Gen Questions -> Solver (train, dashed)
els.append(arrow(GQX + 290, GQY + 32, SX, SY + SH // 2 + 20, color="#888888", dash=True,
                 bend=[(620, GQY + 32), (620, SY + SH // 2 + 20)]))

# Gen Questions -> Solver Answers (inference)
els.append(arrow(GQX + 145, GQY + 65, SAX + 145, SAY, color="#888888"))

# Solver -> Solver Answers
els.append(arrow(SX + SW // 2, SY + SH, SX + SW // 2, SAY, color="#27ae60"))

# Solver Answers -> Verifier Data
els.append(arrow(SAX + 290, SAY + 32, VDX, VDY + 42, color="#888888"))

# Verifier Data -> Verifier
els.append(arrow(VDX + VW // 2, VDY, VX + VW // 2, VY + VH, color="#8e44ad"))

# Verifier reward -> Solver (orange loop)
els.append(arrow(VX + VW // 2 + 40, VY + VH, SX + SW - 20, SY + 170,
                 color="#e67e22",
                 bend=[(VX + VW // 2 + 40, 810), (SX + SW - 20, 810)]))
els.append(text(SX + SW // 2 + 20, 816, 180, 20, "reward signal", size=10, color="#e67e22"))

# Solver difficulty -> Challenger reward (orange loop)
els.append(arrow(SX + 30, SY + 170, CX + CW - 20, CY + 170,
                 color="#e67e22",
                 bend=[(SX + 30, 870), (CX + CW - 20, 870)]))
els.append(text(CX + CW // 2, 876, 180, 20, "difficulty signal", size=10, color="#e67e22"))

# Retrieval <-> models (dashed blue)
els.append(arrow(CX + CW // 2, CY + CH, RX + 80, RY, color="#2980b9", dash=True,
                 bend=[(CX + CW // 2, RY + 38), (RX + 80, RY + 38)]))
els.append(arrow(SX + SW // 2, SY + SH, RX + 250, RY, color="#2980b9", dash=True,
                 bend=[(SX + SW // 2, RY + 38), (RX + 250, RY + 38)]))
els.append(arrow(VX + VW // 2, VY + VH, RX + 420, RY, color="#2980b9", dash=True,
                 bend=[(VX + VW // 2, RY + 38), (RX + 420, RY + 38)]))

# -> Next iteration checkpoints
els.append(arrow(CX + CW // 2, CY + CH, NX + 70, NY + 24, color="#555555",
                 bend=[(CX + CW // 2, 532), (NX + 70, 532)]))
els.append(arrow(SX + SW // 2, SY + SH, NX + 300, NY + 24, color="#555555",
                 bend=[(SX + SW // 2, 532), (NX + 300, 532)]))
els.append(arrow(VX + VW // 2, VY + VH, NX + 520, NY + 24, color="#555555",
                 bend=[(VX + VW // 2, 532), (NX + 520, 532)]))

# ── Legend ────────────────────────────────────────────────────────────────────
LX, LY = 40, 660
els.append(rect(LX, LY, 310, 140, bg="#ffffff", stroke="#cccccc"))
els.append(text(LX + 10, LY + 6, 150, 20, "Legend", size=12, bold=True, align="left"))
for i, (clr, dash, lbl) in enumerate([
    ("#000000", False, "Data / model flow"),
    ("#2980b9", True,  "Retrieval search call"),
    ("#e67e22", False, "Reward / feedback signal"),
    ("#555555", True,  "Checkpoint -> next iteration"),
]):
    yy = LY + 35 + i * 26
    els.append(arrow(LX + 12, yy, LX + 60, yy, color=clr, dash=dash))
    els.append(text(LX + 68, yy - 10, 230, 20, lbl, size=10, align="left", color=clr))

doc = {
    "type": "excalidraw",
    "version": 2,
    "source": "https://excalidraw.com",
    "elements": els,
    "appState": {
        "gridSize": None,
        "viewBackgroundColor": "#ffffff"
    },
    "files": {}
}

path = "d:/paper/drzero/paper/figure_architecture.excalidraw"
with open(path, "w", encoding="utf-8") as f:
    json.dump(doc, f, ensure_ascii=False, indent=2)
print(f"Saved {len(els)} elements -> {path}")
