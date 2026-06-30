out = "d:/paper/drzero/paper/figure_v2_twopanel.drawio"

def wrap(inner):
    return '<mxfile host="app.diagrams.net" version="24.7.17"><diagram id="fig" name="Page-1"><mxGraphModel dx="1600" dy="900" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1700" pageHeight="1100" math="0" shadow="0"><root><mxCell id="0"/><mxCell id="1" parent="0"/>' + inner + '</root></mxGraphModel></diagram></mxfile>'

def R(id_, val, style, x, y, w, h):
    return f'<mxCell id="{id_}" value="{val}" style="{style}" vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>\n'

def E(id_, val, style, src, tgt, pts=None):
    geo = '<mxGeometry relative="1" as="geometry"/>'
    if pts:
        geo = '<mxGeometry relative="1" as="geometry"><Array as="points">' + ''.join(f'<mxPoint x="{p[0]}" y="{p[1]}"/>' for p in pts) + '</Array></mxGeometry>'
    return f'<mxCell id="{id_}" value="{val}" style="{style}" edge="1" source="{src}" target="{tgt}" parent="1">{geo}</mxCell>\n'

els = ""
# ── Panel (a): Training ──────────────────────────────────────────────────────
els += R("pa","","rounded=1;fillColor=#F7F7F7;strokeColor=#BDBDBD;strokeWidth=2;arcSize=2;",20,20,1620,480)
els += R("paL","(a)  Training Pipeline","text;html=1;strokeColor=none;fillColor=none;align=left;fontSize=16;fontStyle=1;fontColor=#333;",40,28,360,28)

# Seed
els += R("seed","Seed QA&#xa;Corpus","shape=cylinder3;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;size=10;fillColor=#FFF9C4;strokeColor=#F9A825;strokeWidth=2;fontSize=12;fontStyle=1;",44,180,110,80)

# Challenger
els += R("C","","rounded=1;fillColor=#FFF0F0;strokeColor=#FFAB91;strokeWidth=2;",190,70,420,380)
els += R("CH","(A)  Challenger","rounded=0;fillColor=#FFCCBC;strokeColor=#FFAB91;strokeWidth=2;fontSize=14;fontStyle=1;align=center;html=1;",190,70,420,34)
els += R("CI","Qwen2.5-3B  GRPO-Batch","text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=10;fontColor=#555;",200,110,400,18)
els += R("CR","Reward: solver difficulty","rounded=1;fillColor=#FFE0B2;strokeColor=#FFAB40;strokeWidth=1.5;fontSize=10;fontStyle=1;html=1;",220,134,360,26)
els += R("Cdiv","","line;strokeColor=#FFCCBC;strokeWidth=1;fillColor=none;",208,168,388,8)
els += R("Csec","Multi-hop Generation","text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=12;fontStyle=1;fontColor=#BF360C;",200,176,400,20)
# hop chain compact horizontal
for i,(id_,lbl,fc,fx) in enumerate([("n0","Q0","#FFF9C4",200),("n1","Q1","#FFEBEE",272),("n2","Q2","#FFEBEE",344),("n3","Q_hard","#FFCCBC",420)]):
    els += R(id_,lbl,f"rounded=1;fillColor={fc};strokeColor=#FFAB91;strokeWidth=1.5;fontSize=12;fontStyle=1;html=1;",fx+10,204,56,30)
for s,t,id_ in [("n0","n1","ne01"),("n1","n2","ne12"),("n2","n3","ne23")]:
    els += E(id_,"","endArrow=block;endFill=1;html=1;strokeColor=#FFAB91;strokeWidth=1.5;",s,t)
els += R("sr1","search","ellipse;fillColor=#E3F2FD;strokeColor=#90CAF9;fontSize=9;fontColor=#1565C0;html=1;",272+14,246,42,20)
els += R("sr2","search","ellipse;fillColor=#E3F2FD;strokeColor=#90CAF9;fontSize=9;fontColor=#1565C0;html=1;",344+14,246,42,20)
els += E("evs1","","endArrow=open;endSize=4;html=1;strokeColor=#BBDEFB;","n1","sr1")
els += E("evs2","","endArrow=open;endSize=4;html=1;strokeColor=#BBDEFB;","n2","sr2")
els += E("ef1","fact","endArrow=block;endFill=1;html=1;strokeColor=#64B5F6;fontSize=8;fontColor=#1565C0;exitX=1;exitY=0.5;entryX=0;entryY=1;","sr1","n2")
els += E("ef2","fact","endArrow=block;endFill=1;html=1;strokeColor=#64B5F6;fontSize=8;fontColor=#1565C0;exitX=1;exitY=0.5;entryX=0;entryY=1;","sr2","n3")
els += R("Cnote","~50k questions/iter  hop ratio 4:3:2:1","text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=10;fontColor=#888;",200,276,400,18)
els += R("Cdata","","rounded=1;fillColor=#FFF3E0;strokeColor=#FFB74D;dashed=1;html=1;",218,302,376,60)
els += R("CdataL","D_k (generated questions)&#xa;become Solver training data","text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=11;fontColor=#5D4037;",218,302,376,60)
els += R("Cnxt","Checkpoint to Iter k+1","rounded=1;fillColor=#ECEFF1;strokeColor=#90A4AE;fontSize=10;fontStyle=1;html=1;",245,382,320,36)

# Solver
els += R("S","","rounded=1;fillColor=#F1F8E9;strokeColor=#AED581;strokeWidth=2;",650,70,400,380)
els += R("SH","(B)  Solver","rounded=0;fillColor=#DCEDC8;strokeColor=#AED581;strokeWidth=2;fontSize=14;fontStyle=1;align=center;html=1;",650,70,400,34)
els += R("SI","Qwen2.5-3B  GRPO group=5","text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=10;fontColor=#555;",660,110,380,18)
els += R("SR","Reward = Verifier + EM","rounded=1;fillColor=#FFE0B2;strokeColor=#FFAB40;strokeWidth=1.5;fontSize=10;fontStyle=1;html=1;",690,134,320,26)
els += R("Sdiv","","line;strokeColor=#DCEDC8;strokeWidth=1;fillColor=none;",668,168,368,8)
els += R("Ssec","Multi-turn Reasoning","text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=12;fontStyle=1;fontColor=#33691E;",660,176,380,20)
# vertical chain compact
els += R("Sq","Multi-hop Q","rounded=1;fillColor=#FFCCBC;strokeColor=#FF7043;fontSize=11;fontStyle=1;html=1;",775,204,150,28)
for id_,lbl,fc,sc,y in [("Ss1","Search 1","#E3F2FD","#90CAF9",244),("St1","Think","#F3E5F5","#CE93D8",282),("Ss2","Search 2","#E3F2FD","#90CAF9",320),("St2","Think","#F3E5F5","#CE93D8",358)]:
    els += R(id_,lbl,f"rounded=1;fillColor={fc};strokeColor={sc};strokeWidth=1.5;fontSize=11;html=1;",775,y,150,26)
els += R("Sans","answer","rounded=1;fillColor=#DCEDC8;strokeColor=#7CB342;strokeWidth=2;fontSize=12;fontStyle=1;fontColor=#33691E;html=1;",775,396,150,26)
for s,t,id_ in [("Sq","Ss1","sa1"),("Ss1","St1","sa2"),("St1","Ss2","sa3"),("Ss2","St2","sa4"),("St2","Sans","sa5")]:
    els += E(id_,"","endArrow=block;endFill=1;html=1;strokeColor=#AED581;strokeWidth=1.5;exitX=0.5;exitY=1;entryX=0.5;entryY=0;",s,t)
els += R("Snxt","Checkpoint to Iter k+1","rounded=1;fillColor=#ECEFF1;strokeColor=#90A4AE;fontSize=10;fontStyle=1;html=1;",710,432,280,36)

# Verifier
els += R("V","","rounded=1;fillColor=#F3E5F5;strokeColor=#CE93D8;strokeWidth=2;",1090,70,500,380)
els += R("VH","(C)  Verifier","rounded=0;fillColor=#E1BEE7;strokeColor=#CE93D8;strokeWidth=2;fontSize=14;fontStyle=1;align=center;html=1;",1090,70,500,34)
els += R("VI","Qwen2.5-3B  GRPO group=5","text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=10;fontColor=#555;",1100,110,480,18)
els += R("VR","Reward = verification accuracy","rounded=1;fillColor=#FFE0B2;strokeColor=#FFAB40;strokeWidth=1.5;fontSize=10;fontStyle=1;html=1;",1130,134,400,26)
els += R("Vdiv","","line;strokeColor=#E1BEE7;strokeWidth=1;fillColor=none;",1108,168,468,8)
els += R("Vsec","Backward Verification","text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=12;fontStyle=1;fontColor=#6A1B9A;",1100,176,480,20)
els += R("Vin","(Q, proposed answer)","rounded=1;fillColor=#FFF9C4;strokeColor=#F9A825;strokeWidth=1.5;fontSize=11;fontStyle=1;html=1;",1195,204,210,28)
els += R("Vsr","Backward Search","rounded=1;fillColor=#E3F2FD;strokeColor=#90CAF9;strokeWidth=1.5;fontSize=11;fontColor=#1565C0;html=1;",1200,244,200,26)
els += R("Vth","Reason and Compare","rounded=1;fillColor=#F3E5F5;strokeColor=#CE93D8;strokeWidth=1.5;fontSize=11;fontColor=#6A1B9A;html=1;",1200,282,200,26)
els += R("Vout","yes / no","rounded=1;fillColor=#E1BEE7;strokeColor=#8E24AA;strokeWidth=2;fontSize=14;fontStyle=1;fontColor=#4A148C;html=1;",1210,320,180,32)
for s,t,id_ in [("Vin","Vsr","va1"),("Vsr","Vth","va2"),("Vth","Vout","va3")]:
    els += E(id_,"","endArrow=block;endFill=1;html=1;strokeColor=#CE93D8;strokeWidth=1.5;exitX=0.5;exitY=1;entryX=0.5;entryY=0;",s,t)
els += R("Vdata","","rounded=1;fillColor=#FFF3E0;strokeColor=#FFB74D;dashed=1;html=1;",1110,364,470,56)
els += R("VdataL","Training pairs: (Q, correct ans) pos  (Q, wrong ans) neg&#xa;12,800 pairs/iter  yes/no becomes Solver reward","text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=10;fontColor=#5D4037;",1110,364,470,56)
els += R("Vnxt","Checkpoint to Iter k+1","rounded=1;fillColor=#ECEFF1;strokeColor=#90A4AE;fontSize=10;fontStyle=1;html=1;",1200,438,260,30)

# main arrows panel a
els += E("msc","","endArrow=block;endFill=1;html=1;strokeColor=#F9A825;strokeWidth=2.5;exitX=1;exitY=0.5;entryX=0;entryY=0.5;","seed","C")
els += E("mAB","Multi-hop Q","endArrow=block;endFill=1;html=1;strokeColor=#FF7043;strokeWidth=2.5;fontSize=12;fontStyle=1;fontColor=#BF360C;exitX=1;exitY=0.4;entryX=0;entryY=0.4;","C","S")
els += E("mBC","(Q, Answer)","endArrow=block;endFill=1;html=1;strokeColor=#7CB342;strokeWidth=2.5;fontSize=12;fontStyle=1;fontColor=#33691E;exitX=1;exitY=0.4;entryX=0;entryY=0.4;","S","V")
els += E("rvs","reward","endArrow=block;endFill=1;html=1;strokeColor=#FFAB40;strokeWidth=2;dashed=1;dashPattern=6 4;fontSize=11;fontStyle=1;fontColor=#E65100;exitX=0.5;exitY=1;entryX=0.5;entryY=1;","V","S",[(1340,470),(850,470)])
els += E("rsc","difficulty","endArrow=block;endFill=1;html=1;strokeColor=#FFAB40;strokeWidth=2;dashed=1;dashPattern=6 4;fontSize=11;fontStyle=1;fontColor=#E65100;exitX=0.5;exitY=1;entryX=0.5;entryY=1;","S","C",[(850,484),(400,484)])

# ── Panel (b): Inference ─────────────────────────────────────────────────────
els += R("pb","","rounded=1;fillColor=#F7F7F7;strokeColor=#BDBDBD;strokeWidth=2;arcSize=2;",20,520,1620,440)
els += R("pbL","(b)  Inference  (test time)","text;html=1;strokeColor=none;fillColor=none;align=left;fontSize=16;fontStyle=1;fontColor=#333;",40,528,360,28)

# frozen model icons (3D-style with shadow)
for id_,lbl,fc,sc,x in [("fC","Challenger*","#FFCCBC","#FFAB91",100),("fS","Solver*","#DCEDC8","#AED581",440),("fV","Verifier*","#E1BEE7","#CE93D8",780)]:
    els += R(id_+"sh","",f"rounded=1;fillColor=#CCCCCC;strokeColor=none;",x+6,556,160,52)
    els += R(id_,lbl,f"rounded=1;fillColor={fc};strokeColor={sc};strokeWidth=2;fontSize=13;fontStyle=1;align=center;html=1;",x,550,160,52)
    els += R(id_+"sub","* frozen","text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=9;fontColor=#999;fontStyle=2;",x,604,160,16)

# inference flow
els += R("iq","Test Question","rounded=1;fillColor=#FFF9C4;strokeColor=#F9A825;strokeWidth=2;fontSize=13;fontStyle=1;html=1;",1060,550,200,52)
els += R("isearch1","Search","rounded=1;fillColor=#E3F2FD;strokeColor=#90CAF9;strokeWidth=1.5;fontSize=12;fontColor=#1565C0;html=1;",1300,550,120,26)
els += R("ithink1","Think","rounded=1;fillColor=#F3E5F5;strokeColor=#CE93D8;strokeWidth=1.5;fontSize=12;fontColor=#6A1B9A;html=1;",1300,588,120,26)
els += R("isearch2","Search","rounded=1;fillColor=#E3F2FD;strokeColor=#90CAF9;strokeWidth=1.5;fontSize=12;fontColor=#1565C0;html=1;",1300,626,120,26)
els += R("ithink2","Think","rounded=1;fillColor=#F3E5F5;strokeColor=#CE93D8;strokeWidth=1.5;fontSize:12;fontColor=#6A1B9A;html=1;",1300,664,120,26)
els += R("ians","Final Answer","rounded=1;fillColor=#DCEDC8;strokeColor=#7CB342;strokeWidth=2;fontSize=13;fontStyle=1;fontColor=#33691E;html=1;",1300,702,120,36)
for s,t,id_ in [("isearch1","ithink1","ia1"),("ithink1","isearch2","ia2"),("isearch2","ithink2","ia3"),("ithink2","ians","ia4")]:
    els += E(id_,"","endArrow=block;endFill=1;html=1;strokeColor=#AED581;strokeWidth=1.5;exitX=0.5;exitY=1;entryX=0.5;entryY=0;",s,t)
els += E("iq2s","","endArrow=block;endFill=1;html=1;strokeColor=#F9A825;strokeWidth=2;exitX=1;exitY=0.5;entryX=0;entryY=0.5;","iq","isearch1")

# inference step labels
els += R("il1","1. Challenger generates&#xa;multi-hop test question","text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=10;fontColor=#555;",80,616,200,40)
els += R("il2","2. Solver answers with&#xa;iterative search","text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=10;fontColor=#555;",400,616,200,40)
els += R("il3","3. Verifier checks&#xa;answer correctness","text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=10;fontColor=#555;",740,616,200,40)
els += R("il4","4. Multi-turn&#xa;search chain","text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=10;fontColor=#555;",1420,616,140,40)

# Retrieval bar bottom of panel b
els += R("ret","","rounded=1;fillColor=#E3F2FD;strokeColor=#90CAF9;strokeWidth=2;",40,880,1580,60)
els += R("retL","Shared Retrieval Module — Wikipedia (wiki-18)  FAISS  E5-base-v2  Top-3 passages","text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=13;fontStyle=1;fontColor=#1565C0;",40,880,1580,60)

with open(out, 'w', encoding='utf-8') as f:
    f.write(wrap(els))
print("V2 saved")
