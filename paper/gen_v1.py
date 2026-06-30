out = "d:/paper/drzero/paper/figure_v1_triangle.drawio"

def wrap(inner):
    return '<mxfile host="app.diagrams.net" version="24.7.17"><diagram id="fig" name="Page-1"><mxGraphModel dx="1600" dy="900" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1600" pageHeight="1000" math="0" shadow="0"><root><mxCell id="0"/><mxCell id="1" parent="0"/>' + inner + '</root></mxGraphModel></diagram></mxfile>'

def R(id_, val, style, x, y, w, h, parent="1", vertex="1"):
    return f'<mxCell id="{id_}" value="{val}" style="{style}" vertex="{vertex}" parent="{parent}"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>\n'

def E(id_, val, style, src, tgt, pts=None, parent="1"):
    geo = '<mxGeometry relative="1" as="geometry"/>'
    if pts:
        geo = '<mxGeometry relative="1" as="geometry"><Array as="points">' + ''.join(f'<mxPoint x="{p[0]}" y="{p[1]}"/>' for p in pts) + '</Array></mxGeometry>'
    return f'<mxCell id="{id_}" value="{val}" style="{style}" edge="1" source="{src}" target="{tgt}" parent="{parent}">{geo}</mxCell>\n'

els = ""

# background
els += R("bg","","rounded=1;fillColor=#F8F8F8;strokeColor=#CCCCCC;strokeWidth=2;dashed=1;dashPattern=8 4;",20,20,1360,860)
els += R("ttl","Self-Play Training Pipeline","text;html=1;strokeColor=none;fillColor=none;align=left;fontSize=18;fontStyle=1;fontColor=#333;",44,30,400,30)
els += R("itl","Iteration k","text;html=1;strokeColor=none;fillColor=none;align=right;fontSize=13;fontStyle=3;fontColor=#888;",1150,32,200,24)

# Retrieval center
els += R("ret","","rounded=1;fillColor=#E3F2FD;strokeColor=#90CAF9;strokeWidth=2;",530,370,340,70)
els += R("retL","Shared Retrieval Module","text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=13;fontStyle=1;fontColor=#1565C0;",530,374,340,22)
els += R("retD","Wikipedia  FAISS  E5-base-v2  Top-3","text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=10;fontColor=#1565C0;",530,398,340,18)

# Seed
els += R("seed","Seed QA Corpus","shape=cylinder3;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;size=10;fillColor=#FFF9C4;strokeColor=#F9A825;strokeWidth=2;fontSize=11;fontStyle=1;",44,290,120,76)

# CHALLENGER top-center
els += R("C","","rounded=1;fillColor=#FFF0F0;strokeColor=#FFAB91;strokeWidth=2;",480,60,440,260)
els += R("CH","(A)  Challenger  (Proposer)","rounded=0;fillColor=#FFCCBC;strokeColor=#FFAB91;strokeWidth=2;fontSize=15;fontStyle=1;align=center;html=1;",480,60,440,36)
els += R("CI","Qwen2.5-3B  GRPO-Batch  group=1  reward_n=5","text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=11;fontColor=#555;",490,102,420,20)
els += R("CR","Reward proportional to solver difficulty","rounded=1;fillColor=#FFE0B2;strokeColor=#FFAB40;strokeWidth=1.5;fontSize=11;fontStyle=1;html=1;",520,128,360,28)
# hop labels
els += R("lh0","seed","text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=9;fontColor=#999;",497,166,50,14)
els += R("lh1","hop 1","text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=9;fontColor=#E64A19;fontStyle=1;",573,166,56,14)
els += R("lh2","hop 2","text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=9;fontColor=#E64A19;fontStyle=1;",659,166,56,14)
els += R("lhx","output","text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=9;fontColor=#BF360C;fontStyle=1;",743,166,66,14)
# hop nodes
els += R("cq0","Q0","rounded=1;fillColor=#FFFFFF;strokeColor=#FFAB91;strokeWidth=1.5;fontSize=13;fontStyle=1;html=1;",497,182,50,30)
els += R("cq1","Q1","rounded=1;fillColor=#FFEBEE;strokeColor=#EF9A9A;strokeWidth=1.5;fontSize=13;fontStyle=1;html=1;",573,182,56,30)
els += R("cq2","Q2","rounded=1;fillColor=#FFEBEE;strokeColor=#EF9A9A;strokeWidth=1.5;fontSize=13;fontStyle=1;html=1;",659,182,56,30)
els += R("cqh","Q_hard","rounded=1;fillColor=#FFCCBC;strokeColor=#FF7043;strokeWidth=2;fontSize=11;fontStyle=1;html=1;",743,182,68,30)
els += E("ce01","","endArrow=block;endFill=1;html=1;strokeColor=#FFAB91;strokeWidth=1.5;","cq0","cq1")
els += E("ce12","","endArrow=block;endFill=1;html=1;strokeColor=#FFAB91;strokeWidth=1.5;","cq1","cq2")
els += E("ce2h","","endArrow=block;endFill=1;html=1;strokeColor=#FFAB91;strokeWidth=1.5;","cq2","cqh")
els += R("cs1","search","ellipse;fillColor=#E3F2FD;strokeColor=#90CAF9;strokeWidth=1.5;fontSize=9;fontColor=#1565C0;html=1;",568,224,66,22)
els += R("cs2","search","ellipse;fillColor=#E3F2FD;strokeColor=#90CAF9;strokeWidth=1.5;fontSize=9;fontColor=#1565C0;html=1;",654,224,66,22)
els += E("cvs1","","endArrow=open;endSize=4;html=1;strokeColor=#BBDEFB;","cq1","cs1")
els += E("cvs2","","endArrow=open;endSize=4;html=1;strokeColor=#BBDEFB;","cq2","cs2")
els += E("cf1","fact1","endArrow=block;endFill=1;html=1;strokeColor=#64B5F6;strokeWidth=1.2;fontSize=9;fontColor=#1565C0;exitX=1;exitY=0.5;exitDx=0;exitDy=0;entryX=0;entryY=1;entryDx=0;entryDy=0;","cs1","cq2")
els += E("cf2","fact2","endArrow=block;endFill=1;html=1;strokeColor=#64B5F6;strokeWidth=1.2;fontSize=9;fontColor=#1565C0;exitX=1;exitY=0.5;exitDx=0;exitDy=0;entryX=0;entryY=1;entryDx=0;entryDy=0;","cs2","cqh")
els += R("cratio","hop ratio 4:3:2:1  ~50k questions/iteration","text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=10;fontColor=#888;",490,256,420,18)

# SOLVER bottom-left
els += R("S","","rounded=1;fillColor=#F1F8E9;strokeColor=#AED581;strokeWidth=2;",50,490,380,350)
els += R("SH","(B)  Solver","rounded=0;fillColor=#DCEDC8;strokeColor=#AED581;strokeWidth=2;fontSize=15;fontStyle=1;align=center;html=1;",50,490,380,36)
els += R("SI","Qwen2.5-3B  GRPO group=5  Multi-turn","text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=11;fontColor=#555;",60,532,360,18)
els += R("SR","Reward = Verifier judgment + EM score","rounded=1;fillColor=#FFE0B2;strokeColor=#FFAB40;strokeWidth=1.5;fontSize=11;fontStyle=1;html=1;",90,556,300,28)
els += R("Ssec","Multi-turn Retrieval-Augmented Reasoning","text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=12;fontStyle=1;fontColor=#33691E;",60,590,360,20)
els += R("Sq","Multi-hop Q","rounded=1;fillColor=#FFCCBC;strokeColor=#FF7043;fontSize=11;fontStyle=1;html=1;",175,618,130,28)
els += R("Ss1","Search 1","rounded=1;fillColor=#E3F2FD;strokeColor=#90CAF9;strokeWidth=1.5;fontSize=11;fontColor=#1565C0;html=1;",170,658,140,26)
els += R("St1","Think","rounded=1;fillColor=#F3E5F5;strokeColor=#CE93D8;strokeWidth=1.5;fontSize=11;fontColor=#6A1B9A;html=1;",170,696,140,26)
els += R("Ss2","Search 2","rounded=1;fillColor=#E3F2FD;strokeColor=#90CAF9;strokeWidth=1.5;fontSize=11;fontColor=#1565C0;html=1;",170,734,140,26)
els += R("St2","Think","rounded=1;fillColor=#F3E5F5;strokeColor=#CE93D8;strokeWidth=1.5;fontSize=11;fontColor=#6A1B9A;html=1;",170,772,140,26)
els += R("Sans","answer","rounded=1;fillColor=#DCEDC8;strokeColor=#7CB342;strokeWidth=2;fontSize=12;fontStyle=1;fontColor=#33691E;html=1;",170,810,140,26)
for s,t,id_ in [("Sq","Ss1","Sa1"),("Ss1","St1","Sa2"),("St1","Ss2","Sa3"),("Ss2","St2","Sa4"),("St2","Sans","Sa5")]:
    els += E(id_,"","endArrow=block;endFill=1;html=1;strokeColor=#AED581;strokeWidth=1.5;exitX=0.5;exitY=1;exitDx=0;exitDy=0;entryX=0.5;entryY=0;entryDx=0;entryDy=0;",s,t)

# VERIFIER bottom-right
els += R("V","","rounded=1;fillColor=#F3E5F5;strokeColor=#CE93D8;strokeWidth=2;",970,490,380,350)
els += R("VH","(C)  Verifier","rounded=0;fillColor=#E1BEE7;strokeColor=#CE93D8;strokeWidth=2;fontSize=15;fontStyle=1;align=center;html=1;",970,490,380,36)
els += R("VI","Qwen2.5-3B  GRPO group=5  Multi-turn","text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=11;fontColor=#555;",980,532,360,18)
els += R("VR","Reward = verification accuracy vs. ground truth","rounded=1;fillColor=#FFE0B2;strokeColor=#FFAB40;strokeWidth=1.5;fontSize=11;fontStyle=1;html=1;",1000,556,320,28)
els += R("Vsec","Backward Verification via Search","text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=12;fontStyle=1;fontColor=#6A1B9A;",980,590,360,20)
els += R("Vin","(Q, proposed answer)","rounded=1;fillColor=#FFF9C4;strokeColor=#F9A825;strokeWidth=1.5;fontSize=11;fontStyle=1;html=1;",1055,618,210,28)
els += R("Vsr","Backward Search","rounded=1;fillColor=#E3F2FD;strokeColor=#90CAF9;strokeWidth=1.5;fontSize=11;fontColor=#1565C0;html=1;",1060,658,200,26)
els += R("Vth","Reason and Compare","rounded=1;fillColor=#F3E5F5;strokeColor=#CE93D8;strokeWidth=1.5;fontSize=11;fontColor=#6A1B9A;html=1;",1060,696,200,26)
els += R("Vout","yes / no","rounded=1;fillColor=#E1BEE7;strokeColor=#8E24AA;strokeWidth=2;fontSize=14;fontStyle=1;fontColor=#4A148C;html=1;",1080,736,160,34)
for s,t,id_ in [("Vin","Vsr","Va1"),("Vsr","Vth","Va2"),("Vth","Vout","Va3")]:
    els += E(id_,"","endArrow=block;endFill=1;html=1;strokeColor=#CE93D8;strokeWidth=1.5;exitX=0.5;exitY=1;exitDx=0;exitDy=0;entryX=0.5;entryY=0;entryDx=0;entryDy=0;",s,t)

# main flow arrows
els += E("msc","","endArrow=block;endFill=1;html=1;strokeColor=#F9A825;strokeWidth=2.5;exitX=1;exitY=0.5;exitDx=0;exitDy=0;entryX=0;entryY=0.5;entryDx=0;entryDy=0;","seed","C")
els += E("mcs","Multi-hop Q","endArrow=block;endFill=1;html=1;strokeColor=#FF7043;strokeWidth=2.5;fontSize=11;fontStyle=1;fontColor=#BF360C;exitX=0;exitY=1;exitDx=0;exitDy=0;entryX=0.5;entryY=0;entryDx=0;entryDy=0;","C","S",[(480,440),(240,440)])
els += E("msv","(Q, Answer)","endArrow=block;endFill=1;html=1;strokeColor=#7CB342;strokeWidth=2.5;fontSize=11;fontStyle=1;fontColor=#33691E;exitX=1;exitY=0.5;exitDx=0;exitDy=0;entryX=0;entryY=0.5;entryDx=0;entryDy=0;","S","V",[(540,665),(860,665)])
els += E("mvc","Q also sent","endArrow=block;endFill=1;html=1;strokeColor=#FF7043;strokeWidth=2;dashed=1;fontSize=10;fontColor=#BF360C;exitX=1;exitY=0;exitDx=0;exitDy=0;entryX=1;exitY=0.5;entryX=1;entryY=0.5;entryDx=0;entryDy=0;","C","V",[(920,200),(920,440),(1350,440),(1350,500)])
# reward loops
els += E("rvs","reward","endArrow=block;endFill=1;html=1;strokeColor=#FFAB40;strokeWidth=2;dashed=1;dashPattern=6 4;fontSize=11;fontStyle=1;fontColor=#E65100;exitX=0;exitY=0.3;exitDx=0;exitDy=0;entryX=1;entryY=0.3;entryDx=0;entryDy=0;","V","S",[(860,583),(540,583)])
els += E("rsc","difficulty","endArrow=block;endFill=1;html=1;strokeColor=#FFAB40;strokeWidth=2;dashed=1;dashPattern=6 4;fontSize=11;fontStyle=1;fontColor=#E65100;exitX=0.5;exitY=0;exitDx=0;exitDy=0;entryX=0;entryY=0.7;entryDx=0;entryDy=0;","S","C",[(240,450),(480,450)])
# retrieval lines
els += E("rC","","endArrow=open;endSize=6;html=1;strokeColor=#90CAF9;strokeWidth=1.5;dashed=1;exitX=0.5;exitY=0;exitDx=0;exitDy=0;entryX=0.5;entryY=1;entryDx=0;entryDy=0;","ret","C")
els += E("rS","","endArrow=open;endSize=6;html=1;strokeColor=#90CAF9;strokeWidth=1.5;dashed=1;exitX=0;exitY=0.5;exitDx=0;exitDy=0;entryX=1;entryY=0.5;entryDx=0;entryDy=0;","ret","S")
els += E("rV","","endArrow=open;endSize=6;html=1;strokeColor=#90CAF9;strokeWidth=1.5;dashed=1;exitX=1;exitY=0.5;exitDx=0;exitDy=0;entryX=0;entryY=0.5;entryDx=0;entryDy=0;","ret","V")

with open(out, 'w', encoding='utf-8') as f:
    f.write(wrap(els))
print("V1 saved")
