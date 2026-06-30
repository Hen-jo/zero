def wrap(inner, pw=1700, ph=1000):
    return f'<mxfile host="app.diagrams.net" version="24.7.17"><diagram id="fig" name="Page-1"><mxGraphModel dx="1600" dy="900" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="{pw}" pageHeight="{ph}" math="0" shadow="0"><root><mxCell id="0"/><mxCell id="1" parent="0"/>' + inner + '</root></mxGraphModel></diagram></mxfile>'

def R(id_, val, style, x, y, w, h):
    return f'<mxCell id="{id_}" value="{val}" style="{style}" vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>\n'

def E(id_, val, style, src, tgt, pts=None):
    geo = '<mxGeometry relative="1" as="geometry"/>'
    if pts:
        geo = '<mxGeometry relative="1" as="geometry"><Array as="points">' + ''.join(f'<mxPoint x="{p[0]}" y="{p[1]}"/>' for p in pts) + '</Array></mxGeometry>'
    return f'<mxCell id="{id_}" value="{val}" style="{style}" edge="1" source="{src}" target="{tgt}" parent="1">{geo}</mxCell>\n'

# ════════════════════════════════════════
# V3: Layer Cake (iterations stacked)
# ════════════════════════════════════════
v3 = ""
v3 += R("ttl","Iterative Self-Play Training","text;html=1;strokeColor=none;fillColor=none;align=left;fontSize=20;fontStyle=1;fontColor=#333;",30,18,500,32)
v3 += R("sub","All three models improve across iterations — checkpoints carry forward","text;html=1;strokeColor=none;fillColor=none;align=left;fontSize=12;fontColor=#888;fontStyle=2;",30,52,700,20)

ITER_COLORS = [("#FFF0F0","#FFAB91","#F1F8E9","#AED581","#F3E5F5","#CE93D8","#FFCCBC","#DCEDC8","#E1BEE7"),
               ("#FFE4E4","#FF7043","#E8F5E9","#66BB6A","#EDE7F6","#AB47BC","#FF7043","#66BB6A","#AB47BC"),
               ("#FFD6D6","#E64A19","#D7EBD7","#388E3C","#E8D5F5","#7B1FA2","#E64A19","#388E3C","#7B1FA2")]

iter_labels = ["Iteration 1  (shared init)", "Iteration 2", "Iteration 3"]

for it in range(3):
    cy,ch = 88 + it*230, 200
    cfc,csc,sfc,ssc,vfc,vsc,chc,shc,vhc = ITER_COLORS[it]
    prefix = f"i{it}"
    alpha = 100 - it*10
    # iter background
    v3 += R(f"{prefix}bg","",f"rounded=1;fillColor=#FAFAFA;strokeColor=#CCCCCC;strokeWidth=1.5;dashed=1;",20,cy-8,1560,ch+16)
    v3 += R(f"{prefix}lbl",iter_labels[it],f"text;html=1;strokeColor=none;fillColor=none;align=left;fontSize=14;fontStyle=1;fontColor=#555;",36,cy,240,24)
    # 3 component boxes
    for id2,lbl,fc,sc,hc,x in [(f"{prefix}C","Challenger",cfc,csc,chc,260),(f"{prefix}S","Solver",sfc,ssc,shc,680),(f"{prefix}V","Verifier",vfc,vsc,vhc,1100)]:
        v3 += R(id2+"sh","",f"rounded=1;fillColor=#DDDDDD;strokeColor=none;opacity=60;",x+6,cy+30,380,146)
        v3 += R(id2,"",f"rounded=1;fillColor={fc};strokeColor={sc};strokeWidth=2;",x,cy+24,380,146)
        v3 += R(id2+"H",lbl,f"rounded=0;fillColor={hc};strokeColor={sc};strokeWidth=2;fontSize=14;fontStyle=1;align=center;html=1;",x,cy+24,380,34)
        v3 += R(id2+"m","Qwen2.5-3B  GRPO",f"text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=10;fontColor=#666;",x+10,cy+64,360,18)
    # hop chain inside Challenger
    cx = 260
    for ni,nid,nlbl,nfc in [(0,f"{prefix}q0","Q0","#FFFFFF"),(1,f"{prefix}q1","Q1","#FFEBEE"),(2,f"{prefix}q2","Q2","#FFEBEE"),(3,f"{prefix}qh","Qh",chc)]:
        v3 += R(nid,nlbl,f"rounded=1;fillColor={nfc};strokeColor={csc};strokeWidth=1.5;fontSize=11;fontStyle=1;html=1;",cx+8+ni*88,cy+88,68,28)
    for s,t,eid in [(f"{prefix}q0",f"{prefix}q1",f"{prefix}e01"),(f"{prefix}q1",f"{prefix}q2",f"{prefix}e12"),(f"{prefix}q2",f"{prefix}qh",f"{prefix}e2h")]:
        v3 += E(eid,"","endArrow=block;endFill=1;html=1;strokeColor=#FFAB91;strokeWidth=1.3;",s,t)
    # search→think inside Solver
    sx = 680
    v3 += R(f"{prefix}ss","Search","rounded=1;fillColor=#E3F2FD;strokeColor=#90CAF9;fontSize=10;fontColor=#1565C0;html=1;",sx+80,cy+88,100,26)
    v3 += R(f"{prefix}st","Think","rounded=1;fillColor=#F3E5F5;strokeColor=#CE93D8;fontSize=10;fontColor=#6A1B9A;html=1;",sx+200,cy+88,100,26)
    v3 += R(f"{prefix}sa","answer","rounded=1;fillColor={shc};strokeColor={ssc};strokeWidth=2;fontSize=10;fontStyle=1;html=1;",sx+320,cy+88,50,26)
    v3 += E(f"{prefix}es1","","endArrow=block;endFill=1;html=1;strokeColor=#AED581;strokeWidth=1.3;",f"{prefix}ss",f"{prefix}st")
    v3 += E(f"{prefix}es2","","endArrow=block;endFill=1;html=1;strokeColor=#AED581;strokeWidth=1.3;",f"{prefix}st",f"{prefix}sa")
    # yes/no inside Verifier
    vx = 1100
    v3 += R(f"{prefix}vyn","yes / no",f"rounded=1;fillColor={vhc};strokeColor={vsc};strokeWidth=2;fontSize=13;fontStyle=1;fontColor=#4A148C;html=1;",vx+140,cy+88,100,28)
    # main arrows between components
    v3 += E(f"{prefix}mcs","Multi-hop Q","endArrow=block;endFill=1;html=1;strokeColor=#FF7043;strokeWidth=2;fontSize=10;fontStyle=1;fontColor=#BF360C;exitX=1;exitY=0.5;entryX=0;entryY=0.5;",f"{prefix}C",f"{prefix}S")
    v3 += E(f"{prefix}msv","(Q,A)","endArrow=block;endFill=1;html=1;strokeColor=#7CB342;strokeWidth=2;fontSize=10;fontStyle=1;fontColor=#33691E;exitX=1;exitY=0.5;entryX=0;entryY=0.5;",f"{prefix}S",f"{prefix}V")

# checkpoint arrows between iterations
for it in range(2):
    y1 = 88 + it*230 + 170
    y2 = 88 + (it+1)*230 + 24
    for xc,id2 in [(450,"ckC"),(870,"ckS"),(1290,"ckV")]:
        v3 += R(f"ck{it}{id2}","checkpoint","text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=9;fontColor=#90A4AE;fontStyle=2;",xc-40,y1+2,100,14)
        v3 += f'<mxCell id="arr{it}{id2}" value="" style="endArrow=block;endFill=1;html=1;strokeColor=#90A4AE;strokeWidth=1.5;dashed=1;exitX=0.5;exitY=1;entryX=0.5;entryY=0;" edge="1" parent="1"><mxGeometry relative="1" as="geometry"><mxPoint x="{xc}" y="{y1}" as="sourcePoint"/><mxPoint x="{xc}" y="{y2}" as="targetPoint"/></mxGeometry></mxCell>\n'

# performance bar on right
v3 += R("perf","Performance","text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=12;fontStyle=1;fontColor=#333;",1580,88,100,20)
for it,(h,fc) in enumerate([(60,"#FFCCBC"),(90,"#AED581"),(130,"#CE93D8")]):
    by = 88 + it*230 + 24 + (146-h)
    v3 += R(f"pb{it}","",f"rounded=1;fillColor={fc};strokeColor=#888;strokeWidth=1;",1590,by,30,h)
    v3 += R(f"pbl{it}",f"Iter {it+1}","text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=9;fontColor=#555;",1584,by+h+2,42,14)

# retrieval
v3 += R("ret","","rounded=1;fillColor=#E3F2FD;strokeColor=#90CAF9;strokeWidth=2;",20,784,1540,56)
v3 += R("retL","Shared Retrieval — Wikipedia  FAISS  E5-base-v2  Top-3  (used by all models in all iterations)","text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=12;fontStyle=1;fontColor=#1565C0;",20,784,1540,56)

with open("d:/paper/drzero/paper/figure_v3_layercake.drawio","w",encoding="utf-8") as f:
    f.write(wrap(v3, pw=1700, ph=880))
print("V3 saved")

# ════════════════════════════════════════
# V4: Factory Line (horizontal conveyor)
# ════════════════════════════════════════
v4 = ""
v4 += R("ttl","Self-Play Training Pipeline","text;html=1;strokeColor=none;fillColor=none;align=left;fontSize=20;fontStyle=1;fontColor=#333;",30,18,500,32)

# Conveyor belt background
v4 += R("belt","","rounded=0;fillColor=#ECEFF1;strokeColor=#90A4AE;strokeWidth=2;",30,380,1580,40)
v4 += R("beltL","conveyor of training data","text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=10;fontStyle=2;fontColor=#90A4AE;",30,384,1580,32)
# Belt motion arrows
for x in [100,300,500,700,900,1100,1300,1500]:
    v4 += R(f"ba{x}","","endArrow=open;endSize=5;html=1;strokeColor=#BDBDBD;strokeWidth=1;" if False else "",0,0,1,1)  # skip, use edge
    v4 += f'<mxCell id="bta{x}" value="" style="endArrow=open;endSize=5;html=1;strokeColor=#BDBDBD;strokeWidth=1;" edge="1" parent="1"><mxGeometry relative="1" as="geometry"><mxPoint x="{x}" y="400" as="sourcePoint"/><mxPoint x="{x+40}" y="400" as="targetPoint"/></mxGeometry></mxCell>\n'

# 3D shadow + box for each station
stations = [
    ("C","Challenger&#xa;(Proposer)","#FFF0F0","#FFAB91","#FFCCBC",120,70,340,280),
    ("S","Solver","#F1F8E9","#AED581","#DCEDC8",570,70,340,280),
    ("V","Verifier","#F3E5F5","#CE93D8","#E1BEE7",1020,70,340,280),
]
for id_,lbl,fc,sc,hc,x,y,w,h in stations:
    # 3D shadow
    v4 += R(id_+"sh3","",f"rounded=1;fillColor=#CCCCCC;strokeColor=none;opacity=70;",x+10,y+10,w,h)
    v4 += R(id_+"sh2","",f"rounded=1;fillColor=#BBBBBB;strokeColor=none;opacity=50;",x+6,y+6,w,h)
    # main box
    v4 += R(id_,"",f"rounded=1;fillColor={fc};strokeColor={sc};strokeWidth=2;",x,y,w,h)
    v4 += R(id_+"H",lbl,f"rounded=0;fillColor={hc};strokeColor={sc};strokeWidth=2;fontSize=14;fontStyle=1;align=center;html=1;",x,y,w,38)
    v4 += R(id_+"I","Qwen2.5-3B  GRPO",f"text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=10;fontColor=#666;",x+10,y+44,w-20,18)
    v4 += R(id_+"R","Reward: special per model",f"rounded=1;fillColor=#FFE0B2;strokeColor=#FFAB40;fontSize=10;fontStyle=1;html=1;",x+20,y+68,w-40,26)

# Challenger internals: hop chain
for ni,nid,nlbl,nfc,nx in [(0,"Cq0","Q0","#FFF9C4",136),(1,"Cq1","Q1","#FFEBEE",210),(2,"Cq2","Q2","#FFEBEE",284),(3,"Cqh","Qh","#FFCCBC",358)]:
    v4 += R(nid,nlbl,f"rounded=1;fillColor={nfc};strokeColor=#FFAB91;strokeWidth=1.5;fontSize=11;fontStyle=1;html=1;",nx,170,60,28)
for s,t,eid in [("Cq0","Cq1","Ce01"),("Cq1","Cq2","Ce12"),("Cq2","Cqh","Ce2h")]:
    v4 += E(eid,"","endArrow=block;endFill=1;html=1;strokeColor=#FFAB91;strokeWidth=1.5;",s,t)
v4 += R("Csr1","search","ellipse;fillColor=#E3F2FD;strokeColor=#90CAF9;fontSize=9;fontColor=#1565C0;html=1;",214,208,52,20)
v4 += R("Csr2","search","ellipse;fillColor=#E3F2FD;strokeColor=#90CAF9;fontSize=9;fontColor=#1565C0;html=1;",288,208,52,20)
v4 += E("Cvs1","","endArrow=open;endSize=4;html=1;strokeColor=#BBDEFB;","Cq1","Csr1")
v4 += E("Cvs2","","endArrow=open;endSize=4;html=1;strokeColor=#BBDEFB;","Cq2","Csr2")

# Solver internals: horizontal search chain
for id_,lbl,fc,sc,x in [("Ss1","Search 1","#E3F2FD","#90CAF9",586),("St1","Think","#F3E5F5","#CE93D8",672),("Ss2","Search 2","#E3F2FD","#90CAF9",758),("St2","Think","#F3E5F5","#CE93D8",844)]:
    v4 += R(id_,lbl,f"rounded=1;fillColor={fc};strokeColor={sc};strokeWidth=1.5;fontSize=10;fontColor={'#1565C0' if 'Search' in lbl else '#6A1B9A'};html=1;",x,168,76,28)
for s,t,eid in [("Ss1","St1","sa1"),("St1","Ss2","sa2"),("Ss2","St2","sa3")]:
    v4 += E(eid,"","endArrow=block;endFill=1;html=1;strokeColor=#AED581;strokeWidth=1.5;",s,t)
v4 += R("Sans","answer","rounded=1;fillColor=#DCEDC8;strokeColor=#7CB342;strokeWidth=2;fontSize=11;fontStyle=1;fontColor=#33691E;html=1;",845+86,168,80,28)
v4 += E("sa4","","endArrow=block;endFill=1;html=1;strokeColor=#AED581;strokeWidth=1.5;","St2","Sans")

# Verifier internals
v4 += R("Vin","(Q, A)","rounded=1;fillColor=#FFF9C4;strokeColor=#F9A825;fontSize=10;fontStyle=1;html=1;",1040,168,70,28)
v4 += R("Vsr","Backward&#xa;Search","rounded=1;fillColor=#E3F2FD;strokeColor=#90CAF9;fontSize=10;fontColor=#1565C0;html=1;",1128,158,80,36)
v4 += R("Vyn","yes/no","rounded=1;fillColor=#E1BEE7;strokeColor=#8E24AA;strokeWidth=2;fontSize=13;fontStyle=1;fontColor=#4A148C;html=1;",1230,162,90,32)
v4 += E("va1","","endArrow=block;endFill=1;html=1;strokeColor=#CE93D8;strokeWidth=1.5;","Vin","Vsr")
v4 += E("va2","","endArrow=block;endFill=1;html=1;strokeColor=#CE93D8;strokeWidth=1.5;","Vsr","Vyn")

# Seed
v4 += R("seed","Seed&#xa;QA","shape=cylinder3;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;size=8;fillColor=#FFF9C4;strokeColor=#F9A825;strokeWidth=2;fontSize=11;fontStyle=1;",30,160,80,76)
v4 += E("msc","","endArrow=block;endFill=1;html=1;strokeColor=#F9A825;strokeWidth=2.5;exitX=1;exitY=0.5;entryX=0;entryY=0.5;","seed","C")

# Main horizontal arrows
v4 += E("mAB","Multi-hop Q","endArrow=block;endFill=1;html=1;strokeColor=#FF7043;strokeWidth=2.5;fontSize=12;fontStyle=1;fontColor=#BF360C;exitX=1;exitY=0.4;entryX=0;entryY=0.4;","C","S")
v4 += E("mBC","(Q, Answer)","endArrow=block;endFill=1;html=1;strokeColor=#7CB342;strokeWidth=2.5;fontSize=12;fontStyle=1;fontColor=#33691E;exitX=1;exitY=0.4;entryX=0;entryY=0.4;","S","V")

# Feedback belt (below, going right to left)
v4 += R("fbelt","","rounded=0;fillColor=#FFF8E1;strokeColor=#FFAB40;strokeWidth=2;dashed=1;dashPattern=10 5;",30,440,1580,36)
v4 += R("fbeltL","reward and difficulty signals (feedback)","text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=11;fontStyle=1;fontColor=#E65100;",30,440,1580,36)
# Feedback arrows right-to-left
v4 += E("rvs","reward","endArrow=block;endFill=1;html=1;strokeColor=#FFAB40;strokeWidth=2;dashed=1;fontSize=11;fontStyle=1;fontColor=#E65100;exitX=0;exitY=0.7;entryX=1;entryY=0.7;","V","S",[(870,478),(870,478)])
v4 += E("rsc","difficulty","endArrow=block;endFill=1;html=1;strokeColor=#FFAB40;strokeWidth=2;dashed=1;fontSize=11;fontStyle=1;fontColor=#E65100;exitX=0;exitY=0.7;entryX=1;entryY=0.7;","S","C",[(420,478),(420,478)])

# Retrieval ground
v4 += R("ret","","rounded=1;fillColor=#E3F2FD;strokeColor=#90CAF9;strokeWidth=2;",30,520,1580,60)
v4 += R("retL","Shared Retrieval Module — Wikipedia (wiki-18)  FAISS  E5-base-v2  Top-3 passages","text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=13;fontStyle=1;fontColor=#1565C0;",30,520,1580,60)
# Retrieval dashed lines up to each station
v4 += E("rC","","endArrow=open;endSize=6;html=1;strokeColor=#90CAF9;strokeWidth=1.5;dashed=1;exitX=0.19;exitY=0;entryX=0.5;entryY=1;","ret","C")
v4 += E("rS","","endArrow=open;endSize=6;html=1;strokeColor=#90CAF9;strokeWidth=1.5;dashed=1;exitX=0.5;exitY=0;entryX=0.5;entryY=1;","ret","S")
v4 += E("rV","","endArrow=open;endSize=6;html=1;strokeColor=#90CAF9;strokeWidth=1.5;dashed=1;exitX=0.81;exitY=0;entryX=0.5;entryY=1;","ret","V")

with open("d:/paper/drzero/paper/figure_v4_factory.drawio","w",encoding="utf-8") as f:
    f.write(wrap(v4, pw=1680, ph=620))
print("V4 saved")

# ════════════════════════════════════════
# V5: Multi-hop Tree (left) + Pipeline (right)
# ════════════════════════════════════════
v5 = ""
v5 += R("ttl","Self-Play Training Pipeline","text;html=1;strokeColor=none;fillColor=none;align=left;fontSize=20;fontStyle=1;fontColor=#333;",30,18,500,32)

# LEFT: Multi-hop tree panel
v5 += R("treepnl","","rounded=1;fillColor=#FFFDE7;strokeColor=#F9A825;strokeWidth=2;dashed=1;dashPattern=8 4;",20,60,560,800)
v5 += R("treetitle","Multi-hop Question Structure","text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=15;fontStyle=1;fontColor=#BF360C;",30,68,540,26)
v5 += R("treesub","(Challenger generates questions at each hop level)","text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=10;fontStyle=2;fontColor=#888;",30,96,540,18)

# Root: seed document
v5 += R("tr","Wikipedia Document","rounded=1;fillColor=#FFF9C4;strokeColor=#F9A825;strokeWidth=2;fontSize=13;fontStyle=1;html=1;",190,130,220,36)

# 1-hop branch
v5 += R("h1","1-hop Q","rounded=1;fillColor=#FFECB3;strokeColor=#FFB300;strokeWidth=1.5;fontSize=12;fontStyle=1;html=1;",60,210,140,30)
v5 += R("h1a","e.g. Who wrote X?","text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=9;fontColor=#888;fontStyle=2;",30,242,200,16)
v5 += E("etr1","1 search","endArrow=block;endFill=1;html=1;strokeColor=#FFB300;strokeWidth=1.5;fontSize=9;fontColor=#F57F17;exitX=0;exitY=1;entryX=0.5;entryY=0;","tr","h1",[(300,180),(130,180)])

# 2-hop branch
v5 += R("h2a","search 1","ellipse;fillColor=#E3F2FD;strokeColor=#90CAF9;fontSize=9;fontColor=#1565C0;html=1;",265,210,80,24)
v5 += R("h2b","2-hop Q","rounded=1;fillColor=#FFCCBC;strokeColor=#FF7043;strokeWidth=1.5;fontSize=12;fontStyle=1;html=1;",245,260,120,30)
v5 += E("etr2a","","endArrow=block;endFill=1;html=1;strokeColor=#90CAF9;strokeWidth=1.5;exitX=0.5;exitY=1;entryX=0.5;entryY=0;","tr","h2a")
v5 += E("etr2b","fact","endArrow=block;endFill=1;html=1;strokeColor=#FF7043;strokeWidth=1.5;fontSize=9;fontColor=#BF360C;exitX=0.5;exitY=1;entryX=0.5;entryY=0;","h2a","h2b")

# 3-hop branch
v5 += R("h3a","search 1","ellipse;fillColor=#E3F2FD;strokeColor=#90CAF9;fontSize=9;fontColor=#1565C0;html=1;",390,210,80,24)
v5 += R("h3b","search 2","ellipse;fillColor=#E3F2FD;strokeColor=#90CAF9;fontSize=9;fontColor=#1565C0;html=1;",410,268,80,24)
v5 += R("h3c","3-hop Q","rounded=1;fillColor=#EF9A9A;strokeColor=#E53935;strokeWidth=1.5;fontSize=12;fontStyle=1;html=1;",390,316,120,30)
v5 += E("etr3a","","endArrow=block;endFill=1;html=1;strokeColor=#90CAF9;strokeWidth=1.5;exitX=1;exitY=1;entryX=0.5;entryY=0;","tr","h3a",[(410,180),(430,180)])
v5 += E("etr3b","","endArrow=block;endFill=1;html=1;strokeColor=#90CAF9;strokeWidth=1.5;exitX=0.5;exitY=1;entryX=0.5;entryY=0;","h3a","h3b")
v5 += E("etr3c","","endArrow=block;endFill=1;html=1;strokeColor=#E53935;strokeWidth=1.5;exitX=0.5;exitY=1;entryX=0.5;entryY=0;","h3b","h3c")

# 4-hop branch
v5 += R("h4a","s1","ellipse;fillColor=#E3F2FD;strokeColor=#90CAF9;fontSize=8;fontColor=#1565C0;html=1;",488,200,52,20)
v5 += R("h4b","s2","ellipse;fillColor=#E3F2FD;strokeColor=#90CAF9;fontSize=8;fontColor=#1565C0;html=1;",498,232,52,20)
v5 += R("h4c","s3","ellipse;fillColor=#E3F2FD;strokeColor=#90CAF9;fontSize=8;fontColor=#1565C0;html=1;",488,264,52,20)
v5 += R("h4d","4-hop Q","rounded=1;fillColor=#B71C1C;strokeColor=#B71C1C;strokeWidth=2;fontSize=12;fontStyle=1;fontColor=#FFFFFF;html=1;",476,296,120,30)
v5 += E("etr4","","endArrow=block;endFill=1;html=1;strokeColor=#90CAF9;strokeWidth=1.2;exitX=1;exitY=0.5;entryX=0.5;entryY=0;","tr","h4a",[(460,148),(514,148)])
v5 += E("e4ab","","endArrow=block;endFill=1;html=1;strokeColor=#90CAF9;strokeWidth=1.2;exitX=0.5;exitY=1;entryX=0.5;entryY=0;","h4a","h4b")
v5 += E("e4bc","","endArrow=block;endFill=1;html=1;strokeColor=#90CAF9;strokeWidth=1.2;exitX=0.5;exitY=1;entryX=0.5;entryY=0;","h4b","h4c")
v5 += E("e4cd","","endArrow=block;endFill=1;html=1;strokeColor=#B71C1C;strokeWidth=1.5;exitX=0.5;exitY=1;entryX=0.5;entryY=0;","h4c","h4d")

# Ratio info
v5 += R("ratio_bg","","rounded=1;fillColor=#FFF3E0;strokeColor=#FFB74D;strokeWidth=1.5;",40,380,500,80)
v5 += R("ratioh","Hop Ratio  4 : 3 : 2 : 1","text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=13;fontStyle=1;fontColor=#E65100;",40,386,500,22)
v5 += R("ratiol","Most questions are 4-hop (hardest)&#xa;~50,000 questions generated per iteration","text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=11;fontColor=#5D4037;",40,410,500,44)

# Ratio bar chart
for i,(lbl,h,fc) in enumerate([("4-hop",80,"#B71C1C"),("3-hop",60,"#EF9A9A"),("2-hop",40,"#FFCCBC"),("1-hop",20,"#FFE0B2")]):
    bx = 70 + i*110
    v5 += R(f"bar{i}","",f"rounded=0;fillColor={fc};strokeColor=#888;strokeWidth=1;",bx,490-h,80,h)
    v5 += R(f"barl{i}",lbl,"text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=10;fontColor=#555;",bx,494,80,16)

# Challenger label at bottom of tree
v5 += R("crole","(A) Challenger generates all of these","text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=11;fontStyle=1;fontColor=#BF360C;",30,520,540,20)

# RIGHT: Solver + Verifier pipeline
v5 += R("rpnl","","rounded=1;fillColor=#F7F7F7;strokeColor=#BDBDBD;strokeWidth=2;",600,60,1060,800)

# Solver section
v5 += R("Spnl","","rounded=1;fillColor=#F1F8E9;strokeColor=#AED581;strokeWidth=2;dashed=1;dashPattern=8 4;",620,80,480,380)
v5 += R("SbadgeEL","(B)","ellipse;fillColor=#66BB6A;strokeColor=none;fontColor=#FFFFFF;fontSize=13;fontStyle=1;html=1;",620,80,36,36)
v5 += R("SH","Solver","rounded=0;fillColor=#DCEDC8;strokeColor=#AED581;strokeWidth=2;fontSize=15;fontStyle=1;align=center;html=1;",656,80,444,36)
v5 += R("SI","Qwen2.5-3B  GRPO group=5  Multi-turn retrieval","text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=11;fontColor=#555;",630,122,460,18)
v5 += R("SR","Reward = Verifier judgment + EM score","rounded=1;fillColor=#FFE0B2;strokeColor=#FFAB40;fontSize=11;fontStyle=1;html=1;",650,146,430,28)
v5 += R("Sdiv","","line;strokeColor=#DCEDC8;strokeWidth=1.5;fillColor=none;",638,182,450,8)
v5 += R("Ssec","Multi-turn Retrieval-Augmented Reasoning","text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=13;fontStyle=1;fontColor=#33691E;",630,192,460,22)
# Solver vertical chain
els2 = [("Sq","Multi-hop Q","#FFCCBC","#FF7043",816,222,168,30),
        ("Ss1","Search 1","#E3F2FD","#90CAF9",820,266,160,28),
        ("St1","Think","#F3E5F5","#CE93D8",820,306,160,28),
        ("Ss2","Search 2","#E3F2FD","#90CAF9",820,346,160,28),
        ("St2","Think","#F3E5F5","#CE93D8",820,386,160,28),
        ("Sans","answer","#DCEDC8","#7CB342",810,426,180,30)]
for id_,lbl,fc,sc,x,y,w,h in els2:
    sty = f"rounded=1;fillColor={fc};strokeColor={sc};strokeWidth={'2' if lbl in ('Multi-hop Q','answer') else '1.5'};fontSize=12;fontStyle={'1' if lbl in ('Multi-hop Q','answer') else '0'};fontColor={'#33691E' if lbl=='answer' else '#BF360C' if lbl=='Multi-hop Q' else '#1565C0' if 'Search' in lbl else '#6A1B9A'};html=1;"
    v5 += R(id_,lbl,sty,x,y,w,h)
for s,t,eid in [("Sq","Ss1","sa1"),("Ss1","St1","sa2"),("St1","Ss2","sa3"),("Ss2","St2","sa4"),("St2","Sans","sa5")]:
    v5 += E(eid,"","endArrow=block;endFill=1;html=1;strokeColor=#AED581;strokeWidth=1.5;exitX=0.5;exitY=1;entryX=0.5;entryY=0;",s,t)

# Verifier section
v5 += R("Vpnl","","rounded=1;fillColor=#F3E5F5;strokeColor=#CE93D8;strokeWidth=2;dashed=1;dashPattern=8 4;",1120,80,520,380)
v5 += R("VbadgeEL","(C)","ellipse;fillColor=#AB47BC;strokeColor=none;fontColor=#FFFFFF;fontSize=13;fontStyle=1;html=1;",1120,80,36,36)
v5 += R("VH","Verifier","rounded=0;fillColor=#E1BEE7;strokeColor=#CE93D8;strokeWidth=2;fontSize=15;fontStyle=1;align=center;html=1;",1156,80,484,36)
v5 += R("VI","Qwen2.5-3B  GRPO group=5  Multi-turn retrieval","text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=11;fontColor=#555;",1130,122,500,18)
v5 += R("VR","Reward = verification accuracy vs. ground truth","rounded=1;fillColor=#FFE0B2;strokeColor=#FFAB40;fontSize=11;fontStyle=1;html=1;",1148,146,460,28)
v5 += R("Vdiv","","line;strokeColor=#E1BEE7;strokeWidth=1.5;fillColor=none;",1138,182,490,8)
v5 += R("Vsec","Backward Verification via Search","text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=13;fontStyle=1;fontColor=#6A1B9A;",1130,192,500,22)
v5 += R("Vin","(Q, proposed answer)","rounded=1;fillColor=#FFF9C4;strokeColor=#F9A825;strokeWidth=1.5;fontSize=12;fontStyle=1;html=1;",1230,222,220,30)
v5 += R("Vsr","Backward Search","rounded=1;fillColor=#E3F2FD;strokeColor=#90CAF9;strokeWidth=1.5;fontSize=12;fontColor=#1565C0;html=1;",1240,268,200,28)
v5 += R("Vbwd","answer to Q entity (reverse)","text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=9;fontColor=#999;fontStyle=2;",1210,298,270,16)
v5 += R("Vth","Reason and Compare","rounded=1;fillColor=#F3E5F5;strokeColor=#CE93D8;strokeWidth=1.5;fontSize=12;fontColor=#6A1B9A;html=1;",1240,320,200,28)
v5 += R("Vout","yes / no","rounded=1;fillColor=#E1BEE7;strokeColor=#8E24AA;strokeWidth=2;fontSize=14;fontStyle=1;fontColor=#4A148C;html=1;",1258,362,168,34)
for s,t,eid in [("Vin","Vsr","va1"),("Vsr","Vth","va2"),("Vth","Vout","va3")]:
    v5 += E(eid,"","endArrow=block;endFill=1;html=1;strokeColor=#CE93D8;strokeWidth=1.5;exitX=0.5;exitY=1;entryX=0.5;entryY=0;",s,t)

# Right panel: data flow notes
v5 += R("nb1","","rounded=1;fillColor=#FFF3E0;strokeColor=#FFB74D;dashed=1;html=1;",630,480,460,130)
v5 += R("nb1L","Data Construction&#xa;Challenger generates D_k (~50k Qs)&#xa;Solver runs inference on 12,800 sampled Qs&#xa;Both correct and wrong answers used as&#xa;Verifier training data (pos/neg pairs)","text;html=1;strokeColor=none;fillColor=none;align=left;fontSize=11;fontColor=#5D4037;",648,484,430,124)
v5 += R("nb2","","rounded=1;fillColor=#FFF3E0;strokeColor=#FFB74D;dashed=1;html=1;",1130,480,490,130)
v5 += R("nb2L","Reward Signals&#xa;Verifier yes/no becomes Solver reward&#xa;Solver correctness difficulty becomes Challenger reward&#xa;All three models checkpoint to Iter k+1&#xa;(3 iterations total)","text;html=1;strokeColor=none;fillColor=none;align=left;fontSize=11;fontColor=#5D4037;",1148,484,460,124)

# Arrows between panels
v5 += E("mSV","(Q, Answer)","endArrow=block;endFill=1;html=1;strokeColor=#7CB342;strokeWidth=2.5;fontSize=12;fontStyle=1;fontColor=#33691E;exitX=1;exitY=0.4;entryX=0;entryY=0.4;","Spnl","Vpnl")
v5 += E("mTreeS","Multi-hop Q","endArrow=block;endFill=1;html=1;strokeColor=#FF7043;strokeWidth=2.5;fontSize=12;fontStyle=1;fontColor=#BF360C;exitX=1;exitY=0.3;entryX=0;entryY=0.3;","treepnl","Spnl")
v5 += E("rvs","reward","endArrow=block;endFill=1;html=1;strokeColor=#FFAB40;strokeWidth=2;dashed=1;dashPattern=6 4;fontSize=11;fontStyle=1;fontColor=#E65100;exitX=0.5;exitY=1;entryX=0.5;entryY=1;","Vpnl","Spnl",[(1380,638),(900,638)])
v5 += E("rsc","difficulty","endArrow=block;endFill=1;html=1;strokeColor=#FFAB40;strokeWidth=2;dashed=1;dashPattern=6 4;fontSize=11;fontStyle=1;fontColor=#E65100;exitX=0;exitY=1;entryX=1;entryY=1;","Spnl","treepnl",[(620,656),(580,656)])

# Retrieval
v5 += R("ret","","rounded=1;fillColor=#E3F2FD;strokeColor=#90CAF9;strokeWidth=2;",20,882,1640,58)
v5 += R("retL","Shared Retrieval — Wikipedia (wiki-18)  FAISS Index  E5-base-v2  Top-3 passages  (used by Challenger, Solver, Verifier)","text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=13;fontStyle=1;fontColor=#1565C0;",20,882,1640,58)
v5 += E("rT","","endArrow=open;endSize=6;html=1;strokeColor=#90CAF9;strokeWidth=1.5;dashed=1;exitX=0.2;exitY=0;entryX=0.5;entryY=1;","ret","treepnl")
v5 += E("rS","","endArrow=open;endSize=6;html=1;strokeColor=#90CAF9;strokeWidth=1.5;dashed=1;exitX=0.52;exitY=0;entryX=0.5;entryY=1;","ret","Spnl")
v5 += E("rV","","endArrow=open;endSize=6;html=1;strokeColor=#90CAF9;strokeWidth=1.5;dashed=1;exitX=0.82;exitY=0;entryX=0.5;entryY=1;","ret","Vpnl")

with open("d:/paper/drzero/paper/figure_v5_tree.drawio","w",encoding="utf-8") as f:
    f.write(wrap(v5, pw=1700, ph=980))
print("V5 saved")
