import json, collections, os
D=json.load(open("full.json")); S=set(D["seeds"]); I=D["info"]; E=[tuple(e) for e in D["edges"]]
orgs=json.load(open("orgs.json"))
I["Q102291966"]["en"]="Carlos Guestrin"   # 위키데이터 지도 관계 쪽 이름이 중간 이름까지 붙어 있다
byname={v["en"]:k for k,v in I.items()}
manual=[("Geoffrey Hinton","Yann LeCun","포닥"),("Yoshua Bengio","Kyunghyun Cho","포닥"),("Jure Leskovec","William L. Hamilton","박사"),("Jure Leskovec","Rex Ying","박사")]
ME=[(byname[a],byname[b],t) for a,b,t in manual]
segs=collections.defaultdict(set)
for a,b in E:
    if a in S: segs[b].add(a)
    if b in S: segs[a].add(b)
core=set(S)|{q for q in I if len(segs[q])>=2 or I[q]["sl"]>=15 or I[q]["awards"]}
KO=json.load(open("ko.json"))
nodes=[{"id":q,"ko":KO.get(v["en"],v["ko"]),"en":v["en"],"d":v["desc"],"b":int(v["born"]) if v["born"][:4].isdigit() else None,"aw":v["awards"],"seed":q in S,"core":q in core,"sl":v["sl"]} for q,v in I.items()]
links=[{"s":a,"t":b,"k":"wd"} for a,b in E]+[{"s":a,"t":b,"k":"manual","n":t} for a,b,t in ME]
seedorg=collections.defaultdict(set); olab={}
for p,o,ol,olk,rel in orgs:
    if p in S: seedorg[o].add(p); olab[o]=(ol,olk)
keep={o for o,ps in seedorg.items() if len(ps)>=2 and "High School" not in olab[o][0]}|{o for p,o,ol,olk,rel in orgs if rel=="founder"}
OI=json.load(open("org_info.json"))   # 세운 해 · 본사 · 홈페이지 · 설명 (위키데이터 + 직접 정리)
onodes=[{"id":o,"ko":olab[o][1],"en":olab[o][0],**{k:v for k,v in OI.get(o,{}).items() if v}} for o in sorted(keep)]
olinks=sorted({(p,o,rel) for p,o,ol,olk,rel in orgs if o in keep and p in S and rel in("employer","founder")})
edu,emp={},{}
for p,o,ol,olk,rel in orgs:
    if rel=="educated": edu.setdefault(p,set()).add(olk or ol)
    else: emp.setdefault(p,set()).add((olk or ol)+(" (창업)" if rel=="founder" else ""))
out={"nodes":nodes,"links":links,"orgs":onodes,"olinks":[{"s":p,"t":o,"r":r} for p,o,r in olinks],"edu":{k:sorted(v) for k,v in edu.items()},"emp":{k:sorted(v) for k,v in emp.items()}}
json.dump(out,open("data.json","w"),ensure_ascii=False,separators=(",",":"))
s=open("template.html").read().replace("주요 인물 82명","주요 인물 83명")
open("ai_lineage.html","w").write(s.replace("__DATA__",json.dumps(out,ensure_ascii=False,separators=(",",":"))))
print(len(nodes),sum(n["core"] for n in nodes),len(links))

# ---- 직접 추가한 사람 (위키데이터 지도 관계 데이터 밖) ----
EXTRA={"Q107525448":{"en":"Dmitri Dolgov","ko":"","desc":"Russian American businessman","sl":2,"born":"","awards":[]}}
for q_,v in json.load(open("extra_people.json")).items():
    EXTRA[q_]={"en":v["en"],"ko":v["ko"],"desc":"","sl":v.get("sl",5),"born":str(v["born"] or ""),"awards":[]}
for q_,v in EXTRA.items():
    if q_ not in byname.values():
        out["nodes"].append({"id":q_,"ko":KO.get(v["en"],""),"en":v["en"],"d":v["desc"],"b":None,"aw":[],"seed":True,"core":True,"sl":v["sl"]})
        byname[v["en"]]=q_; I[q_]=v
# 위키데이터 지도 관계에 빠진 것 (위키백과 · 위키데이터 P184 기준)
for a_,b_,t_,k_ in [("Tang Xiao'ou","Kaiming He","박사","manual"),("Raj Reddy","Kai-Fu Lee","박사","wd"),("B. F. Skinner","William Kaye Estes","박사","wd"),
               ("Carlos Guestrin","Tianqi Chen","박사","wd"),("Leslie Valiant","Michael Kearns","박사","wd"),("Thomas M. Cover","Peter E. Hart","박사","wd"),("Berthold K. P. Horn","Marc Raibert","박사","wd"),("Albert W. Tucker","David Gale","박사","wd"),("Albert W. Tucker","Lloyd Shapley","박사","wd"),("Vannevar Bush","Frederick Terman","박사","wd"),("Yoshua Bengio","Edward J. Hu","박사","manual"),("Marcus Hutter","Jan Leike","박사","wd")]:
    if a_ in byname and b_ in byname and not any(l["s"]==byname[a_] and l["t"]==byname[b_] for l in out["links"]):
        out["links"].append({"s":byname[a_],"t":byname[b_],"k":k_,**({"n":t_} if k_=="manual" else {})})
# ---- v3: 유명 인물 · 사진 · 업적 · 공동 연구 · 논문 그래프 ----
C=json.load(open("curated.json")); PH=json.load(open("photos.json")); SC=json.load(open("scholarly.json"))
byq={n["id"]:n for n in out["nodes"]}
fam={byname[k] for k in C}
for n in out["nodes"]:
    c=C.get(n["en"])
    n["fam"]=bool(c)
    if c:
        n.update({k:c[k] for k in ("uni","phd","ach","books","field","show") if k in c}); n["ent"]=bool(c.get("ent"))
    if n["id"] in PH: n["img"]=PH[n["id"]]["img"]; n["imgPage"]=PH[n["id"]]["page"]
up=collections.defaultdict(set); dn=collections.defaultdict(set)
for l in out["links"]: up[l["t"]].add(l["s"]); dn[l["s"]].add(l["t"])
def reach(start,g):
    s=set(); st=list(start)
    while st:
        x=st.pop()
        for y in g[x]:
            if y not in s: s.add(y); st.append(y)
    return s
chain=(reach(fam,up)&reach(fam,dn))-fam
# 유명 인물 둘 이상과 바로 이어진 사람 (공동 지도 등)도 중간 다리로 남긴다
# 섀넌 · 민스키를 잇는 대니 힐리스만 예외로 남긴다 (공동 지도 제자를 다 넣으면 점이 너무 많아진다)
chain.add(byname["William Daniel Hillis"])
# 민스키 · 게일 · 섀플리의 지도교수 터커도 (위로는 유명 인물이 없어 위 규칙에 안 걸린다)
chain.add(byname["Albert W. Tucker"])
for n in out["nodes"]: n["chain"]=n["id"] in chain
B=json.load(open("births.json")); MT=json.load(open("metrics.json"))
# 출생 연도: 영어 위키백과 첫 문단 > 위키데이터(연도 정밀도, 값 하나일 때만) > 직접 확인한 값. 나머지는 표시 안 함
WD=json.load(open("births_wd.json")); DIED=json.load(open("died.json")); WP=json.load(open("births_check.json"))
for n in out["nodes"]:
    v=WD.get(n["id"]); n["b"]=v if isinstance(v,int) else None
    w=WP.get(n["en"],{}).get("wp")
    if w: n["b"]=int(w)
    if n["en"] in B["fix"]: n["b"]=B["fix"][n["en"]]
    if n["en"] in B["est"]: n["yEst"]=B["est"][n["en"]]
    if n["id"] in DIED: n["died"]=DIED[n["id"]]
    n["star"]=n["sl"]>=50
    m=MT.get(n["id"])
    if m and n["fam"]: n["m"]={"c":m["cited_by_count"],"h":m["summary_stats"]["h_index"],"oa":m["id"].split("/")[-1]}
G=json.load(open("collab.json"))
import re
CM=json.load(open("collab_meta.json"))
CE=json.load(open("curated_en2.json"))
for n in out["nodes"]:
    e=CE.get(n["en"])
    if e and n["fam"]:
        for k in ("uni","phd","ach","books"):
            if e.get(k) is not None or k in ("phd",): n[k+"_en"]=e.get(k)
# 소속 · 창업 (위키데이터) 영어판
emp_en={}
for pp,o,ol,olk,rel in orgs:
    if rel in ("employer","founder"): emp_en.setdefault(pp,set()).add((ol or olk)+(" (founder)" if rel=="founder" else ""))
out["emp_en"]={k:sorted(v) for k,v in emp_en.items()}
HP=json.load(open("homepages_ok.json"))
for n in out["nodes"]:
    if n["fam"] and n["en"] in HP: n["hp"]=HP[n["en"]]
WE=json.load(open("work_edu.json"))
for n in out["nodes"]:
    if n["fam"] and n["en"] in WE: n["work"]=WE[n["en"]]["work"]; n["edu"]=WE[n["en"]]["edu"]
PC=json.load(open("paper_cites.json")) if os.path.exists("paper_cites.json") else {}
PF=json.load(open("paper_figs.json")); PF.update({k:{**v,"logo":True} for k,v in json.load(open("logos.json")).items()})
# 제목 대조에서 엉뚱한 기록이 잡힌 인용 수는 쓰지 않는다
for bad in ["CTC (2006)","『Deep Learning』 Nature 리뷰 (2015)","GAN (2014)","교과서 AIMA (1995)","Mask R-CNN (2017)","구글 브레인 공동 설립 (2011)"]: PC.pop(bad,None)
PC={k:v for k,v in PC.items() if "창업" not in k and "설립" not in k}
if os.path.exists("paper_cites_fix.json"): PC.update(json.load(open("paper_cites_fix.json")))   # arXiv · DOI 로 다시 받은 값
LBL=json.load(open("labels_en.json"))
out["collab"]=[{"p":LBL.get(p,p),"pk":p,"ids":[byname[x] for x in ppl],"biz":("창업" in p or "설립" in p),"c":PC.get(p,{}).get("c"),"csrc":PC.get(p,{}).get("src"),"fig":PF.get(p),"y":int(re.search(r"(1[89]\d\d|20\d\d)",p).group(1)),
                "book":("교과서" in p or "『Perceptrons』" in p),"proj":("프로젝트" in p),**CM.get(p,{})} for p,ppl in G]
out["co"]=[{"a":k.split("|")[0],"b":k.split("|")[1],"n":v} for k,v in SC["co"].items()]
out["cit"]=[{"a":k.split(">")[0],"b":k.split(">")[1],"n":v} for k,v in SC["cit"].items()]
t=open("template.html").read()
open("ai_lineage.html","w").write(t.replace("__DATA__",json.dumps(out,ensure_ascii=False,separators=(",",":"))))
# ---- 공개판 (smhanlab.com/book-ml/ai-lineage/) ----
if os.environ.get("PUBLIC_OUT"):
    import copy
    pub=copy.deepcopy(out)
    for n in pub["nodes"]:
        if n.get("img") and PH.get(n["id"],{}).get("press"): n.pop("img",None); n.pop("imgPage",None)   # 언론 · 공정 이용 사진만 뺀다
    for g in pub["collab"]:
        pass   # 소개 페이지 대표 이미지는 공식 페이지 것이라 그대로 둔다
    body=t.replace("__DATA__",json.dumps(pub,ensure_ascii=False,separators=(",",":")))
    title=body[body.find("<title>")+7:body.find("</title>")]
    body=body.replace("<title>"+title+"</title>","",1)
    head=("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
          "<title>AI Researcher Family Tree</title>\n"
          "<meta name=\"description\" content=\"From Turing to Hinton, Bengio and LeCun to today's AI founders: advisor-student lineages, key papers, citations, h-index and companies. By Sumin Han (KSA).\">\n"
          "<script async src=\"https://www.googletagmanager.com/gtag/js?id=G-DT1RTEZ926\"></script>\n"
          "<script>window.dataLayer = window.dataLayer || []; function gtag(){dataLayer.push(arguments);} gtag(\"js\", new Date()); gtag(\"config\", \"G-DT1RTEZ926\");</script>\n"
          "<script data-goatcounter=\"https://smhanlab.goatcounter.com/count\" async src=\"//gc.zgo.at/count.js\"></script>\n"
          "<!-- notebooks/ai_lineage/prep.py 가 만든다 (PUBLIC_OUT=...). 직접 고치지 말 것 -->\n"
          "<style>*,*::before,*::after{box-sizing:border-box} body{margin:0}</style>\n</head>\n<body>\n")
    os.makedirs(os.path.dirname(os.environ["PUBLIC_OUT"]),exist_ok=True)
    open(os.environ["PUBLIC_OUT"],"w").write(head+body+"\n</body>\n</html>\n")
    print("public:",os.environ["PUBLIC_OUT"],os.path.getsize(os.environ["PUBLIC_OUT"])//1024,"KB",
          "photos",sum(1 for n in pub["nodes"] if n.get("img") and n.get("fam")),"figs",sum(1 for g in pub["collab"] if g.get("fig")))
print("fam",len(fam),"chain",len(chain),"KB",os.path.getsize("ai_lineage.html")//1024)
