# -*- coding: utf-8 -*-
"""ProtoAtlas 架构图（图 4-1）

三段路径：L3 自评（请求方）→ L1 坐标 → L2 路由 → L3 作答（终点）
画布 1800x790，按 17cm 栏宽印刷时最小字号 24px ≈ 6.5pt
"""

import os
FIGDIR = os.path.dirname(os.path.abspath(__file__))

W, H = 1800, 790
INK = "#1F2D3A"
SUB = "#4E5F6E"
LINE = "#9DB0BF"
AF = "#EDF2F7"
BLUE = "#35617F"
RED = "#B04A32"
GRN = "#4A7A5C"
AMB = "#C08A2E"
GRY = "#8A9AA6"
CLS = ["#CFE0EE", "#F3DECF", "#DCEADA"]
CLD = ["#5C87A6", "#BE8560", "#628A67"]

P = []


def T(x, y, s, size=24, fill=INK, weight="normal", anchor="start", op=1.0):
    P.append(
        f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" '
        f'font-weight="{weight}" text-anchor="{anchor}" opacity="{op}">{s}</text>'
    )


def R(x, y, w, h, fill, rx=8, stroke="none", sw=0, op=1.0, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    P.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{sw}" opacity="{op}"{d}/>'
    )


def C(cx, cy, r, fill, stroke="none", sw=0):
    P.append(
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
    )


def L(x1, y1, x2, y2, stroke=LINE, sw=1.8, dash=None, op=1.0):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    P.append(
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" '
        f'stroke-width="{sw}" opacity="{op}"{d}/>'
    )


def A(x1, y1, x2, y2, color=RED, sw=3.0):
    P.append(
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
        f'stroke-width="{sw}" marker-end="url(#ar)"/>'
    )


# ================= 协议资产 =================
R(40, 28, 1720, 108, AF, rx=10)
T(60, 72, "协议资产：φ 全网唯一，SHA256 校验", 28, INK, "bold")
blocks = [
    (60, 620, "φ　ViT-S/14，84 MB，给出 384 维特征"),
    (820, 880, "一次分发，此后不再改动；自评一侧不用它"),
]
for (bx, bw, lab) in blocks:
    R(bx, 86, bw, 40, "#FFFFFF", rx=6, stroke="#C9D4DD", sw=1.6)
    T(bx + 20, 113, lab, 24, SUB)

# ================= ① 请求方：坐标与自评 =================
T(40, 198, "①　请求方：自评（L3）", 28, INK, "bold")
R(40, 214, 500, 372, "#FFFFFF", rx=10, stroke=BLUE, sw=2.4)

R(60, 234, 110, 38, CLS[0], rx=8, stroke=CLD[0], sw=1.6)
T(115, 260, "输入 x", 24, INK, anchor="middle")
A(176, 253, 204, 253, BLUE, 2.6)
R(210, 234, 168, 38, "#F4F8FB", rx=8, stroke=CLD[0], sw=1.6)
T(294, 260, "本地模型", 24, INK, anchor="middle")
T(390, 260, "给出各类分值", 22, GRY)

R(60, 294, 460, 56, "#F7FAFC", rx=8, stroke="#D8DFE5", sw=1.6)
T(76, 329, "能量 E = -log Σ exp(f_c)，越小越熟", 23, INK)

R(60, 372, 460, 56, "#FFFFFF", rx=8, stroke=AMB, sw=2.2)
T(290, 408, "E ≤ E₀ ？", 26, INK, "bold", "middle")

A(290, 350, 290, 372, SUB, 2.2)

# 是：就地作答
A(180, 428, 180, 462, GRN, 2.8)
T(192, 452, "是", 22, GRN, "bold")
R(60, 462, 250, 92, "#EDF4EF", rx=8, stroke=GRN, sw=2.0)
T(76, 496, "本地模型作答", 24, INK, "bold")
T(76, 528, "查询结束，不发报文", 21, SUB)

# 否：发出请求
T(547, 390, "否", 21, RED, "bold", "middle")
A(520, 400, 574, 400, RED, 3.2)

T(40, 622, "E₀ 按本地样本的分位点定，各节点各自取值", 24, SUB)
T(40, 652, "判不出才由 L1 算出 z = φ(x)，坐标只服务路由", 24, SUB)

# ================= ② 覆盖网 =================
T(600, 198, "②　覆盖网贪婪下降（L2）", 28, INK, "bold")
bodies = [(690, 300), (815, 268), (940, 308), (690, 452), (815, 428), (940, 464)]
edges = [(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)]
for (a, b) in edges:
    x1, y1 = bodies[a]
    x2, y2 = bodies[b]
    L(x1, y1, x2, y2, LINE, 1.6, dash="7 6", op=0.9)

vpos = []
for (bx, by) in bodies:
    vpos.append(((bx - 12, by - 46), (bx + 40, by + 8)))

for i, (bx, by) in enumerate(bodies):
    for k, (vx, vy) in enumerate(vpos[i]):
        C(vx, vy, 13, CLS[(i + k) % 3], CLD[(i + k) % 3], 1.6)
    R(bx - 22, by - 22, 44, 44, "#FFFFFF", rx=8, stroke=BLUE, sw=2.4)

start = vpos[3][1]
mid = vpos[4][0]
goal = vpos[2][0]
P.append(
    f'<polyline points="{start[0]},{start[1]} {mid[0]},{mid[1]} {goal[0]},{goal[1]}" '
    f'fill="none" stroke="#B04A32" stroke-width="4" stroke-dasharray="11 7" '
    f'marker-end="url(#ar)"/>'
)
for (gx, gy) in [start, mid, goal]:
    C(gx, gy, 18, "none", RED, 2.6)
T(start[0] - 4, start[1] + 42, "起点", 21, RED, anchor="middle")
T(goal[0] + 26, goal[1] - 32, "命中", 21, RED, "bold")

T(600, 582, "邻居表存坐标，比较在本地完成", 24, SUB)
T(600, 620, "每跳只转发给 1 个邻居（约 1.6 KB）", 24, SUB)
T(600, 658, "平均 h = 2–4 跳，全程不含样本", 24, SUB)

# 交付箭头
A(985, 428, 1064, 428, RED, 3.4)
T(1024, 410, "交付样本", 22, RED, "bold", "middle")

# ================= ③ 宿主作答 =================
T(1090, 198, "③　抵达后由宿主作答（L3）", 28, INK, "bold")
R(1090, 262, 650, 152, "#FFFFFF", rx=9, stroke="#D8DFE5", sw=1.8)
R(1090, 262, 8, 152, GRN, rx=4)
T(1122, 308, "一律作答", 28, INK, "bold")
T(1122, 352, "以本地判别头给出本地各类的分值", 24, SUB)
T(1122, 390, "取最高者，交回请求方", 24, SUB)

T(1090, 462, "样本由请求方直接发往终点宿主，只送一次", 24, SUB)
T(1090, 500, "跳数只作用于坐标，不放大样例的传输量", 24, SUB)

# ================= 底部硬约束 =================
T(40, 700, "四条硬约束，全部实验中同时成立", 26, INK, "bold")
caps = [(40, "A1　无服务器"), (480, "A2　无全局标签空间"),
        (920, "A3　无统一 ID 空间"), (1360, "A4　无预置邻居")]
for (cx, lab) in caps:
    R(cx, 712, 400, 46, "#FFFFFF", rx=23, stroke="#C9D4DD", sw=2.0)
    T(cx + 200, 743, lab, 24, INK, anchor="middle")

svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
<defs>
  <marker id="ar" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="{RED}"/>
  </marker>
</defs>
<rect width="{W}" height="{H}" fill="#FFFFFF"/>
<g font-family="SimSun, 'Times New Roman', serif">
{chr(10).join(P)}
</g>
</svg>'''

out = os.path.join(FIGDIR, "fig4_1_arch.svg")
with open(out, "w", encoding="utf-8") as f:
    f.write(svg)
print("SVG:", out)
