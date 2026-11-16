# -*- coding: utf-8 -*-
"""生成 ProtoAtlas 引言图（Fig.1 候选）：从参数匹配到坐标寻址"""

import os
FIGDIR = os.path.dirname(os.path.abspath(__file__))

W, H = 1800, 850
INK  = "#1F2D3A"
SUB  = "#5B6B7B"
LINE = "#9DB0BF"
AF   = "#EDF2F7"
BF   = "#FBF4EE"
BLUE = "#3D6E8F"
RED  = "#B04A32"
CLS  = ["#CFE0EE", "#F3DECF", "#DCEADA"]
CLD  = ["#6C93AE", "#C08A63", "#6E9670"]

P = []


def T(x, y, s, size=24, fill=INK, weight="normal", anchor="start", op=1.0):
    P.append(
        f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" '
        f'font-weight="{weight}" text-anchor="{anchor}" opacity="{op}">{s}</text>'
    )


def R(x, y, w, h, fill, rx=6, stroke="none", sw=0, op=1.0, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    P.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{sw}" opacity="{op}"{d}/>'
    )


def C(cx, cy, r, fill, stroke="none", sw=0):
    P.append(
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
    )


def L(x1, y1, x2, y2, stroke=LINE, sw=1.5, dash=None, op=1.0):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    P.append(
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" '
        f'stroke-width="{sw}" opacity="{op}"{d}/>'
    )


# ============ 左：现有 FL / DFL ============
R(40, 50, 650, 640, AF, rx=10)
T(70, 102, "现有 FL / DFL：平均参数，收敛到同一模型", 24, INK, "bold")

# 中心聚合器
R(300, 138, 130, 58, BLUE, rx=9)
T(365, 174, "中心聚合", 21, "#FFFFFF", "bold", "middle")

nodes_A = [(200, 300), (365, 300), (530, 300), (200, 430), (365, 430), (530, 430)]
for (nx, ny) in nodes_A:
    L(365, 196, nx, ny - 30, BLUE, 1.6, op=0.40)
for i, (nx, ny) in enumerate(nodes_A):
    C(nx, ny, 36, "#FFFFFF", BLUE, 2.4)
    for j, (ox, oy) in enumerate([(-14, -8), (2, -13), (11, 9)]):
        C(nx + ox, ny + oy, 7.5, CLS[(i + j) % 3], CLD[(i + j) % 3], 1.0)

P.append(
    '<path d="M 365 196 L 365 208" stroke="#3D6E8F" stroke-width="2"/>'
)

T(70, 528, "· 依赖被全体认可的中心节点", 23, SUB)
T(70, 566, "· 要求统一标签空间与模型结构", 23, SUB)
T(70, 604, "· 每轮交换 84 MB，收敛需数百轮", 23, SUB)
T(70, 660, "目标：全网收敛到同一个模型", 23, BLUE, "bold")

# ============ 中：问题转移 ============
L(690, 50, 690, 690, LINE, 1.6, dash="9 7")
T(845, 165, "问题转移", 22, SUB, anchor="middle")
T(845, 240, "训练侧", 35, INK, "bold", "middle")
P.append(
    '<line x1="845" y1="265" x2="845" y2="345" stroke="#B04A32" stroke-width="5" '
    'marker-end="url(#ar)"/>'
)
T(845, 405, "检索侧", 35, RED, "bold", "middle")
T(845, 482, "不再复制模型，", 21, SUB, anchor="middle")
T(845, 514, "而是让能力可被寻址", 21, SUB, anchor="middle")

# ============ 右：ProtoAtlas ============
R(1000, 50, 760, 640, BF, rx=10)
T(1030, 102, "ProtoAtlas：把每个类别变成一个地址", 24, INK, "bold")

bodies = [(1085, 215), (1245, 190), (1405, 225), (1085, 360), (1245, 340), (1405, 375)]
edges = [(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5), (0, 4), (1, 5)]
for (a, b) in edges:
    x1, y1 = bodies[a]
    x2, y2 = bodies[b]
    L(x1, y1, x2, y2, LINE, 1.4, dash="6 5", op=0.85)

vpos = []
for i, (bx, by) in enumerate(bodies):
    vpos.append(((bx - 12, by - 42), (bx + 34, by + 8)))

for i, (bx, by) in enumerate(bodies):
    for k, (vx, vy) in enumerate(vpos[i]):
        C(vx, vy, 12, CLS[(i + k) % 3], CLD[(i + k) % 3], 1.6)
    R(bx - 21, by - 21, 42, 42, "#FFFFFF", rx=7, stroke=BLUE, sw=2.6)
    R(bx - 8, by - 8, 16, 16, "none", rx=3, stroke=BLUE, sw=1.5)

# 路径：起点 vnode -> 中间 vnode -> 目标 vnode
start = vpos[3][1]
mid = vpos[4][0]
goal = vpos[2][0]
P.append(
    f'<polyline points="{start[0]},{start[1]} {mid[0]},{mid[1]} {goal[0]},{goal[1]}" '
    f'fill="none" stroke="#B04A32" stroke-width="3.6" stroke-dasharray="10 6" '
    f'marker-end="url(#ar)"/>'
)
for (gx, gy) in [start, mid, goal]:
    C(gx, gy, 16, "none", RED, 2.6)
T(start[0] - 2, start[1] + 34, "起点", 18, RED, anchor="middle")
T(goal[0], goal[1] - 30, "命中：类 c", 18, RED, "bold", "middle")

# 图例
lx = 1520
items = [
    ("c", 0, "类 A 原型"),
    ("c", 1, "类 B 原型"),
    ("c", 2, "类 C 原型"),
]
yy = 205
for kind, idx, lab in items:
    C(lx + 8, yy - 6, 9, CLS[idx], CLD[idx], 1.4)
    T(lx + 28, yy, lab, 17, SUB)
    yy += 38
R(lx - 1, yy - 16, 18, 18, "#FFFFFF", rx=3, stroke=BLUE, sw=2.0)
T(lx + 28, yy, "本地节点", 17, SUB)
yy += 38
P.append(
    f'<line x1="{lx}" y1="{yy - 6}" x2="{lx + 18}" y2="{yy - 6}" stroke="#B04A32" '
    f'stroke-width="3.2" stroke-dasharray="6 4"/>'
)
T(lx + 28, yy, "路径 h≈2–4 跳", 17, SUB)

# 三环节卡片
cards = [
    (1030, 340, "①", "路由", "沿邻居表向更近处转发"),
    (1400, 340, "②", "受理", "终点宿主一律受理，交回若干候选"),
]
for (cx, cw, num, tt, dd) in cards:
    R(cx, 500, cw, 88, "#FFFFFF", rx=8, stroke="#D8DFE5", sw=1.6)
    T(cx + 18, 534, num, 20, RED, "bold")
    T(cx + 46, 534, tt, 20, INK, "bold")
    T(cx + 18, 568, dd, 17, SUB)

T(1030, 632, "协议资产 84 MB　｜　一次发布，此后零同步", 21, SUB)

# ============ 底部：四条硬约束 ============
T(40, 714, "全部实验中同时成立的四条硬约束", 21, INK, "bold")
caps = [
    (40, "A1　无服务器"),
    (480, "A2　无全局标签空间"),
    (920, "A3　无统一 ID 空间"),
    (1360, "A4　无预置邻居"),
]
for (cx, lab) in caps:
    R(cx, 728, 400, 54, "#FFFFFF", rx=27, stroke="#C9D4DD", sw=1.8)
    T(cx + 200, 763, lab, 23, INK, anchor="middle")

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

out = os.path.join(FIGDIR, "fig1_1_loc.svg")
with open(out, "w", encoding="utf-8") as f:
    f.write(svg)
print("SVG written:", out)
