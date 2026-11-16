# -*- coding: utf-8 -*-
"""图 3-1：从物理节点到覆盖网，以及坐标的来历（图内不出现任何字母）"""

import os
FIGDIR = os.path.dirname(os.path.abspath(__file__))

W, H = 1800, 880
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


def T(x, y, s, size=22, fill=INK, weight="normal", anchor="start", op=1.0):
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
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{sw}"/>'
    )


def L(x1, y1, x2, y2, stroke=LINE, sw=1.6, dash=None, op=1.0, marker=False):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    mk = ' marker-end="url(#ar)"' if marker else ""
    P.append(
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" '
        f'stroke-width="{sw}" opacity="{op}"{d}{mk}/>'
    )


def arrow(x1, y1, x2, y2, color=RED, sw=3.0, dash=None, mk="ar"):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    P.append(
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
        f'stroke-width="{sw}"{d} marker-end="url(#{mk})"/>'
    )


# ==================================================== 上排：三栏
# ---------- 栏一：物理节点
T(40, 96, "物理节点", 28, INK, "bold")
T(40, 130, "每个节点持有若干类别", 20, SUB)

node_y = [170, 300, 430]
for i, ny in enumerate(node_y):
    R(50, ny, 190, 84, "#FFFFFF", rx=10, stroke=CLD[i], sw=2.4)
    for k in range(3):
        R(70 + k * 38, ny + 30, 24, 24, CLS[i], rx=5, stroke=CLD[i], sw=1.4)
    T(60, ny + 76, "本地的类", 17, SUB)

# ---------- 栏二：可寻址单元
T(400, 96, "每类一个可寻址单元", 28, INK, "bold")
T(400, 130, "把“节点—类别”对拆开", 20, SUB)

unit_x = [500, 610, 720]
for i, ny in enumerate(node_y):
    arrow(248, ny + 42, 466, ny + 42, "#9DB0BF", 2.4, mk="ag")
    for k, ux in enumerate(unit_x):
        C(ux, ny + 42, 26, CLS[i], CLD[i], 2.0)
    # 同一节点的三个单元用虚线框圈起
    R(470, ny + 8, 288, 68, "none", rx=34, stroke=CLD[i], sw=1.6, dash="7 5")
    T(474, ny + 100, "同一节点", 18, CLD[i])

T(400, 618, "同一节点在网中是多个彼此独立的顶点", 20, SUB)

# ---------- 栏三：覆盖网
R(880, 70, 880, 560, BF, rx=12)
T(910, 112, "覆盖网", 28, INK, "bold")
T(910, 146, "每个顶点即一个可寻址单元", 20, SUB)
T(910, 176, "顶点按坐标远近相连，连线即边", 20, SUB)

# 顶点按类聚成簇：同一类的若干副本坐标相近，故挨在一起。各簇画出的副本数不等，
# 画不下的用省略号表示；若一例一律三个，读者会以为每类恰好三份。
import os
import math


def ring(n, rad, phase=-math.pi / 2, jit=(1.0,)):
    pts = []
    for k in range(n):
        a = phase + 2 * math.pi * k / n
        f = jit[k % len(jit)]
        pts.append((round(rad * f * math.cos(a)), round(rad * f * math.sin(a))))
    return pts


clusters = [
    dict(c=(1040, 300), pts=ring(4, 36, jit=(1.0, 0.94, 1.06, 0.97)), tail=False),
    dict(c=(1230, 264), pts=[(-31, -18), (0, -36), (31, -18)], tail=True),
    dict(c=(1420, 300), pts=ring(6, 42, jit=(1.0, 0.93, 1.05, 0.97, 1.03, 0.95)), tail=False),
    dict(c=(1600, 258), pts=ring(3, 34, jit=(1.0, 0.95, 1.05)), tail=False),
    dict(c=(1120, 472), pts=ring(5, 38, phase=0.3 - math.pi / 2, jit=(1.0, 0.94, 1.05, 0.97, 1.02)), tail=False),
    dict(c=(1330, 486), pts=[(-31, -18), (0, -36), (31, -18)], tail=True),
    dict(c=(1560, 466), pts=ring(5, 38, jit=(1.0, 1.04, 0.95, 1.02, 0.96)), tail=False),
]
vpos, groups = [], []
for gi, cl in enumerate(clusters):
    cx, cy = cl["c"]
    idx = []
    for k, (dx, dy) in enumerate(cl["pts"]):
        idx.append(len(vpos))
        vpos.append((cx + dx, cy + dy, (k + gi) % 3))
    groups.append(idx)

edges = []
for cl, g in zip(clusters, groups):   # 簇内：同类副本依次相连
    for i in range(len(g) - 1):
        edges.append((g[i], g[i + 1]))
    if not cl["tail"]:
        edges.append((g[-1], g[0]))


def nearest(ga, gb):
    best, pair = 1e18, (ga[0], gb[0])
    for i in ga:
        for j in gb:
            dd = (vpos[i][0] - vpos[j][0]) ** 2 + (vpos[i][1] - vpos[j][1]) ** 2
            if dd < best:
                best, pair = dd, (i, j)
    return pair


cross = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (1, 5), (3, 6)]
cross_pairs = [nearest(groups[a], groups[b]) for a, b in cross]
edges += cross_pairs

for a, b in edges:
    L(vpos[a][0], vpos[a][1], vpos[b][0], vpos[b][1], LINE, 1.4, op=0.85)

# 用虚线椭圆把同一类的副本圈起来，大小按簇内实际排布算
for cl in clusters:
    cx, cy = cl["c"]
    rx = max(abs(dx) for dx, dy in cl["pts"]) + 26
    ry = max(abs(dy) for dx, dy in cl["pts"]) + 26
    P.append(
        f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="none" '
        f'stroke="{SUB}" stroke-width="1.6" stroke-dasharray="7 5" opacity="0.9"/>'
    )

for (vx, vy, ci) in vpos:
    C(vx, vy, 18, CLS[ci], CLD[ci], 2.0)

# 画不下的副本用省略号表示：三个小点，与顶点尺寸拉开，不会被当成顶点
for cl in clusters:
    if not cl["tail"]:
        continue
    cx, cy = cl["c"]
    for k in (-1, 0, 1):
        C(cx + k * 11, cy + 20, 3.4, SUB)

# 引线标注：顶点、边
L(vpos[0][0], vpos[0][1], 962, 274, SUB, 1.4, dash="5 4")
T(906, 280, "顶点", 20, INK, "bold")
mx, my = cross_pairs[7]
L((vpos[mx][0] + vpos[my][0]) / 2, (vpos[mx][1] + vpos[my][1]) / 2, 1634, 380, SUB, 1.4, dash="5 4")
T(1642, 386, "边", 20, INK, "bold")

# 底部说明
T(900, 580, "虚线椭圆圈住的是同一类的若干副本", 19, SUB)
T(900, 608, "省略号表示该类还有更多副本", 19, SUB)

# ==================================================== 下排：坐标的来历
T(40, 690, "顶点的坐标从何而来", 28, INK, "bold")

by = 740   # 基线
# 输入
R(50, 700, 96, 80, AF, rx=10, stroke=LINE, sw=1.6)
for k in range(3):
    L(66, 716 + k * 20, 130, 716 + k * 20, SUB, 1.4, op=0.55)
T(98, 806, "输入", 21, SUB, anchor="middle")

arrow(152, by, 208, by, "#9DB0BF", 2.6, mk="ag")

# 编码器
R(212, 698, 250, 84, "#FFFFFF", rx=10, stroke=BLUE, sw=2.2)
T(337, 736, "编码器", 22, BLUE, "bold", "middle")
T(337, 766, "不参与本地训练", 18, SUB, anchor="middle")
T(337, 806, "一次发布", 21, SUB, anchor="middle")

arrow(468, by, 522, by, "#9DB0BF", 2.6, mk="ag")

# 高维特征
for k in range(14):
    hh = 30 + ((k * 37) % 34)
    L(540 + k * 15, by + 40, 540 + k * 15, by + 40 - hh, CLD[0], 5.0, op=0.75)
T(645, 806, "高维特征", 21, SUB, anchor="middle")

arrow(760, by, 814, by, "#9DB0BF", 2.6, mk="ag")

# 判别投影
R(818, 698, 230, 84, "#FFFFFF", rx=10, stroke=RED, sw=2.2)
T(933, 736, "坐标", 22, RED, "bold", "middle")
T(933, 766, "即这 384 维特征", 18, SUB, anchor="middle")
T(933, 806, "无须再变换", 21, SUB, anchor="middle")

arrow(1054, by, 1108, by, "#9DB0BF", 2.6, mk="ag")

# 低维坐标
C(1140, by, 24, CLS[1], CLD[1], 2.2)
T(1140, 806, "一个坐标", 21, SUB, anchor="middle")

# 指向覆盖网
arrow(1160, 718, 1371, 529, RED, 2.4, dash="8 6")
T(1372, 672, "坐标决定它落在网中的位置", 20, RED)

# 协议资产说明
R(1380, 690, 380, 130, AF, rx=12, stroke=LINE, sw=1.6)
T(1402, 726, "以上一次发布", 22, INK)
T(1402, 760, "各节点下载后长期复用", 22, INK)
T(1402, 796, "即协议资产，无须逐轮同步", 22, SUB)

svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
<defs>
  <marker id="ar" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="{RED}"/>
  </marker>
  <marker id="ag" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="#8FA3B4"/>
  </marker>
</defs>
<rect width="{W}" height="{H}" fill="#FFFFFF"/>
<g font-family="SimSun, 'Times New Roman', serif">
{chr(10).join(P)}
</g>
</svg>'''

out = os.path.join(FIGDIR, "fig3_1_vnode_map.svg")
with open(out, "w", encoding="utf-8") as f:
    f.write(svg)
print("SVG written:", out)
