# -*- coding: utf-8 -*-
"""图 3-1（中文版）：从物理节点到覆盖网，以及坐标的来历

与 build_fig3_1_en.py 同源，仅文字换成中文；几何、形状、颜色完全一致
"""


import os
import math
FIGDIR = os.path.dirname(os.path.abspath(__file__))

W, H = 1800, 880
INK  = "#1F2D3A"
SUB  = "#5B6B7B"
LINE = "#9DB0BF"
AF   = "#EDF2F7"
MOD  = "#9DB0BF"      # 模块虚线框的描边，浅灰
BLUE = "#3D6E8F"
RED  = "#B04A32"
CLS  = ["#CFE0EE", "#F3DECF", "#DCEADA"]      # 三个物理节点：蓝 / 橙 / 绿
CLD  = ["#6C93AE", "#C08A63", "#6E9670"]
HOL  = "#9AA9B5"                              # 空心（没画出来的节点）的描边

BX = 220      # 作者手改：底栏整体右移 220，在页面上居中

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


def POLY(pts, fill, stroke, sw=2.0):
    s = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    P.append(f'<polygon points="{s}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')


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


# ==================================================== 形状
# 形状编码「类」，颜色编码物理节点
SH_C, SH_Q, SH_T, SH_D, SH_H, SH_P, SH_O = range(7)


def mark(cx, cy, r, shape, fill, stroke, sw=2.0):
    """在 (cx,cy) 画一个半径为 r 的 vnode，形状由 shape 指定"""
    if shape == SH_C:
        C(cx, cy, r, fill, stroke, sw)
        return
    if shape == SH_Q:
        s = r * 1.78
        R(cx - s / 2, cy - s / 2, s, s, fill, rx=s * 0.16, stroke=stroke, sw=sw)
        return
    if shape == SH_T:
        pts = [(cx, cy - r * 1.22), (cx + r * 1.08, cy + r * 0.72),
               (cx - r * 1.08, cy + r * 0.72)]
    elif shape == SH_D:
        pts = [(cx, cy - r * 1.30), (cx + r * 1.10, cy),
               (cx, cy + r * 1.30), (cx - r * 1.10, cy)]
    elif shape == SH_H:
        pts = [(cx + r * 1.10 * math.cos(math.pi / 3 * k),
                cy + r * 1.10 * math.sin(math.pi / 3 * k)) for k in range(6)]
    elif shape == SH_P:
        pts = [(cx + r * 1.18 * math.cos(-math.pi / 2 + 2 * math.pi / 5 * k),
                cy + r * 1.18 * math.sin(-math.pi / 2 + 2 * math.pi / 5 * k)) for k in range(5)]
    else:
        pts = [(cx + r * 1.14 * math.cos(math.pi / 4 * k + math.pi / 8),
                cy + r * 1.14 * math.sin(math.pi / 4 * k + math.pi / 8)) for k in range(8)]
    POLY(pts, fill, stroke, sw)


def dots(x, y, gap=12, r=3.6, axis="x", n=3, fill=SUB):
    """省略号：三个小圆点"""
    c0 = -(n - 1) / 2.0
    for k in range(n):
        if axis == "x":
            C(x + (c0 + k) * gap, y, r, fill)
        else:
            C(x, y + (c0 + k) * gap, r, fill)


def lighten(h, k=0.45):
    """把颜色往白里掺，用于空心形状的描边"""
    h = h.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    f = lambda v: int(round(v + (255 - v) * k))
    return "#%02X%02X%02X" % (f(r), f(g), f(b))


# 每个物理节点画 3 个注册的类 + 2 个「看到但不注册」的类，其余见省略号
NODE_CLASSES = [[SH_Q, SH_D, SH_H],      # 节点一：方 / 菱形 / 六边形
                [SH_C, SH_T, SH_O],      # 节点二：圆 / 三角 / 八边形
                [SH_Q, SH_P]]            # 节点三：方 / 五边形
NODE_UNREG = [[SH_C, SH_T],              # 节点一不注册：圆、三角
              [SH_D, SH_P],              # 节点二不注册：菱形、五边形
              [SH_H, SH_O]]              # 节点三不注册：六边形、八边形

# ==================================================== 上排：三栏
# ---------- 栏一：物理节点
R(30, 60, 286, 545, "none", rx=20, stroke=MOD, sw=1.4, dash="8 6")
T(40, 96, "物理节点", 28, INK, "bold")
T(40, 130, "每个节点持有若干类", 20, SUB)

node_y = [170, 300, 430]
for i, ny in enumerate(node_y):
    R(50, ny, 240, 84, "#FFFFFF", rx=14, stroke=CLD[i], sw=2.4)
    cls = NODE_CLASSES[i]
    for k, sh in enumerate(cls):                      # 实心：本节点注册的类
        mark(72 + k * 34, ny + 42, 11, sh, CLS[i], CLD[i], 1.6)
    for k, sh in enumerate(NODE_UNREG[i]):            # 空心：看得到但不注册的类
        mark(72 + (len(cls) + k) * 34, ny + 42, 11, sh,
             "none", lighten(CLD[i], 0.20), 1.5)
    dots(72 + (len(cls) + len(NODE_UNREG[i])) * 34 + 16, ny + 42, gap=9, r=3.0)

dots(145, 540, gap=13, axis="y")           # 还有更多物理节点
T(40, 585, "空心：看得到，但不注册", 15, SUB)

# ---------- 栏二：可寻址单元
R(388, 60, 422, 580, "none", rx=20, stroke=MOD, sw=1.4, dash="8 6")
T(400, 96, "一类一地址", 28, INK, "bold")
T(400, 130, "vnode 是地址，不是计算单元", 20, SUB)

for i, ny in enumerate(node_y):
    arrow(318, ny + 42, 386, ny + 42, "#9DB0BF", 2.4, mk="ag")
    cls = NODE_CLASSES[i]
    for k, sh in enumerate(cls):
        mark(502 + k * 86, ny + 42, 24, sh, CLS[i], CLD[i], 2.0)
    dx = 502 + (len(cls) - 1) * 86 + 46
    dots(dx, ny + 42, gap=13)              # 还有更多类
    # 同一节点的若干单元用虚线框圈起，宽度随内容
    R(470, ny + 8, dx + 46 - 470, 68, "none", rx=34, stroke=CLD[i], sw=1.6, dash="7 5")
    T(474, ny + 100, "同一节点", 18, CLD[i])

dots(620, 566, gap=13, axis="y")           # 还有更多节点
T(400, 616, "一个节点占用若干独立的 vnode", 17, SUB)

# ---------- 栏三：覆盖网
R(890, 60, 882, 580, "none", rx=20, stroke=MOD, sw=1.4, dash="8 6")
T(910, 112, "覆盖网", 28, INK, "bold")
T(910, 146, "每个顶点是一个地址", 20, SUB)
T(910, 176, "按坐标远近相连，连线为边", 18, SUB)


def ring(n, rad, phase=-math.pi / 2, jit=(1.0,)):
    pts = []
    for k in range(n):
        a = phase + 2 * math.pi * k / n
        f = jit[k % len(jit)]
        pts.append((round(rad * f * math.cos(a)), round(rad * f * math.sin(a))))
    return pts


# 顶点按类聚成簇：同一类的若干副本坐标相近，故挨在一起；簇的形状就是该类的形状。
# 一个颜色只代表一个物理节点，故同一簇内颜色互不重复：
# 簇里有几颗，就是有几个节点持有该类。仍画不下的用省略号表示。
# 每个椭圆是一类的全部副本，画 2~4 颗。
#   cols —— 该副本所属节点在左栏出现过，用该节点的颜色画实心；
#   hol  —— 该副本所属节点没在左栏画出（节点不止三个，见左栏省略号），画成空心。
# 一个颜色代表一个节点，故同一椭圆内 cols 互不重复。
clusters = [
    dict(c=(1042, 312), sh=SH_C, cols=[1], hol=2, tail=False),
    dict(c=(1252, 278), sh=SH_Q, cols=[0, 2], hol=2, tail=False),
    dict(c=(1482, 312), sh=SH_T, cols=[1], hol=1, tail=True),
    dict(c=(1692, 275), sh=SH_D, cols=[0], hol=2, tail=False),
    dict(c=(1100, 472), sh=SH_H, cols=[0], hol=2, tail=False),
    dict(c=(1340, 480), sh=SH_P, cols=[2], hol=1, tail=True),
    dict(c=(1600, 455), sh=SH_O, cols=[1], hol=2, tail=False),
]


def layout(n, rad=44):
    """簇内成员的位置：两颗横排，其余围成环"""
    if n == 2:
        return [(-rad, 0), (rad, 0)]
    return ring(n, rad)


vpos, groups = [], []
for gi, cl in enumerate(clusters):
    cx, cy = cl["c"]
    n = len(cl["cols"]) + cl["hol"]
    cl["pts"] = layout(n)
    idx = []
    for k, (dx, dy) in enumerate(cl["pts"]):
        ci = cl["cols"][k] if k < len(cl["cols"]) else -1   # -1 表示空心
        idx.append(len(vpos))
        vpos.append((cx + dx, cy + dy, ci, cl["sh"]))
    groups.append(idx)

edges = []
for cl, g in zip(clusters, groups):        # 簇内：同类副本依次相连
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
    mxx = max(abs(dx) for dx, dy in cl["pts"])
    myy = max(abs(dy) for dx, dy in cl["pts"])
    rx = mxx + (34 if cl["tail"] else 30)
    ry = myy + (46 if cl["tail"] else 30)
    P.append(
        f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="none" '
        f'stroke="{SUB}" stroke-width="1.6" stroke-dasharray="7 5" opacity="0.9"/>'
    )

for (vx, vy, ci, sh) in vpos:
    if ci < 0:
        mark(vx, vy, 18, sh, "none", HOL, 1.6)
    else:
        mark(vx, vy, 18, sh, CLS[ci], CLD[ci], 2.0)

# 画不下的副本用省略号表示（放在簇的正下方）
for cl in clusters:
    if not cl["tail"]:
        continue
    cx, cy = cl["c"]
    myy = max(abs(dy) for dx, dy in cl["pts"])
    dots(cx, cy + myy + 26, gap=11, r=3.4)

# 引线标注：顶点、边
L(vpos[0][0], vpos[0][1], 962, 274, SUB, 1.4, dash="5 4")
T(906, 280, "vnode", 20, INK, "bold")
mx, my = cross_pairs[7]
L((vpos[mx][0] + vpos[my][0]) / 2, (vpos[mx][1] + vpos[my][1]) / 2, 1694, 366, SUB, 1.4, dash="5 4")
T(1700, 372, "边", 20, INK, "bold")

# 底部说明（三行短句，右端都留出红线经过的位置，避免压线）
T(900, 566, "形状区分类，颜色区分节点", 17, SUB)
T(900, 592, "虚线椭圆圈住同一类的副本", 17, SUB)
T(900, 618, "空心：未注册；省略号：还有更多副本", 17, SUB)

# ==================================================== 下排：坐标的来历
R(250, 652, 1270, 184, "none", rx=20, stroke=MOD, sw=1.4, dash="8 6")
T(40 + BX, 690, "坐标的来历", 28, INK, "bold")

by = 740   # 基线
# 输入
R(50 + BX, 700, 96, 80, AF, rx=14, stroke=LINE, sw=1.6)
for k in range(3):
    L(66 + BX, 716 + k * 20, 130 + BX, 716 + k * 20, SUB, 1.4, op=0.55)
T(98 + BX, 806, "输入", 21, SUB, anchor="middle")

arrow(152 + BX, by, 208 + BX, by, "#9DB0BF", 2.6, mk="ag")

# 编码器
R(212 + BX, 698, 250, 84, "#FFFFFF", rx=14, stroke=BLUE, sw=2.2)
T(337 + BX, 736, "编码器", 22, BLUE, "bold", "middle")
T(337 + BX, 766, "不在本地训练", 18, SUB, anchor="middle")
T(337 + BX, 806, "一次发布", 21, SUB, anchor="middle")

arrow(468 + BX, by, 522 + BX, by, "#9DB0BF", 2.6, mk="ag")

# 高维特征
for k in range(14):
    hh = 30 + ((k * 37) % 34)
    L(540 + BX + k * 15, by + 40, 540 + BX + k * 15, by + 40 - hh, CLD[0], 5.0, op=0.75)
T(645 + BX, 806, "特征", 21, SUB, anchor="middle")

arrow(760 + BX, by, 814 + BX, by, "#9DB0BF", 2.6, mk="ag")

# 坐标
R(818 + BX, 698, 230, 84, "#FFFFFF", rx=14, stroke=RED, sw=2.2)
T(933 + BX, 736, "坐标", 22, RED, "bold", "middle")
T(933 + BX, 766, "即 384 维特征", 18, SUB, anchor="middle")
T(933 + BX, 806, "不作变换", 21, SUB, anchor="middle")

arrow(1054 + BX, by, 1108 + BX, by, "#9DB0BF", 2.6, mk="ag")

# 一个坐标
C(1140 + BX, by, 24, CLS[1], CLD[1], 2.2)
T(1140 + BX, 806, "一个坐标", 21, SUB, anchor="middle")

# 从圆顶竖直向上指进覆盖网
arrow(1140 + BX, by - 24, 1365, 505, RED, 2.4, dash="8 6")

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
