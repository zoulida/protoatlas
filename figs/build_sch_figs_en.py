# -*- coding: utf-8 -*-
"""第 4 章三张示意图：vnode 构造 / 贪婪下降的局部极小 / 按类去重

版式与 build_fig1.py 一致：1800 宽画布，SimSun serif。
排版原则：图形主体靠左，注释文字放在预留的右侧文字栏，避免文字压在点上。
"""

import os
import math
import random
FIGDIR = os.path.dirname(os.path.abspath(__file__))

INK = "#1F2D3A"
SUB = "#5B6B7B"
LINE = "#9DB0BF"
AF = "#EDF2F7"
BF = "#FBF4EE"
GF = "#F4F6F8"
BLUE = "#3D6E8F"
RED = "#B04A32"
GRN = "#4A7A5C"
AMB = "#C08A2E"
CLS = ["#CFE0EE", "#F3DECF", "#DCEADA"]
CLD = ["#6C93AE", "#C08A63", "#6E9670"]

P = []


def T(x, y, s, size=26, fill=INK, weight="normal", anchor="start", op=1.0):
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


def C(cx, cy, r, fill, stroke="none", sw=0, op=1.0):
    P.append(
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke}" '
        f'stroke-width="{sw}" opacity="{op}"/>'
    )


def L(x1, y1, x2, y2, stroke=LINE, sw=1.6, dash=None, op=1.0, marker=False):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    m = ' marker-end="url(#ar)"' if marker else ""
    P.append(
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" '
        f'stroke-width="{sw}" opacity="{op}"{d}{m}/>'
    )


def PATH(d, stroke=RED, sw=3.2, dash=None, marker=True, op=1.0):
    ds = f' stroke-dasharray="{dash}"' if dash else ""
    m = ' marker-end="url(#ar)"' if marker else ""
    P.append(
        f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{sw}" '
        f'opacity="{op}"{ds}{m}/>'
    )


def cluster(cx, cy, r, n, color, seed, size=6.6):
    rng = random.Random(seed)
    for _ in range(n):
        a = rng.uniform(0, 2 * math.pi)
        d = r * math.sqrt(rng.random()) * 0.92
        C(cx + d * math.cos(a), cy + d * math.sin(a) * 0.74, size, color,
          "#FFFFFF", 0.6)


def wrap(body, w, h):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
<defs>
  <marker id="ar" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="#B04A32"/>
  </marker>
</defs>
<rect width="{w}" height="{h}" fill="#FFFFFF"/>
<g font-family="SimSun, 'Times New Roman', serif">
{body}
</g>
</svg>'''


# ============================================================ 图 4-2
def fig4_2():
    P.clear()
    W, H = 1800, 720

    # ---- 左：物理节点
    R(40, 60, 430, 560, "none", rx=20, stroke=LINE, sw=1.4, dash="8 6")
    T(68, 108, "Node n (host of its vnodes)", 23, INK, "bold")
    R(70, 134, 370, 96, "#FFFFFF", rx=8, stroke=BLUE, sw=2.2)
    T(96, 172, "Local model: trained here only", 17, INK)
    T(96, 208, "one pass scores local classes", 17, SUB)

    T(68, 276, "Local data (never shared)", 21, INK, "bold")
    for i, (cx, cy, lab) in enumerate([(140, 350, "class a"), (255, 385, "class b"),
                                       (370, 345, "class c")]):
        cluster(cx, cy, 40, 12, CLD[i], seed=10 + i)
        T(cx, cy + 76, lab, 21, CLD[i], "bold", "middle")

    T(68, 512, "local sample mean = class prototype", 17, SUB)
    T(68, 552, "3 classes registered as 3 vnodes", 19, INK, "bold")

    # ---- 中：坐标空间
    R(520, 60, 610, 560, "none", rx=20, stroke=LINE, sw=1.4, dash="8 6")
    T(548, 108, "Semantic coordinate space (D-dim)", 23, INK, "bold")
    for i, (cx, cy) in enumerate([(645, 275), (825, 400), (1010, 260)]):
        cluster(cx, cy, 58, 24, CLD[i], seed=30 + i)
        C(cx, cy, 11, "#FFFFFF", CLD[i], 3.4)
        T(cx, cy - 78, f"p(n,{chr(97 + i)})", 21, CLD[i], "bold", "middle")
    T(548, 520, "coordinates come only from φ", 17, SUB)
    T(548, 560, "z=φ(x)", 26, INK, "bold")
    T(548, 598, "the same x maps to the same z on any node", 17, SUB)

    # ---- 右：覆盖网（上 vnode 盘、下节点盘，透视椭圆）
    R(1180, 60, 580, 560, "none", rx=20, stroke=LINE, sw=1.4, dash="8 6")
    T(1208, 104, "Overlay network", 23, INK, "bold")
    T(1201, 146, "vnode layer", 18, SUB)

    # 两层的虚线平行四边形框（作者 2026-11-16 微调时加的，形态照搬他的 drawio）
    for pts in [((1306, 137.6), (1714, 137.6), (1678, 280.0), (1270, 280.0)),
                ((1302, 359.8), (1680, 359.8), (1614, 492.7), (1236, 492.7))]:
        P.append('<polygon points="' + " ".join(f"{x},{y}" for x, y in pts) +
                 '" fill="none" stroke="#9DB0BF" stroke-width="1.4" stroke-dasharray="8 6"/>')

    angs = [-90, -18, 54, 126, 198]
    ux, uy, urx, ury = 1470, 214, 146, 44
    lx, ly, lrx, lry = 1470, 428, 152, 46
    cr = 27
    clusters = [(ux + urx * math.cos(math.radians(a)),
                 uy + ury * math.sin(math.radians(a))) for a in angs]
    bodies = [(lx + lrx * 0.90 * math.cos(math.radians(a)),
               ly + lry * 0.90 * math.sin(math.radians(a))) for a in angs]

    P.append(f'<ellipse cx="{ux}" cy="{uy + 5}" rx="{urx}" ry="{ury}" fill="#F1F5F8" '
             f'stroke="none"/>')
    P.append(f'<ellipse cx="{ux}" cy="{uy}" rx="{urx}" ry="{ury}" fill="#FCFDFE" '
             f'stroke="#8FA2B2" stroke-width="1.6"/>')
    P.append(f'<ellipse cx="{lx}" cy="{ly + 5}" rx="{lrx}" ry="{lry}" fill="#F1F5F8" '
             f'stroke="none"/>')
    P.append(f'<ellipse cx="{lx}" cy="{ly}" rx="{lrx}" ry="{lry}" fill="#FCFDFE" '
             f'stroke="#AEC0CE" stroke-width="1.3" stroke-dasharray="6 5"/>')

    mine_idx = {4: 0, 0: 1, 1: 2}
    mine = {}
    for ci, k in mine_idx.items():
        cx, cy = clusters[ci]
        mine[k] = (cx - 1, cy + 10)

    for k, (px, py) in mine.items():
        L(bodies[0][0], bodies[0][1] - 18, px, py + 9, "#A8BFD0", 1.4)

    for cx, cy in clusters:
        P.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{cr}" fill="#FBFCFD" '
                 f'stroke="#B9C5CF" stroke-width="1.1" stroke-dasharray="5 4"/>')
    offs = [(-10, -8), (10, -7), (-11, 6), (9, 8)]
    for ci, (cx, cy) in enumerate(clusters):
        for k, (dx, dy) in enumerate(offs):
            if ci in mine_idx and k == 3:
                continue
            C(cx + dx + 1, cy + dy + 2, 7.0, "#DCE3E9")
            C(cx + dx, cy + dy, 7.0, "#EAEFF4", "#AEBCC8", 1.1)
    for k, (px, py) in mine.items():
        C(px + 1, py + 2, 9.6, "#C9D6E0")
        C(px, py, 9.6, CLS[k], CLD[k], 2.0)

    T(1203, 372, "node layer", 18, SUB)
    for k in range(len(bodies)):
        x1, y1 = bodies[k]
        x2, y2 = bodies[(k + 1) % len(bodies)]
        L(x1, y1, x2, y2, "#B4C4D1", 1.3, dash="6 5")
    for bi, (bx, by) in enumerate(bodies):
        R(bx - 16, by - 14, 34, 34, "#DCE3E9", rx=6)
        if bi == 0:
            R(bx - 18, by - 18, 36, 36, "#FFFFFF", rx=6, stroke=BLUE, sw=3.2)
        else:
            R(bx - 16, by - 16, 32, 32, "#FFFFFF", rx=6, stroke="#C3CED8", sw=1.6)
    T(1470, 424, "node n", 16, BLUE, "bold", "middle")


    # ---- 流程
    L(420, 330, 512, 330, RED, 2.4, marker=True)
    R(416, 292, 100, 24, "#FFFFFF", rx=4)        # 白底衬，免得虚线框线穿过字
    T(466, 310, "coordinate", 17, RED, anchor="middle")
    L(1130, 330, 1172, 330, RED, 2.4, marker=True)
    R(1110, 292, 82, 24, "#FFFFFF", rx=4)
    T(1151, 310, "register", 17, RED, anchor="middle")

    T(40, 682, "A node specializing in K classes occupies K addresses; "
               "physical nodes and addresses stand in a host relation, not one-to-one",
      17, SUB)
    return wrap("\n".join(P), W, H)


# ============================================================ 图 4-3
def fig4_3():
    P.clear()
    W, H = 1800, 680

    def panel(ox, fill, title, sub, multi):
        R(ox, 55, 830, 500, fill, rx=10)
        T(ox + 26, 100, title, 26, INK, "bold")
        T(ox + 26, 138, sub, 20, SUB)

        # 四个类簇，全部落在左侧 60%
        cs = [(ox + 120, 250), (ox + 290, 175), (ox + 175, 420), (ox + 400, 330)]
        for i, (cx, cy) in enumerate(cs):
            cluster(cx, cy, 52, 16, CLD[i % 3], seed=50 + i, size=6.0)

        gx, gy = cs[3]
        C(gx, gy, 12, "#FFFFFF", RED, 3.0)
        T(gx + 20, gy + 74, "真类", 20, RED, "bold")

        tx = ox + 470
        if not multi:
            sx, sy = ox + 55, 300
            C(sx, sy, 10, RED, "#FFFFFF", 2.0)
            T(sx - 6, sy + 40, "起点", 20, RED, "middle")
            PATH(f"M {sx + 12} {sy - 8} L {ox + 105} {sy - 44} L {ox + 168} {sy - 52}")
            C(ox + 168, sy - 52, 14, "none", AMB, 3.0)
            T(tx, 205, "走到本簇边缘", 22, AMB, "bold")
            T(tx, 238, "邻居都比它远，", 20, SUB)
            T(tx, 266, "贪婪判定终止", 20, SUB)
            T(tx, 330, "簇间没有近邻边", 21, INK, "bold")
            T(tx, 362, "边主要存在于簇内，", 20, SUB)
            T(tx, 390, "空白的另一侧无法跨越", 20, SUB)
            T(tx, 452, "20.4%", 30, RED, "bold")
            T(tx, 484, "的查询未到达全局最近的原型", 19, SUB)
        else:
            starts = [(ox + 42, 330), (ox + 330, 470), (ox + 300, 258)]
            for j, (sx, sy) in enumerate(starts):
                C(sx, sy, 10, RED, "#FFFFFF", 2.0)
                T(sx - 4, sy + 34, f"起点 {j + 1}", 19, RED)
            # 起点 1 落在第一簇边缘
            PATH(f"M {starts[0][0] + 12} {starts[0][1] - 10} L {ox + 90} 300 L {ox + 138} 274")
            C(ox + 138, 274, 13, "none", AMB, 2.6)
            # 起点 2 落在第三簇边缘
            PATH(f"M {starts[1][0] - 10} {starts[1][1] - 10} L {ox + 250} 470 L {ox + 205} 452")
            C(ox + 205, 452, 13, "none", AMB, 2.6)
            # 起点 3 越过分界，进入目标簇
            PATH(f"M {starts[2][0] - 10} {starts[2][1] + 6} L {ox + 250} 300 L {ox + 330} 330 L {gx - 22} {gy - 16}")
            C(gx - 22, gy - 16, 15, "none", GRN, 3.0)

            T(tx, 205, "三支并行下降", 22, GRN, "bold")
            T(tx, 238, "各落各的局部极小，", 20, SUB)
            T(tx, 266, "任一支进入目标簇即命中", 20, SUB)
            T(tx, 330, "到达后向邻居扩散一跳", 21, INK, "bold")
            T(tx, 362, "把候选补全再排序", 20, SUB)
            T(tx, 452, "0.8883", 30, GRN, "bold")
            T(tx, 484, "召回@3，询问 19.4 个 vnode", 19, SUB)

    panel(40, GF, "(a) 单起点：被簇边界截住",
          "近邻表只连最近的若干点，边因此主要落在簇内", False)
    panel(930, BF, "(b) 三起点加一跳扩散：从别的区域绕进去",
          "同样的询问预算，起点数比扩散半径更划算", True)

    T(40, 640, "实测：3 起点 1 跳用 19.4 个询问得到 0.8883，"
               "优于 1 起点 2 跳的 55.7 个询问与 0.8573；全局扫描为 0.9143",
      21, SUB)
    return wrap("\n".join(P), W, H)


# ============================================================ 图 4-4
def fig4_4():
    """二维示意：同类副本挤在一处，前四名被同一类占满，真类排第五。"""
    P.clear()
    W, H = 1800, 720
    K = 7.6                      # 每单位距离的像素数，标注值按此等比例放置
    qx, qy = 700, 372

    def at(dist, deg):
        a = math.radians(deg)
        return qx + dist * K * math.cos(a), qy + dist * K * math.sin(a)

    def CC(cx, cy, r, stroke, sw=1.4, dash="7 6", op=0.55):
        P.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{stroke}" '
                 f'stroke-width="{sw}" stroke-dasharray="{dash}" opacity="{op}"/>')

    # ---- 左：坐标平面
    R(40, 56, 1090, 600, GF, rx=10)
    T(70, 100, "同一查询在语义坐标空间中的位置（二维示意）", 25, INK, "bold")
    L(110, 560, 1070, 560, LINE, 1.2, op=0.6)
    L(152, 132, 152, 600, LINE, 1.2, op=0.6)
    T(1064, 586, "坐标轴 1", 18, SUB, anchor="end")
    T(160, 150, "坐标轴 2", 18, SUB)

    # 以查询为心的两个球：内圈取第 4 名，外圈取第 5 名
    CC(qx, qy, 21.33 * K, RED, 1.5)
    CC(qx, qy, 28.37 * K, GRN, 1.5)

    deer = [(20.94, 152), (21.12, 157), (21.33, 162), (21.33, 167)]
    bird = [(28.37, -48), (28.90, -52), (29.60, -56), (30.20, -60)]

    # 查询到最近两个候选的连线
    ax, ay = at(*deer[0])
    bx, by = at(*bird[0])
    L(qx, qy, ax, ay, RED, 1.8, dash="8 6", op=0.85, marker=True)
    L(qx, qy, bx, by, GRN, 1.8, dash="8 6", op=0.85, marker=True)
    T(qx + (ax - qx) * 0.30, qy + (ay - qy) * 0.30 - 16, "20.94", 20, RED,
      "bold", "middle")
    T(qx + (bx - qx) * 0.52, qy + (by - qy) * 0.52 - 14, "28.37", 20, GRN,
      "bold", "middle")

    for d, a in deer:
        x, y = at(d, a)
        C(x, y, 9, "#FFFFFF", BLUE, 2.6)
    for d, a in bird:
        x, y = at(d, a)
        C(x, y, 9, "#FFFFFF", GRN, 2.6)

    C(qx, qy, 9, RED, "#FFFFFF", 2.2)
    T(qx + 16, qy + 38, "q：查询", 21, RED, "bold")

    L(300, 446, 528, 452, LINE, 1.2, op=0.9)
    T(180, 434, "鹿的原型：4 个副本几乎重合", 21, BLUE, "bold")
    T(180, 464, "20.94 至 21.33，团内距离 1.43", 19, SUB)
    T(700, 126, "鸟的原型：最近的副本 28.37，排在第五", 19, GRN, "bold")
    L(820, 138, 824, 186, GRN, 1.2, op=0.8)

    T(70, 614, "两个虚线球以查询为心，半径分别取第 4 名与第 5 名的距离",
      19, SUB)
    T(70, 642, "内球含 4 个鹿副本、0 个鸟副本，故按半径取 3 个必然全属鹿",
      19, RED, "bold")

    # ---- 右：两份名单
    R(1160, 56, 600, 246, BF, rx=10)
    T(1190, 100, "未去重：按距离取前 3", 23, RED, "bold")
    for i, (nm, v) in enumerate([("鹿 · 副本 1", "20.94"), ("鹿 · 副本 2", "21.12"),
                                 ("鹿 · 副本 3", "21.33")]):
        y = 142 + i * 34
        T(1190, y, nm, 20, BLUE)
        T(1718, y, v, 20, BLUE, anchor="end")
    T(1190, 258, "三名候选同属一类，", 20, RED)
    T(1190, 286, "平均只含 1.03 个不同的类", 20, RED)

    R(1160, 332, 600, 324, "#F1F7F2", rx=10)
    T(1190, 376, "按类去重：同类只留最近的一个", 23, GRN, "bold")
    for i, (mark, nm, v, col) in enumerate(
            [("①", "鹿", "20.94", BLUE),
             ("②", "鸟（真类）", "28.37", GRN),
             ("③", "下一个不同的类 …", "29.60", AMB)]):
        y = 420 + i * 36
        T(1190, y, mark, 20, col)
        T(1224, y, nm, 20, col)
        T(1718, y, v, 20, col, anchor="end")
    T(1190, 556, "真类进入名单第二，", 20, GRN)
    T(1190, 584, "通信量与未去重时相同", 20, GRN)
    T(1190, 628, "去重只改变名单构成，", 19, SUB)
    T(1190, 652, "不改变命中@1 的 0.9542", 19, SUB)

    T(40, 700, "去重不增加任何通信：3 候选召回由 0.9542 升至 0.9938", 19, SUB)
    return wrap("\n".join(P), W, H)


if __name__ == "__main__":
    # 输出文件名与正文图号一致（2026-10-22 重排，图 4-2 为字节构成图）
    for name, fn in [("fig4_3_vnode", fig4_2)]:
        path = os.path.join(FIGDIR, "en", f"{name}.svg")
        with open(path, "w", encoding="utf-8") as f:
            f.write(fn())
        print("written", path)
