# -*- coding: utf-8 -*-
"""把示意图 SVG 转成 Illustrator 友好的版本。

两处改动：
1) width/height 由 px 数值改为 pt（Illustrator CS3 按 pt 建画板；若给纯数字，
   它会当成 1800pt 塞进默认 A4 画板，结果只显示左上角一块）
2) <marker> 箭头展开成实心三角形 path（AI 对 marker 支持不稳定，容易丢箭头）

用法：python make_ai_svg.py fig3_1_vnode_map:fig3_1_vnode_ai fig4_3_vnode:fig4_2_vnode_ai
"""
import math
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
MK_COLOR = {"ar": "#B04A32", "ag": "#8FA3B4"}


def marker_colors(svg):
    """从 defs 里读出每个 marker 的填充色。"""
    out = dict(MK_COLOR)
    for m in re.finditer(r'<marker id="(\w+)"[^>]*>\s*<path[^>]*fill="(#[0-9A-Fa-f]{6})"', svg):
        out[m.group(1)] = m.group(2)
    return out


def px_to_pt(svg):
    m = re.search(r'width="(\d+)" height="(\d+)"', svg)
    if not m:
        return svg
    w, h = int(m.group(1)), int(m.group(2))
    return svg.replace(m.group(0), 'width="%gpt" height="%gpt"' % (w * 0.75, h * 0.75), 1)


def expand_markers(svg, colors):
    """把 line 上的 marker-end 换成同尺寸的三角形。"""
    count = [0]

    def rep(m):
        tag = m.group(0)
        mid = re.search(r'marker-end="url\(#(\w+)\)"', tag).group(1)

        def g(k):
            return float(re.search(k + r'="([-0-9.]+)"', tag).group(1))

        x1, y1, x2, y2 = g("x1"), g("y1"), g("x2"), g("y2")
        sw = g("stroke-width") if 'stroke-width="' in tag else 1.0
        dx, dy = x2 - x1, y2 - y1
        L = math.hypot(dx, dy) or 1.0
        ux, uy = dx / L, dy / L
        nx, ny = -uy, ux
        k = 0.7 * sw                      # markerWidth 7 / viewBox 10 × strokeWidth
        p0 = (x2 + 2 * k * ux, y2 + 2 * k * uy)
        p1 = (x2 - 8 * k * ux - 5 * k * nx, y2 - 8 * k * uy - 5 * k * ny)
        p2 = (x2 - 8 * k * ux + 5 * k * nx, y2 - 8 * k * uy + 5 * k * ny)
        line = re.sub(r' marker-end="url\(#\w+\)"', "", tag)
        tri = ('<path d="M %.2f %.2f L %.2f %.2f L %.2f %.2f Z" fill="%s"/>'
               % (p0[0], p0[1], p1[0], p1[1], p2[0], p2[1], colors.get(mid, "#000000")))
        count[0] += 1
        return line + tri

    svg = re.sub(r'<line[^>]*marker-end="url\(#\w+\)"[^>]*/>', rep, svg)
    return svg, count[0]


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return
    for a in args:
        src_name, _, dst_name = a.partition(":")
        dst_name = dst_name or (src_name + "_ai")
        svg = open(os.path.join(ROOT, src_name + ".svg"), encoding="utf-8").read()
        colors = marker_colors(svg)
        svg = px_to_pt(svg)
        svg, n = expand_markers(svg, colors)
        out = os.path.join(ROOT, dst_name + ".svg")
        open(out, "w", encoding="utf-8", newline="\n").write(svg)
        left = svg.count("marker-end")
        print("%s -> %s  箭头 %d  残留 marker %d" % (src_name, dst_name + ".svg", n, left))


if __name__ == "__main__":
    main()
