# -*- coding: utf-8 -*-
"""把 figs 下手工写的 SVG 转成原生 .drawio（mxGraphModel XML）

用法:
    python svg2drawio.py <in.svg> <out.drawio> [--font "Times New Roman"]

转换规则:
    rect      -> vertex, rounded=1, arcSize 按 rx/min(w,h) 折算
    circle    -> vertex, ellipse
    ellipse   -> vertex, ellipse（虚线椭圆照样保留）
    line      -> edge（有 marker-end 则 classic 箭头）
    text      -> vertex, 纯文本标签；宽度用真实字体度量，避免框太窄
    <defs> / 满画布白底 rect 跳过

坐标 1:1 照搬，页面尺寸等于 SVG 画布，打开后位置与 PDF 完全一致。
"""
import sys
import os
import re
from xml.etree import ElementTree as ET

from PIL import ImageFont

FONT_DIR = r"C:\Windows\Fonts"
_cache = {}

# 字体名 -> (常规字体文件, 粗体字体文件, ttc 索引)
FONT_FILES = {
    "SimSun": ("simsun.ttc", "simsun.ttc", 0),
    "Times New Roman": ("times.ttf", "timesbd.ttf", 0),
}


def text_width(s, size, bold, family="Times New Roman"):
    """用真实字体量文本宽度（px），保证标签框不窄于文字"""
    key = (bold, size, family)
    if key not in _cache:
        reg, bd, idx = FONT_FILES.get(family, FONT_FILES["Times New Roman"])
        path = os.path.join(FONT_DIR, bd if bold else reg)
        _cache[key] = ImageFont.truetype(path, max(1, int(round(size))), index=idx)
    return _cache[key].getlength(s)


def esc(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
             .replace('"', "&quot;").replace("\n", "&#10;"))


def num(v, d=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def main():
    src, dst = sys.argv[1], sys.argv[2]
    font = "Times New Roman"
    if "--font" in sys.argv:
        font = sys.argv[sys.argv.index("--font") + 1]

    tree = ET.parse(src)
    root = tree.getroot()
    W = num(root.get("width"))
    H = num(root.get("height"))

    cells = []
    nid = [1]

    def cell(value, style, geom, vertex=True):
        nid[0] += 1
        kind = 'vertex="1"' if vertex else 'edge="1"'
        cells.append(
            f'        <mxCell id="{nid[0]}" value="{esc(value)}" style="{style}" '
            f'{kind} parent="1">\n'
            f'          <mxGeometry {geom} as="geometry"' +
            ('/>' if vertex else '>\n' + geom_pts[0] + '          </mxGeometry>') +
            '\n        </mxCell>'
        )

    geom_pts = [""]

    def walk(el):
        tag = el.tag.split("}")[-1]

        if tag == "defs":
            return

        if tag == "rect":
            x, y = num(el.get("x")), num(el.get("y"))
            w, h = num(el.get("width")), num(el.get("height"))
            if x == 0 and y == 0 and w >= W - 1 and h >= H - 1:
                return  # 满画布白底，页面本身已是白色
            rx = num(el.get("rx"))
            fill = el.get("fill", "#FFFFFF")
            stroke = el.get("stroke", "none")
            sw = num(el.get("stroke-width"))
            op = num(el.get("opacity"), 1.0)
            st = ["rounded=1;whiteSpace=wrap;html=1;"]
            st.append(f"arcSize={min(50, round(100 * rx / max(1, min(w, h)), 1))};")
            st.append(f"fillColor={'none' if fill in (None, 'none') else fill};")
            st.append(f"strokeColor={'none' if stroke in (None, 'none') else stroke};")
            if sw:
                st.append(f"strokeWidth={sw:g};")
            if el.get("stroke-dasharray"):
                st.append("dashed=1;fixDash=1;dashPattern="
                          + el.get("stroke-dasharray") + ";")
            if op < 1:
                st.append(f"opacity={round(op * 100):d};")
            geom_pts[0] = ""
            cell("", "".join(st), f'x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}"')
            return

        if tag == "polygon":
            # 顶点数决定形状：3 三角 / 4 菱形 / 5 五边形 / 6 六边形 / 8 八边形
            pts = [(float(a.split(",")[0]), float(a.split(",")[1]))
                   for a in el.get("points", "").split() if "," in a]
            if not pts:
                return
            SHAPES = {3: "shape=triangle;direction=north;",
                      4: "shape=rhombus;",
                      5: "shape=mxgraph.basic.pentagon;",
                      6: "shape=hexagon;",
                      8: "shape=mxgraph.basic.octagon;"}
            shape = SHAPES.get(len(pts))
            if shape is None:
                print("  警告: %d 边形没有对应形状，退回矩形" % len(pts))
                shape = "rounded=0;"
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            x0, y0 = min(xs), min(ys)
            w, h = max(xs) - x0, max(ys) - y0
            fill = el.get("fill", "#FFFFFF")
            stroke = el.get("stroke", "none")
            sw = num(el.get("stroke-width"))
            op = num(el.get("opacity"), 1.0)
            st = [shape + "whiteSpace=wrap;html=1;"]
            st.append(f"fillColor={'none' if fill in (None, 'none') else fill};")
            st.append(f"strokeColor={'none' if stroke in (None, 'none') else stroke};")
            if sw:
                st.append(f"strokeWidth={sw:g};")
            if op < 1:
                st.append(f"opacity={round(op * 100):d};")
            geom_pts[0] = ""
            cell("", "".join(st), f'x="{x0:g}" y="{y0:g}" width="{w:g}" height="{h:g}"')
            return

        if tag == "circle" or tag == "ellipse":
            cx, cy = num(el.get("cx")), num(el.get("cy"))
            rx = num(el.get("r")) if tag == "circle" else num(el.get("rx"))
            ry = num(el.get("r")) if tag == "circle" else num(el.get("ry"))
            fill = el.get("fill", "#FFFFFF")
            stroke = el.get("stroke", "none")
            sw = num(el.get("stroke-width"))
            op = num(el.get("opacity"), 1.0)
            st = ["ellipse;whiteSpace=wrap;html=1;"]
            st.append(f"fillColor={'none' if fill in (None, 'none') else fill};")
            st.append(f"strokeColor={'none' if stroke in (None, 'none') else stroke};")
            if sw:
                st.append(f"strokeWidth={sw:g};")
            if el.get("stroke-dasharray"):
                st.append("dashed=1;fixDash=1;dashPattern="
                          + el.get("stroke-dasharray") + ";")
            if op < 1:
                st.append(f"opacity={round(op * 100):d};")
            geom_pts[0] = ""
            cell("", "".join(st),
                 f'x="{cx - rx:g}" y="{cy - ry:g}" width="{2 * rx:g}" height="{2 * ry:g}"')
            return

        if tag == "line":
            x1, y1 = num(el.get("x1")), num(el.get("y1"))
            x2, y2 = num(el.get("x2")), num(el.get("y2"))
            stroke = el.get("stroke", "#000000")
            sw = num(el.get("stroke-width"), 1.0)
            op = num(el.get("opacity"), 1.0)
            has_arrow = bool(el.get("marker-end"))
            st = ["edgeStyle=none;rounded=0;html=1;"]
            st.append(f"strokeColor={stroke};strokeWidth={sw:g};")
            st.append("endArrow=classic;endSize=7;endFill=1;" if has_arrow
                      else "endArrow=none;startArrow=none;")
            if el.get("stroke-dasharray"):
                st.append("dashed=1;fixDash=1;dashPattern="
                          + el.get("stroke-dasharray") + ";")
            if op < 1:
                st.append(f"opacity={round(op * 100):d};")
            geom_pts[0] = (f'          <mxPoint x="{x1:g}" y="{y1:g}" as="sourcePoint"/>\n'
                           f'          <mxPoint x="{x2:g}" y="{y2:g}" as="targetPoint"/>\n')
            cell("", "".join(st), 'width="50" height="50" relative="1"', vertex=False)
            return

        if tag == "text":
            s = "".join(el.itertext())
            if not s.strip():
                return
            x, y = num(el.get("x")), num(el.get("y"))
            size = num(el.get("font-size"), 20)
            fill = el.get("fill", "#000000")
            weight = el.get("font-weight", "normal")
            anchor = el.get("text-anchor", "start")
            op = num(el.get("opacity"), 1.0)
            bold = weight in ("bold", "600", "700")

            tw = text_width(s, size, bold, font) + 3
            th = size * 1.34
            if anchor == "middle":
                bx = x - tw / 2
                align = "center"
            elif anchor == "end":
                bx = x - tw
                align = "right"
            else:
                bx = x
                align = "left"
            by = y - size * 0.34 - th / 2   # y 是基线，换算成框中线

            st = (f"text;html=1;whiteSpace=nowrap;align={align};verticalAlign=middle;"
                  f"spacing=0;spacingLeft=0;fontSize={size:g};"
                  f"fontFamily={font};fontColor={fill};"
                  f"fontStyle={1 if bold else 0};")
            if op < 1:
                st.append(f"opacity={round(op * 100):d};")
            geom_pts[0] = ""
            cell(s, "".join(st), f'x="{bx:g}" y="{by:g}" width="{tw:g}" height="{th:g}"')
            return

        for ch in list(el):
            walk(ch)

    for ch in list(root):
        walk(ch)

    body = "\n".join(cells)
    xml = f'''<mxfile host="drawio" agent="svg2drawio" version="32.6.10">
  <diagram id="fig" name="{os.path.splitext(os.path.basename(dst))[0]}">
    <mxGraphModel dx="{W:g}" dy="{H:g}" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="{W:g}" pageHeight="{H:g}" background="#FFFFFF" math="0" shadow="0">
      <root>
        <mxCell id="0"/>
        <mxCell id="1" parent="0"/>
{body}
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
'''
    with open(dst, "w", encoding="utf-8") as f:
        f.write(xml)
    print(f"{dst}  cells={len(cells)}  page={W:g}x{H:g}")


if __name__ == "__main__":
    main()
