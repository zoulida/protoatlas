# -*- coding: utf-8 -*-
"""把 figs/*.svg 渲染为 3x PNG 与矢量 PDF（playwright + chromium）

用法：python render_svg.py fig1_1_loc fig4_1_arch ...
不带参数时渲染全部已登记的图。
"""
import os
import sys

from playwright.sync_api import sync_playwright

ROOT = os.path.dirname(os.path.abspath(__file__))
ALL = ["fig1_1_loc", "fig4_1_arch", "fig4_2_vnode", "fig4_3_greedy",
       "fig4_4_dedup"]


def render(name, page, ctx):
    svg_path = os.path.join(ROOT, f"{name}.svg")
    with open(svg_path, encoding="utf-8") as f:
        svg = f.read()
    w = int(svg.split('width="')[1].split('"')[0])
    h = int(svg.split('height="')[1].split('"')[0])
    html = ('<html><head><meta charset="utf-8"><style>html,body{margin:0;'
            'padding:0;background:#fff}</style></head><body>' + svg + '</body></html>')
    pg = ctx.new_page()
    pg.set_viewport_size({"width": w, "height": h})
    pg.set_content(html)
    pg.wait_for_timeout(400)
    pg.screenshot(path=os.path.join(ROOT, f"{name}.png"),
                  clip={"x": 0, "y": 0, "width": w, "height": h})
    pg.pdf(path=os.path.join(ROOT, f"{name}.pdf"), width=f"{w}px", height=f"{h}px",
           print_background=True, page_ranges="1")
    pg.close()
    print(f"rendered {name}  {w}x{h}")


def main():
    names = sys.argv[1:] or ALL
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(device_scale_factor=3)
        for n in names:
            render(n, None, ctx)
        browser.close()


if __name__ == "__main__":
    main()
