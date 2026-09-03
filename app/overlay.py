"""Build the overlay HTML served to OBS.

Sizes and fonts match the scoreboard generator overlay (.ovBox): 207 px wide,
21 px Anton title, 25 px rows, 24 px rank column, 42 px points column.

The static page carries a short meta refresh; the live page pulls state instead.
"""

from __future__ import annotations

PROTOCOL = 2  # module interface version; mixing different values is unsafe
FILE_SET = "2026-09-04-k"  # release this file belongs to

from settings import HOLD_MS, MOVE_MS, OVERLAY_WIDTH as WIDTH
REFRESH_SECONDS = 1.0

STYLE = """
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.css');
@font-face{font-family:'Anton';src:url('https://cdn.jsdelivr.net/npm/@fontsource/anton@5.0.19/files/anton-latin-400-normal.woff2') format('woff2');font-weight:400;font-display:swap}
@font-face{font-family:'Rajdhani';src:url('https://cdn.jsdelivr.net/npm/@fontsource/rajdhani@5.0.18/files/rajdhani-latin-700-normal.woff2') format('woff2');font-weight:700;font-display:swap}
@font-face{font-family:'Oswald';src:url('https://cdn.jsdelivr.net/npm/@fontsource/oswald@5.0.18/files/oswald-latin-600-normal.woff2') format('woff2');font-weight:600;font-display:swap}
html,body{margin:0;background:transparent}
.ovScaler{position:absolute;top:0;left:0;transform-origin:top left}
.ovBox{position:relative;display:block;background:#0b1016;
  border:1px solid #ffffff2e;border-radius:5px;padding:4px 0;color:#fff;
  font-family:'Rajdhani','Pretendard',sans-serif}
.ovBox .ovt{font-family:'Anton','Pretendard',sans-serif;font-size:21px;letter-spacing:1px;line-height:1;
  text-align:center;padding:1px 0 2px;margin-bottom:2px;border-bottom:1px solid #ffffff2e}
.ovBox .ovr{display:flex;align-items:stretch;height:25px;font-weight:700;font-size:15px}
.ovBox .ovr:not(:last-child){border-bottom:1px solid #ffffff24}
.ovBox .ovr .rk{width:24px;display:flex;align-items:center;justify-content:center;
  font-family:'Oswald','Rajdhani',sans-serif;font-weight:600;border-right:1px solid #ffffff24}
.ovBox .ovr .nm{flex:1;display:flex;align-items:center;justify-content:center;padding:0 4px;
  white-space:nowrap;overflow:hidden}
.ovBox .ovr .pt{width:42px;display:flex;align-items:center;justify-content:center;
  border-left:1px solid #ffffff24;font-family:'Oswald','Rajdhani',sans-serif;font-weight:600}
"""


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _number(value: float) -> str:
    """Format a score, dropping a trailing zero decimal."""
    rounded = round(float(value), 1)
    return str(int(rounded)) if rounded == int(rounded) else str(rounded)


def rows_html(
    standings,
    highlight: int = 0,
    rank_color: str = "#ffffff",
    highlight_color: str = "#d0cb3e",
    fade: bool = True,
) -> str:
    """Render one row per team.

    Args:
        standings: TeamStanding list, already ordered.
        highlight: how many top rows get the highlight background.
    """
    out = []
    for index, team in enumerate(standings):
        rank = index + 1
        style = ""
        if highlight and rank <= highlight:
            r = int(highlight_color[1:3], 16)
            g = int(highlight_color[3:5], 16)
            b = int(highlight_color[5:7], 16)
            style = (
                f"background:linear-gradient(to right,rgba({r},{g},{b},.62),rgba({r},{g},{b},0))"
                if fade
                else f"background:rgba({r},{g},{b},.5)"
            )
        out.append(
            f'<div class="ovr" style="{style}">'
            f'<span class="rk" style="color:{rank_color}">#{rank}</span>'
            f'<span class="nm">{_escape(team.name)}</span>'
            f'<span class="pt">{_number(team.total)}</span>'
            f"</div>"
        )
    return "\n".join(out)


def build_html(
    standings,
    title: str = "LEADERBOARD",
    width: int = WIDTH,
    scale: float = 1.0,
    highlight: int = 0,
    refresh: float = REFRESH_SECONDS,
) -> str:
    """Return a complete static overlay page."""
    body = (
        f'<div class="ovScaler" style="transform:scale({scale})">'
        f'<div class="ovBox" style="width:{width}px">'
        f'<div class="ovt">{_escape(title)}</div>'
        f"{rows_html(standings, highlight)}"
        f"</div></div>"
    )
    refresh_tag = (
        f'<meta http-equiv="refresh" content="{refresh}">' if refresh > 0 else ""
    )
    return (
        "<!DOCTYPE html><html lang='ko'><head><meta charset='utf-8'>"
        f"{refresh_tag}<style>{STYLE}</style></head><body>{body}</body></html>"
    )


def write_html(path, standings, **kwargs) -> None:
    """Write a static overlay page to disk."""
    from pathlib import Path

    Path(path).write_text(build_html(standings, **kwargs), encoding="utf-8")


LIVE_SCRIPT = """
const box = document.getElementById('rows');
let seen = new Map();      // team name -> previous rank and total
let version = -1;

const MOVE_MS = __MOVE_MS__;       // row travel time
const HOLD_MS = __HOLD_MS__;       // extra hold after the move ends
const FADE_IN = 0.08;      // colour reaches full at this fraction

function numText(v){
  const n = Math.round(Number(v) * 10) / 10;
  return String(n);
}

// Driving the highlight with setTimeout drifts from the move when the browser
// is busy, so the highlight runs on the same Web Animations clock and ends with
// the animation.
const running = new WeakMap();   // highlight currently running per cell

function flash(el, color, ms){
  const previous = running.get(el);
  if (previous) previous.cancel();       // cancel the previous highlight first
  const anim = el.animate(
    [
      {background:'rgba(0,0,0,0)', offset:0},
      {background:color, offset:FADE_IN},
      {background:color, offset:1},
    ],
    {duration:ms, easing:'linear', fill:'none'}
  );
  running.set(el, anim);
  return anim;
}

function rowFor(name){
  // Rows are looked up by name instead of being rebuilt: rebuilding kills the
  // running move and highlight, so overlapping updates would cut them short.
  let el = box.querySelector(`[data-name="${CSS.escape(name)}"]`);
  if (!el) {
    el = document.createElement('div');
    el.className = 'ovr';
    el.dataset.name = name;
    el.innerHTML = '<span class="rk"></span><span class="nm"></span><span class="pt"></span>';
    el.querySelector('.nm').textContent = name;
  }
  return el;
}

function draw(state){
  const before = new Map();
  for (const el of box.children) before.set(el.dataset.name, el.getBoundingClientRect().top);

  // Append the wanted rows in order; existing rows are only moved.
  const wanted = state.rows.map(t => rowFor(t.name));
  wanted.forEach((el, i) => {
    const row = state.rows[i];
    el.querySelector('.rk').textContent = '#' + (i + 1);
    el.querySelector('.pt').textContent = numText(row.total);
    el.classList.toggle('out', !!row.out);
    box.appendChild(el);
  });
  for (const el of [...box.children]) {
    if (!wanted.includes(el)) el.remove();
  }

  wanted.forEach((el, index) => {
    const name = el.dataset.name;
    const was = seen.get(name);
    const now = {rank: index + 1, total: Number(state.rows[index].total)};

    const old = before.get(name);
    const move = old === undefined ? 0 : old - el.getBoundingClientRect().top;
    let mover = null;
    if (move) {
      mover = el.animate(
        [{transform:`translateY(${move}px)`}, {transform:'translateY(0)'}],
        {duration:MOVE_MS, easing:'cubic-bezier(.25,.1,.25,1)'}
      );
    }

    if (was && was.rank !== now.rank && mover) {
      flash(el, 'rgba(208,203,62,.20)', MOVE_MS + HOLD_MS);
    } else if (was && was.total !== now.total) {
      flash(el.querySelector('.pt'), 'rgba(208,203,62,.28)', MOVE_MS);
    }
  });

  seen = new Map(state.rows.map((t, i) => [t.name, {rank: i + 1, total: Number(t.total)}]));
}

async function tick(){
  try {
    const res = await fetch('/state.json', {cache:'no-store'});
    const state = await res.json();
    if (state.version !== version) { version = state.version; draw(state); }
  } catch (e) { /* ignore a missing server; try again next tick */ }
}
setInterval(tick, 300);
tick();
"""

LIVE_STYLE = """
/* Teams eliminated this round: dimmed text and a line across the row to show
   that the value will not change again. */
.ovBox .ovr{position:relative}
.ovBox .ovr.out{color:#8b95a2}
.ovBox .ovr.out .rk{color:#8b95a2}
.ovBox .ovr.out::after{content:'';position:absolute;left:0;right:0;top:0;bottom:0;
  pointer-events:none;
  background:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100' preserveAspectRatio='none'%3E%3Cline x1='0' y1='100' x2='100' y2='0' stroke='%23e6e6eb' stroke-opacity='0.45' stroke-폭='2' vector-effect='non-scaling-stroke'/%3E%3C/svg%3E") 0 0/100% 100% no-repeat}
"""


def _script() -> str:
    """Fill timing values from settings into the page script."""
    return LIVE_SCRIPT.replace("__MOVE_MS__", str(MOVE_MS)).replace(
        "__HOLD_MS__", str(HOLD_MS)
    )


def live_page(title: str = "LEADERBOARD", width: int = WIDTH, scale: float = 1.0) -> str:
    """Return the live overlay page served over HTTP.

    Rows are patched in place, so moves can animate and changed rows can flash.
    """
    return (
        "<!DOCTYPE html><html lang='ko'><head><meta charset='utf-8'>"
        f"<style>{STYLE}{LIVE_STYLE}</style></head><body>"
        f"<div class='ovScaler' style='transform:scale({scale})'>"
        f"<div class='ovBox' style='width:{width}px'>"
        f"<div class='ovt'>{_escape(title)}</div>"
        "<div id='rows'></div><div id='dead'></div>"
        "</div></div>"
        f"<script>{_script()}</script></body></html>"
    )
