# Figure and table standards

These are hard rules. Two of them are enforced mechanically (`tools/figstyle.save`, the
Overfull gate in `tools/check_manuscript.py`); the rest are enforced by looking at the
rendered page before any figure or table is reported as done. They exist because the same
class of defect was corrected by hand three times on 18 September 2026.

## Figures

**F1. No prose inside a figure.** Text inside the artwork is limited to axis labels, tick
labels, legend entries, and series or reference labels of at most three words. A clause, a
colon, a verb, an explanation — anything a reader has to *read* rather than *recognise* —
belongs in the caption. `fs.save()` refuses to write a figure containing a text artist that
breaks this.

**F2. Every symbol is math, and the fonts are the paper's.** Any Greek letter, variable,
or expression is typeset as `$...$` (`$\rho=0.3$`, `$\kappa_4/q^2$`), never as a Unicode
character inside a string. `fs.apply()` sets the text font to the paper body's serif
(Times) and math to Computer Modern via mathtext, which is what the ICML class produces;
a generator never overrides `font.family` or `mathtext.fontset`. If a real LaTeX binary is
ever installed, switch `figstyle` to the PGF backend and the figures will be typeset by
the paper's own compiler; until then this is the closest faithful rendering.

**F3. No label may overlap a curve, a marker, or another label.** The tools for that are
in `figstyle`: `repel` for margin labels (minimum gap computed from the text height and
the panel geometry, not a guessed constant), `label_on_curve` for fans of curves that
will not fit horizontal labels, `halo` as a backstop for a label that must sit near its
own series. A legend goes only where the data are not. Then render the page and look.

**F4. Size for placement.** A single-column figure is generated at `fs.COL_W` (3.25 in)
and placed in `figure`; a full-width figure at `fs.FULL_W` (6.75 in) and placed in
`figure*`. A figure generated at one width and placed at the other is wrong at every
font size it contains.

**F5. One legend per figure when panels share series.** Repeating a legend per panel
costs space and, when every curve starts high on the same side, lands it on the data.

**F6. Marking a point is not prose.** A flagged or excluded datum is marked with a
distinct glyph (hollow circle, caret) and explained in the caption, not annotated with a
sentence inside the panel.

## Tables

**T1. A table fits its environment.** `table` ≤ column width, `table*` ≤ text width. Any
`Overfull \hbox` reported by the build for a table file fails `check_manuscript`.

**T2. Fewer columns before smaller fonts.** Abbreviate headers ("bi-GRU", "TCN",
"U-Net"); move standard errors into macros or a note; split a wide table into two before
shrinking it. `\resizebox` on a table is forbidden — it changes the font size
inconsistently with the surrounding text.

**T3. Tables are generated.** Numbers reach a table only through a generator reading a
frozen output; a table file is never hand-edited.

## Consistency

**C1. Figure text = paper text.** Same font family, same math rendering, the notation of
`paper/notation.tex` (write `$\corr$` in captions; in matplotlib strings write the
underlying `\rho`, since mathtext does not see the paper's macros).

**C2. Render and look.** After any change to a figure or table, rasterise the page it
lands on and inspect it — for collisions, clipping, overflow, and font mismatch — before
reporting the change as done. This step is not optional and is not replaced by a green
build.

## Reports (`reports/`)

The same rules, with `\textwidth` in place of the column: booktabs, `p{}` columns for
any cell that wraps, no table wider than the text block.
