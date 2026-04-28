---
name: tikz-workflow
description: Create publication-quality TikZ workflow/process diagrams for academic papers. Use when user asks for flowcharts, workflow diagrams, process charts, development timelines, or pipeline diagrams in TikZ/LaTeX.
---

# Publication-Quality TikZ Workflow Diagrams

Generate clean, well-aligned workflow and process diagrams using standalone TikZ for inclusion in academic papers.

## How to Use

When invoked:
1. **Understand** the workflow structure (phases, steps, groupings)
2. **Write** a standalone TikZ `.tex` file
3. **Compile** with `pdflatex` and **visually inspect** the output
4. **Iterate** until no overlaps, no hyphenation, uniform alignment

## Core Architecture: The Rigid Grid

**Never use `fit` for outer containers.** It adapts to content and produces uneven widths. Instead, define a fixed coordinate grid and draw rectangles explicitly.

```latex
% Fixed outer box edges — everything derives from these
\def\boxL{-7.0}
\def\boxR{11.0}
\def\boxmid{2.0}  % (\boxL+\boxR)/2

% Draw gray background boxes with exact edges
\newcommand{\graybox}[2]{%
    \begin{pgfonlayer}{background}
        \fill[phasebg, rounded corners=6pt] (\boxL, #1) rectangle (\boxR, #2);
    \end{pgfonlayer}
}

% Column positions — evenly spaced, centered on \boxmid
% 4-col: spacing=4.0, positions = boxmid + {-6, -2, +2, +6}/2... just compute them
\def\colI{-4.0}   \def\colII{0.0}
\def\colIII{4.0}  \def\colIV{8.0}
```

**Why:** Using `fit` with `minimum width` still shifts boxes when content varies. Explicit rectangles guarantee pixel-perfect alignment across all rows.

## Critical Rules

### 1. Prevent text overflow and hyphenation

**This is the #1 source of ugly diagrams.** At every font size, text wraps differently.

- After EVERY compile, **read the PDF** and check for hyphenated words ("Kom-\npaneets", "sub-\nprocess")
- If a word hyphenates, **rewrite the text** to avoid it — don't fight LaTeX's line-breaking
- Use `\\` for explicit line breaks; aim for exactly 3 lines per box
- Keep each line under ~30 characters at `\scriptsize` in a 3.0cm box
- Test problematic words: long technical terms (Kompaneets, bremsstrahlung, pentadiagonal) often hyphenate

```latex
% BAD: will hyphenate "Kompaneets" at box boundary
{confirmed Kompaneets approach\\is sufficient for $z < 10^7$}

% GOOD: restructured to avoid the break
{Reviewed Compton scattering\\literature; FP approach is\\sufficient for $z < 10^7$}
```

### 2. Use background layers for container boxes

```latex
\pgfdeclarelayer{background}
\pgfsetlayers{background,main}

% Container boxes go on background layer — task boxes on main
\begin{pgfonlayer}{background}
    \fill[phasebg, rounded corners=6pt] (\boxL, -0.5) rectangle (\boxR, -4.0);
\end{pgfonlayer}
```

**Why:** Without this, gray boxes drawn after task nodes will cover them. This was the original showstopper bug.

### 3. Keep arrows in empty space

Never place labels (phase titles, annotations) in the gap between boxes where arrows run. Either:
- Put phase titles **inside** the gray container box (as a header row)
- Or put them **right-aligned** on the same line as the session label

```latex
% Phase title right-aligned inside the box header
\node[font=\sffamily\footnotesize\bfseries, text=phasecolor!70, anchor=east]
    at (\boxR-0.4, \headerY) {Physics debugging};
```

### 4. Uniform box sizing

- Set `minimum height` on task boxes so all boxes in a row are the same height
- Use the same `text width` for all boxes in a row
- For 3-col rows that need wider boxes, override `text width` explicitly

```latex
task/.style={
    draw=#1!50!black, fill=#1!8, rounded corners=2pt,
    align=center, font=\sffamily\scriptsize, line width=0.35pt,
    inner xsep=8pt, inner ysep=6pt, text width=3.4cm,
    minimum height=1.2cm
},

% Override for 3-col row
\node[task=research, text width=4.2cm, minimum height=1.3cm] at ...
```

### 5. Session grouping with separators

Group related phases (e.g., exploration + debugging) in a single gray box with a thin separator line:

```latex
% Thin separator between sub-phases within one session
\draw[black!12, line width=0.4pt] (\boxL+0.5, -7.1) -- (\boxR-0.5, -7.1);
```

### 6. Legend alignment

Use uniform `minimum width` for legend boxes to prevent uneven spacing:

```latex
\foreach \col/\lab/\idx in {
    research/{Literature \& web research}/0,
    implement/{Physics implementation}/1,
    ...
}{
    \pgfmathsetmacro{\lx}{\legStart + \idx*(\legW + \legGap)}
    \node[fill=\col!10, draw=\col!50!black, rounded corners=2pt,
          font=\sffamily\scriptsize, minimum width=\legW cm,
          minimum height=0.5cm] at (\lx+\legW/2, \yleg) {\lab};
}
```

## Color Palette

Colorblind-friendly (Tol Vibrant), with low-saturation fills:

```latex
\definecolor{research}{HTML}{4477AA}    % blue
\definecolor{implement}{HTML}{228833}   % green
\definecolor{debug}{HTML}{EE6677}       % red/pink
\definecolor{validate}{HTML}{CCBB44}    % yellow
\definecolor{infra}{HTML}{AA3377}       % purple
\definecolor{phasebg}{HTML}{F0F0F0}     % container background
\definecolor{timecolor}{HTML}{999999}   % secondary text
\definecolor{titlecolor}{HTML}{555555}  % header text

% Task boxes: draw at 50% black mix, fill at 8% saturation
task/.style={draw=#1!50!black, fill=#1!8, ...}
```

## Typography Hierarchy

| Element | Font | Size | Color |
|---------|------|------|-------|
| Session label | `\sffamily\footnotesize\bfseries` | ~10pt | `titlecolor` |
| Duration | `\sffamily\footnotesize` | ~10pt | `timecolor` |
| Phase title | `\sffamily\footnotesize\bfseries` | ~10pt | `phasecolor!70` |
| Task box text | `\sffamily\scriptsize` | ~8pt | black |
| Legend | `\sffamily\scriptsize` | ~8pt | black |
| Final output | `\sffamily\bfseries` | ~12pt | black |

## Vertical Spacing Template

```
Phase title + session label (header row)
    |  0.9 cm gap
Task boxes (centered vertically)
    |  bottom of gray box
    |  0.7 cm arrow gap
Next gray box top
```

For two-row sessions (e.g., exploration + debugging):
```
Header row 1
    |  0.95 cm
Task row 1
    |  separator line (midpoint between rows)
Header row 2 (time label only, no "Session N")
    |  0.95 cm
Task row 2
    |  bottom of gray box
```

## Compilation Workflow

```bash
cd /path/to/figures
pdflatex -interaction=nonstopmode diagram.tex
# Then READ the PDF to visually inspect before declaring done
```

## Checklist Before Declaring Done

- [ ] All gray container boxes are the same width
- [ ] All task boxes in each row are the same height
- [ ] No hyphenated words anywhere
- [ ] Arrows don't overlap with any text
- [ ] Legend boxes are uniform width and evenly spaced
- [ ] Session labels are left-aligned at the same x coordinate
- [ ] Phase titles are right-aligned at the same x coordinate
- [ ] No text touches the edge of its containing box
- [ ] Colors are distinguishable (colorblind-safe)
- [ ] Compiled as PDF (vector graphics, not rasterized)

## Common Pitfalls

1. **`fit` produces uneven boxes** — Use explicit `\fill` rectangles instead
2. **Phase titles between boxes overlap arrows** — Put titles inside boxes
3. **`\scriptsize` text overflows `3.0cm` boxes** — Either widen or use `\tiny`
4. **Legend items have different widths** — Use `minimum width` on all
5. **Forgetting background layer** — Gray boxes cover task boxes
6. **Not inspecting PDF after compile** — Text wrapping is unpredictable; always check
7. **Math mode text is wider** — `$O(\theta_e)$` takes more space than plain text; account for it
