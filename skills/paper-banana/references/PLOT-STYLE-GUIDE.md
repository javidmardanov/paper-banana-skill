# Plot Style Guide: NeurIPS 2025 Aesthetics

Comprehensive style guide for statistical plots, derived from PaperBanana's auto-summarized NeurIPS 2025 conventions (arXiv:2601.23265, Appendix F). Apply these rules during the Stylist phase in Plot Mode.

---

## Color Palettes

### Categorical Data (Discrete Series)
Use colorblind-friendly palettes. Default to Okabe-Ito derived palette from `assets/palettes/colorblind_safe.json`:
- Series 1: Soft blue (#4477AA → "steel blue")
- Series 2: Soft orange (#EE6677 → "salmon")
- Series 3: Teal (#228833 → "forest green")
- Series 4: Golden (#CCBB44 → "golden yellow")
- Series 5: Purple (#AA3377 → "plum")
- Series 6: Cyan (#66CCEE → "sky blue")

For fewer than 4 series, use the "Soft Pastels" from `assets/palettes/neurips_plots.json`.

### Sequential Data (Continuous Scale)
- **Default**: Viridis (perceptually uniform, colorblind-safe)
- **Warm emphasis**: Magma or Inferno
- **Cool emphasis**: Cividis
- Avoid: Rainbow/jet (perceptually non-uniform), hot (poor grayscale)

### Diverging Data (Centered Scale)
- **Default**: RdBu_r (red-blue, reversed so blue=positive)
- **Alternative**: coolwarm
- Center the colormap at the meaningful zero/reference value

### Color Rules
- Maximum 7 categorical colors per plot (beyond this, use faceting)
- Ensure all colors distinguishable in grayscale (vary lightness, not just hue)
- Use hatching/patterns as secondary encoding for grayscale printing
- Avoid pure white as a data color (indistinguishable from background)

---

## Axes & Grids

### Axis Lines
- Keep left (Y) and bottom (X) spines, remove top and right ("open frame")
- Spine width: 0.8pt
- Color: dark gray (#333333), not pure black
- Tick direction: inward
- Tick length: 4pt major, 2pt minor

### Axis Labels
- Font: sans-serif (Helvetica, Arial, DejaVu Sans)
- Size: 10pt for axis labels, 8pt for tick labels
- X-axis label: centered below axis
- Y-axis label: centered, rotated 90 degrees
- Include units in parentheses: "Accuracy (%)", "Time (seconds)", "Loss (log scale)"

### Grid Lines
- Major grid only (no minor grid unless log scale)
- Style: dashed, 0.5pt width
- Color: light gray (#E0E0E0)
- Behind data (zorder=0)
- Optional for bar charts (horizontal grid only, aids value reading)
- Required for line charts and scatter plots

### Scale
- Linear by default
- Log scale: when data spans >2 orders of magnitude
- When using log scale, label clearly: "Loss (log₁₀)"
- Symmetric log for data spanning positive and negative with large range

---

## Layout & Typography

### Figure Dimensions
- **Single column** (NeurIPS): 3.25" wide (can be up to 5.5" for single-col format)
- **Double column**: 6.875" wide
- **Height**: Typically 60-75% of width (golden ratio ≈ 0.618)
- **Multi-panel**: Use `plt.subplots()`, consistent panel sizes, shared axes where appropriate

### Fonts
- Family: sans-serif
- Title: 12pt, bold (often omitted — caption serves as title)
- Axis labels: 10pt, regular
- Tick labels: 8pt, regular
- Legend: 9pt, regular
- Annotations: 8-9pt, regular or italic

### Legend
- Position: "best" (auto) or upper-right corner
- For many series: outside plot area (right side or below)
- No border frame (frameon=False)
- Slightly transparent background (framealpha=0.8) if overlapping data
- One column unless space-constrained
- Marker + short label, no long descriptions

### Margins
- Use `bbox_inches='tight'` to auto-crop
- `plt.tight_layout()` for multi-panel figures
- Ensure no labels cut off

---

## Plot Type Guidelines

### Bar Charts
- Vertical bars for categories on X-axis (default)
- Horizontal bars when category labels are long
- Gap between groups ≈ 1.5x bar width
- Gap between bars within group ≈ 0.1x bar width
- Add value labels above/right of bars when <10 bars
- Edge color: slightly darker than fill (or thin dark gray line)
- Consider error bars with caps (capsize=3)

### Line Charts
- **Always include markers** — lines alone are hard to distinguish in print
- Distinct markers per series: o (circle), s (square), ^ (triangle up), D (diamond), v (triangle down)
- Line width: 1.5-2.0pt
- Marker size: 6-8pt
- Use dashes/dots for additional line styles in grayscale
- Connect data points only (no interpolation beyond data range)
- Shaded confidence intervals: same color with alpha=0.2

### Scatter Plots
- Marker size: 20-40 (plt.scatter `s` parameter)
- Alpha: 0.7 for overlapping data, 1.0 for sparse data
- Add regression line if showing correlation (dashed, with R² annotation)
- Size encoding for third variable: include size legend
- Edge color: darker shade or gray for definition

### Heatmaps
- Annotate cells with values (fmt='.2f' for floats, 'd' for integers)
- Use diverging colormap centered at meaningful value
- Include colorbar with label
- Font size for annotations: adjust based on cell count (8pt for <10x10, 6pt for larger)
- Square cells (`ax.set_aspect('equal')`) for confusion matrices

### Box Plots & Violin Plots
- Show individual data points overlaid (strip plot) when n < 30
- Box plots: show median, quartiles, whiskers (1.5 IQR), outliers
- Violin plots: show quartile lines inside
- Paired/grouped: side by side with distinct colors
- Horizontal orientation for many categories

### Radar/Spider Charts
- Fill with low alpha (0.2)
- Strong border line (1.5pt)
- Normalize all axes to same scale (0-1 or 0-100)
- Label each axis at the spoke end
- Include grid circles/polygon at regular intervals
- Limit to 5-8 axes (more becomes unreadable)

### Pie/Donut Charts
- Use sparingly — bar charts usually better for comparison
- Maximum 6 slices (group small categories into "Other")
- Start at 12 o'clock, clockwise by decreasing size
- Donut preferred over pie (center for annotations)
- Explode only 1 slice maximum for emphasis
- Include percentage labels (>5% slices only)

---

## Common Pitfalls

1. **3D effects**: Never use 3D bar charts, pie charts, or surface plots unless data is truly 3D. They distort proportions
2. **Dual Y-axes**: Avoid — they make comparison misleading. Use faceted panels instead
3. **Truncated axes**: Always start bar chart Y-axis at zero. Line charts may start at meaningful minimum
4. **Rainbow colormaps**: Jet/rainbow is perceptually non-uniform. Use Viridis, Magma, or Cividis
5. **Missing error bars**: If data has variance, show it (std dev, confidence intervals, or individual points)
6. **Overcrowded tick labels**: Rotate X labels 45° if overlapping. Better: use shorter labels or horizontal bars
7. **plt.show()**: Never include in saved scripts — blocks execution in non-interactive mode
8. **Low DPI**: Always 300 DPI for publication. 150 DPI minimum for review drafts
9. **Colored backgrounds**: White backgrounds only. Gray backgrounds reduce color discrimination
10. **Inconsistent formatting across panels**: Same font sizes, axis styles, and color assignments in multi-panel figures
