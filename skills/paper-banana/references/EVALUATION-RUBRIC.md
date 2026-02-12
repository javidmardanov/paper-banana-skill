# Evaluation Rubric

Critic agent scoring criteria derived from PaperBanana (arXiv:2601.23265, Section 4.2 + Appendix H). The Critic evaluates generated illustrations on 4 dimensions with hierarchical aggregation.

---

## Hierarchical Aggregation

**Primary dimensions** (must both pass for acceptance):
- Faithfulness
- Readability

**Secondary dimensions** (enhance quality but cannot override primary failures):
- Conciseness
- Aesthetics

**Rule**: A figure with perfect aesthetics but poor faithfulness MUST be revised. Primary > secondary always.

---

## Dimension 1: Faithfulness (Primary)

The generated figure must accurately represent the source methodology.

### Checks
- **Content accuracy**: Every component described in the methodology appears in the figure
- **No hallucinations**: No invented components, connections, or labels that don't exist in the source
- **Relationship fidelity**: Arrows, connections, and data flows match the described architecture
- **Label accuracy**: All text labels match the source exactly (no paraphrasing of technical terms)
- **Caption alignment**: The figure illustrates what the caption claims

### Veto conditions (automatic failure)
- Missing a major component described in the methodology
- Inventing a connection or module not in the source
- Mislabeling a component (e.g., "Encoder" labeled as "Decoder")
- Data values incorrect in plots (wrong numbers, swapped series)

### Scoring guide
- **9-10**: Perfect representation, every detail matches source
- **7-8**: Minor omissions (e.g., missing an optional annotation) but core architecture correct
- **5-6**: Some components missing or connections incorrect
- **1-4**: Major structural errors, misleading representation

---

## Dimension 2: Readability (Primary)

The figure must be immediately understandable without requiring the methodology text.

### Checks
- **Text legibility**: All labels readable at publication print size (single column: 3.25", double column: 6.875")
- **No occlusion**: No overlapping text, arrows, or components
- **Clear flow direction**: Reader can follow the pipeline/architecture without confusion
- **Sufficient contrast**: Text against backgrounds, components against canvas
- **Legend clarity**: If colors/patterns encode meaning, legend is present and clear

### Veto conditions (automatic failure)
- Text too small to read at print size
- Components overlapping making structure ambiguous
- No discernible flow direction in a pipeline diagram
- Key labels completely obscured

### Scoring guide
- **9-10**: Instantly clear, professional journal quality
- **7-8**: Readable with minor effort (e.g., one slightly small label)
- **5-6**: Requires careful study to understand
- **1-4**: Confusing or illegible

---

## Dimension 3: Conciseness (Secondary)

The figure should maximize the signal-to-noise ratio.

### Checks
- **No redundant elements**: Every visual element serves a purpose
- **Appropriate detail level**: Not overly simplified or overly complex
- **White space usage**: Adequate breathing room without wasted space
- **Annotation economy**: Labels are concise, no unnecessary text
- **Visual hierarchy**: Important elements visually prominent, secondary elements subdued

### Scoring guide
- **9-10**: Every pixel earns its place, Tufte-level data-ink ratio
- **7-8**: Minor redundancy (e.g., an unnecessary border or decorative element)
- **5-6**: Noticeable clutter or excessive decoration
- **1-4**: Overwhelming visual noise, hard to identify key information

---

## Dimension 4: Aesthetics (Secondary)

The figure should meet publication-quality visual standards.

### Checks
- **Color harmony**: Palette is cohesive, not jarring
- **Consistent style**: All components share a visual language
- **Professional polish**: No artifacts, jagged edges, or misaligned elements
- **Domain appropriateness**: Style matches the paper's venue and field
- **Typography**: Fonts consistent, sizes proportional, weights appropriate

### Scoring guide
- **9-10**: Best-paper award quality, visually striking
- **7-8**: Professional, publication-ready
- **5-6**: Acceptable but unremarkable
- **1-4**: Unprofessional or visually jarring

---

## Output Format

The Critic must produce a structured JSON assessment:

```json
{
  "scores": {
    "faithfulness": <1-10>,
    "readability": <1-10>,
    "conciseness": <1-10>,
    "aesthetics": <1-10>
  },
  "primary_pass": <true if faithfulness >= 7 AND readability >= 7>,
  "overall_pass": <true if primary_pass AND conciseness >= 5 AND aesthetics >= 5>,
  "critic_suggestions": [
    "Specific actionable suggestion 1",
    "Specific actionable suggestion 2"
  ],
  "revised_description": "<improved description if revision needed, null if acceptable>"
}
```

### Revision trigger
- If `primary_pass` is false: revision is **mandatory**
- If `primary_pass` is true but `overall_pass` is false: revision is **recommended**
- If `overall_pass` is true: no revision needed

### Revision rules
- Maximum 3 revision iterations
- Each revision must address ALL listed suggestions
- The revised description must be a complete, standalone description (not a diff)
- After 3 iterations, accept the best version even if imperfect

---

## Plot-Specific Checks

When evaluating statistical plots (Plot Mode), add these checks:

### Data Fidelity
- Every data point correctly positioned
- Axis scales correct (linear vs log, range matches data)
- No data points missing or duplicated
- Series/categories correctly colored and labeled

### Code Correctness
- Python syntax valid
- All imports from approved set (matplotlib, seaborn, numpy, pandas)
- Code includes `savefig` call
- No `plt.show()` (would block in non-interactive mode)
- Self-contained (no external file dependencies unless user-provided)

### Layout
- No overlapping tick labels
- Legend does not obscure data
- Title and axis labels fully visible
- Adequate margins
