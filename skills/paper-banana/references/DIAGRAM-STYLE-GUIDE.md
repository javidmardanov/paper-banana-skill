# Diagram Style Guide: NeurIPS 2025 Aesthetics

Comprehensive style guide for methodology diagrams, derived from PaperBanana's auto-summarized NeurIPS 2025 conventions (arXiv:2601.23265, Appendix F). Apply these rules during the Stylist phase.

---

## Color Strategy

### Zone Strategy (Background Regions)
Use soft pastel backgrounds to group related components:
- **Cream / warm white**: Default canvas, input regions
- **Pale blue**: Processing / computation zones
- **Mint / pale green**: Output regions, results
- **Pale lavender**: Special modules, attention mechanisms
- **Very light gray**: Auxiliary components, optional paths

### Functional Colors
- **Soft blue + soft orange**: Primary pairing for active module vs comparison/baseline
- **Warm tones (soft coral, amber)**: Trainable parameters, learned components
- **Cool tones (steel blue, sage)**: Frozen parameters, fixed components
- **Muted green**: Success states, positive flow
- **Muted red/coral**: Loss signals, error paths (use sparingly)

### Colors to AVOID
- Primary red (#FF0000) — signals error/danger in academic context
- Neon or saturated colors — look unprofessional in print
- Dark fills — reduce legibility of overlaid text
- More than 5-6 distinct colors per diagram — causes visual confusion

### Critical Rule
Always describe colors in natural language: "soft blue", "pale mint green", "warm cream". NEVER use hex codes, RGB values, or CSS color names in image generation prompts. Image generation models render technical color specifications as garbled text in the output image.

---

## Shapes & Containers

### Component Shapes
- **Rounded rectangles**: Default for modules, layers, processing blocks (corner radius ~10-15% of shorter dimension)
- **Rectangles**: Data stores, tables, matrices
- **Circles/ellipses**: Single operations (addition, concatenation, activation functions)
- **Diamonds**: Decision points, conditional branches
- **Cylinders**: Databases, persistent storage
- **Parallelograms**: Input/output
- **Hexagons**: External systems, APIs
- **Trapezoids**: Aggregation/reduction operations

### Softened Geometry
- Round ALL corners slightly — sharp corners look dated
- Use consistent corner radius across all shapes of the same type
- Borders: thin (1-1.5pt), slightly darker than fill color
- Drop shadows: avoid or use very subtle (1-2px offset, low opacity)

### Grouping Containers
- Dashed rounded rectangles for logical groupings
- Very light fill (near-white tint of the zone color)
- Group label in small caps or bold at top-left corner
- Leave padding between group border and contained elements

---

## Lines & Arrows

### Arrow Types
- **Solid with filled head**: Primary data flow, main pipeline
- **Dashed with open head**: Optional paths, skip connections, residual connections
- **Thick flow arrows**: Major data streams (2-3pt width)
- **Thin arrows**: Secondary connections, annotations (0.75-1pt width)
- **Bidirectional**: Mutual information exchange, attention mechanisms

### Routing
- **Orthogonal routing** (right angles) for structured pipelines — cleaner, more professional
- **Curved routing** (bezier) for organic flows, attention patterns, or when orthogonal creates too many crossings
- Avoid diagonal lines unless representing spatial transformations
- Maintain consistent routing style within a diagram

### Arrow Labels
- Place labels on or near the arrow, not far away
- Use smaller font than component labels
- Common labels: tensor dimensions "(B, T, D)", operation names "concat", data type "features"

---

## Typography

### Font Choices
- **Sans-serif only**: Helvetica, Arial, or similar
- Component labels: 10-12pt, regular weight
- Group labels: 9-10pt, bold or small caps
- Arrow labels: 8-9pt, regular weight
- Mathematical notation: italic for variables, regular for operators

### Text Rules
- All text horizontal (never rotated except for Y-axis labels in embedded plots)
- White or very light text on dark backgrounds, dark text on light backgrounds
- Avoid text on arrows — place adjacent instead
- No more than 3-4 words per component label
- Use standard abbreviations: Enc, Dec, Attn, FFN, MLP, Conv, BN, LN

---

## Icons & Visual Elements

### When to Use Icons
- Input modalities: document icon for text, image thumbnail for vision, waveform for audio
- Tools/services: database cylinder, cloud shape, API hexagon
- Operations: plus circle for addition, cross circle for multiplication
- User/agent: simple avatar silhouette

### Icon Style
- Flat design (no 3D effects, gradients, or photorealistic icons)
- Monochrome or duotone matching the zone color
- Consistent size across the diagram
- SVG-quality clean lines

---

## Layout & Composition

### Flow Direction
- **Left-to-right**: Default for pipelines, sequential processing
- **Top-to-bottom**: Good for hierarchical architectures, encoder-decoder
- **Radial**: Agent systems with central coordinator, attention mechanisms

### Spacing
- Consistent spacing between components at the same level
- More space between major sections than between components within a section
- Minimum component spacing: roughly one component-width apart

### Visual Hierarchy
1. **Size**: Larger components = more important
2. **Position**: Center/top = primary, periphery = auxiliary
3. **Color saturation**: More saturated = more important
4. **Border weight**: Thicker borders = key components

### Alignment
- Align component centers, not edges
- Use a grid (implicit or explicit)
- Input and output at diagram edges
- Main processing pipeline along the primary axis

---

## Domain-Specific Styles

### Agent & Reasoning
- Illustrative, slightly playful style
- Rich iconography: thought bubbles, tool icons, persona avatars
- Warm color palette with personality
- Feedback loops prominently styled (thick dashed arrows)
- Decision points as clear diamonds

### Vision & Perception
- Geometric precision, spatial accuracy
- Include mini image thumbnails at input/feature map locations
- Show spatial dimensions explicitly
- Use image-like heatmap overlays for attention
- Feature pyramid levels at different sizes

### Generative & Learning
- Flow-forward emphasis
- Latent space as a compressed bottleneck (narrowing then widening)
- Generation steps shown as progressive refinement
- Use semi-transparent overlays for probabilistic elements
- Clear training vs inference path distinction

### Science & Applications
- Minimalist, restrained decoration
- Domain-appropriate notation (chemical structures, organ diagrams, maps)
- Conservative color palette (blues, grays, one accent)
- Standard scientific diagram conventions for the field
- Let domain content provide visual interest

---

## Common Pitfalls

1. **Too many colors**: Limit to 5-6 distinct colors max. Use shade variations of 2-3 base colors
2. **Hex codes in prompts**: Image gen models render "#E6F3FF" as literal text. Always use natural language
3. **Pixel dimensions**: Don't specify "400x200px". Use relative sizing: "twice as wide as tall"
4. **Overcrowded labels**: If a component needs more than 4 words, it's too detailed for a label — use an annotation
5. **Missing flow direction**: Every pipeline must have a clear start and end
6. **Inconsistent style**: If one component has rounded corners, ALL components should have rounded corners
7. **Dark backgrounds**: White or very light backgrounds only — dark themes don't print well
8. **Figure caption in image**: Captions go in LaTeX, not in the diagram itself
9. **Watermarks or signatures**: Never include any branding in academic figures
