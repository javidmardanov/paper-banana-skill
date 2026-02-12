# Diagram Mode Agent Prompts

Actual Gemini API prompt templates for the 5-agent pipeline in Diagram Mode, faithfully implementing PaperBanana (arXiv:2601.23265, Appendix G.1). Each agent is a separate Gemini API call with specific prompts and multimodal inputs.

**Implementation**: Each prompt below is sent to the Gemini API by its corresponding Python script in `scripts/`. The orchestrator (`scripts/orchestrate.py`) chains them sequentially.

---

## Phase 1: Retriever (`scripts/retriever.py`)

**Model**: `gemini-2.0-flash`
**Input**: Methodology text (text-only)
**Output**: JSON with category, visual intent, and 2 selected reference IDs

The Retriever loads the 13 curated reference diagrams from `assets/references/index.json`, formats them as a numbered candidate list, and asks Gemini to select the 2 most relevant references.

**Prompt template** (sent to Gemini):

```
You are the Retriever agent in the PaperBanana academic illustration pipeline.

Your task:
1. Read the user's methodology text below.
2. Classify it into one of these 4 diagram categories:
   - Agent & Reasoning
   - Vision & Perception
   - Generative & Learning
   - Science & Applications
3. Identify the visual intent (Framework Overview, Pipeline/Flow, Detailed Module, or Architecture Diagram).
4. From the numbered reference candidates below, select the 2 most relevant examples that would best guide generating a methodology diagram for this text.

{categories_text from DIAGRAM-CATEGORIES.md}

--- REFERENCE CANDIDATES ---
{formatted list from assets/references/index.json}

--- USER METHODOLOGY TEXT ---
{user's methodology text}

--- OUTPUT FORMAT ---
Respond with ONLY valid JSON:
{
  "category": "<one of the 4 categories>",
  "visual_intent": "<Framework Overview | Pipeline/Flow | Detailed Module | Architecture Diagram>",
  "domain_signals": ["keyword1", "keyword2", "keyword3"],
  "selected_references": [
    {"id": "<reference_id>", "reason": "<why this reference is relevant>"},
    {"id": "<reference_id>", "reason": "<why this reference is relevant>"}
  ]
}
```

**CLI usage**:
```bash
python scripts/retriever.py \
  --methodology "source text..." \
  --mode diagram \
  --output retriever_output.json
```

---

## Phase 2: Planner (`scripts/planner.py`)

**Model**: `gemini-2.0-flash`
**Input**: Multimodal — 2 reference images (PNG) + methodology text + caption
**Output**: Detailed textual description of the target diagram

This is the paper's core innovation: **multimodal in-context learning**. The Planner "sees" the 2 reference images selected by the Retriever and generates a description that would produce a diagram of similar quality.

**Prompt template** (sent to Gemini with reference images as image parts):

```
[IMAGE: reference_1.png]
Reference example (agent_reasoning_01): Multi-agent planning framework...

[IMAGE: reference_2.png]
Reference example (science_applications_02): Climate prediction framework...

You are the Planner agent in the PaperBanana academic illustration pipeline.

Your task: Convert the methodology text and figure caption below into an extremely detailed textual description of a methodology diagram. This description will be fed directly to an image generation model.

The reference images provided above show examples of high-quality NeurIPS 2025 methodology diagrams. Use them as visual guides for layout, style, and detail level.

Category: {category from Retriever}
Visual Intent: {visual_intent from Retriever}

--- CRITICAL RULES ---
1. Be MAXIMALLY specific. Vague specifications produce worse figures.
2. Use ONLY natural language for visual attributes. NEVER use hex codes, RGB values, or pixel dimensions.
3. Every component must have: a name, a shape description, a relative position, and a relative size.
4. Every connection must have: a source, a target, a type, and optionally a label.

--- YOUR DESCRIPTION MUST SPECIFY ---
Layout, Components, Connections, Groupings, Annotations (see full spec in script)

--- FIGURE CAPTION ---
{user's caption}

--- METHODOLOGY TEXT ---
{user's methodology text}

--- OUTPUT ---
Write a single, complete textual description as flowing descriptive prose.
```

**CLI usage**:
```bash
python scripts/planner.py \
  --methodology "source text..." \
  --caption "Figure 1: Overview of..." \
  --references retriever_output.json \
  --output planner_output.json
```

---

## Phase 3: Stylist (`scripts/stylist.py`)

**Model**: `gemini-2.0-flash`
**Input**: Planner's description + category + full style guide text
**Output**: Polished, styled description

The Stylist applies NeurIPS 2025 aesthetic conventions from `references/DIAGRAM-STYLE-GUIDE.md` to the Planner's raw description.

**Prompt template** (sent to Gemini):

```
You are the Stylist agent in the PaperBanana academic illustration pipeline.

Your task: Refine the Planner's diagram description below to ensure it meets NeurIPS 2025 publication aesthetics.

--- FIVE CRITICAL RULES ---
1. PRESERVE high-quality aesthetics
2. INTERVENE MINIMALLY
3. RESPECT DOMAIN DIVERSITY
4. ENRICH VAGUE DETAILS
5. PRESERVE CONTENT

--- STYLE APPLICATIONS BY CATEGORY ---
- Agent & Reasoning: persona icons, warm illustrated style
- Vision & Perception: spatial relationships, geometric precision
- Generative & Learning: flow emphasis, gradient fills for latent spaces
- Science & Applications: conservative, domain-appropriate

--- COLOR GUIDANCE ---
"Soft Tech & Scientific Pastels" palette. Natural language only, NEVER hex codes.

--- NEURIPS 2025 STYLE GUIDE ---
{full contents of DIAGRAM-STYLE-GUIDE.md}

--- DIAGRAM CATEGORY ---
{category}

--- PLANNER'S DESCRIPTION ---
{description from Planner}

--- OUTPUT ---
Output the complete polished description ONLY. No explanations or commentary.
```

**CLI usage**:
```bash
python scripts/stylist.py \
  --description planner_output.json \
  --output stylist_output.json
```

---

## Phase 4: Visualizer (`scripts/generate_image.py`)

**Model**: `gemini-3-pro-image-preview` (Nano-Banana-Pro)
**Input**: Styled description text
**Output**: Generated PNG image

The Visualizer calls Google's image generation API. The existing `scripts/generate_image.py` handles this phase. The orchestrator prepends a quality prefix to the styled description.

**Quality prefix** (prepended automatically):
```
High-resolution academic illustration for a top-tier ML conference paper.
Clean white or very light background.
All text must be perfectly legible in clear sans-serif font.
Professional publication quality.
No watermarks, signatures, or decorative borders.
No figure number or caption text within the image.
```

**Aspect ratio selection** (by visual intent):
- Pipeline/Flow → 16:9
- Framework Overview → 16:9
- Detailed Module → 3:2
- Architecture Diagram → 3:2

**CLI usage**:
```bash
python scripts/generate_image.py \
  --prompt-file styled_description.txt \
  --output output/diagram.png \
  --aspect-ratio 16:9
```

---

## Phase 5: Critic (`scripts/critic.py`)

**Model**: `gemini-2.0-flash`
**Input**: Multimodal — generated image (PNG) + methodology text + styled description + rubric
**Output**: JSON with 4-dimension scores, pass/fail, suggestions, optional revised description

The Critic evaluates the generated image by "seeing" it via Gemini's VLM capabilities and comparing against the source text.

**Prompt template** (sent to Gemini with generated image as image part):

```
[IMAGE: generated_diagram.png]

You are the Critic agent in the PaperBanana academic illustration pipeline.

Your task: Evaluate the generated methodology diagram image against the original methodology text and styled description.

--- EVALUATION RUBRIC ---
{full contents of EVALUATION-RUBRIC.md}

--- SCORING DIMENSIONS ---
1. FAITHFULNESS (Primary, must >= 7)
2. READABILITY (Primary, must >= 7)
3. CONCISENESS (Secondary)
4. AESTHETICS (Secondary)

--- ORIGINAL METHODOLOGY TEXT ---
{methodology}

--- STYLED DESCRIPTION ---
{styled_description}

--- OUTPUT FORMAT ---
{
  "scores": {"faithfulness": N, "readability": N, "conciseness": N, "aesthetics": N},
  "primary_pass": true/false,
  "overall_pass": true/false,
  "critic_suggestions": ["..."],
  "revised_description": "..." or null
}
```

**Refinement loop**: If `primary_pass` is false, the Critic generates a `revised_description` and the orchestrator loops back to Phase 4 (Visualizer). Maximum 3 iterations.

**CLI usage**:
```bash
python scripts/critic.py \
  --image output/diagram.png \
  --methodology "source text..." \
  --description stylist_output.json \
  --output critic_output.json
```

---

## End-to-End Orchestration (`scripts/orchestrate.py`)

The orchestrator chains all 5 agents and handles the refinement loop:

```
Retriever → Planner → Stylist → Visualizer ⇄ Critic (max 3 iterations)
```

**CLI usage**:
```bash
python scripts/orchestrate.py \
  --methodology-file methodology.txt \
  --caption "Figure 1: Overview of proposed framework" \
  --mode diagram \
  --output output/diagram.png
```

All intermediate outputs (retriever_output.json, planner_output.json, etc.) are saved to `output/work/` for inspection and debugging.
