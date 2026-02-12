# Diagram Categories

Four categories for methodology diagram classification, derived from Table 3 of arXiv:2601.23265. Use these to guide the Retriever/Categorize phase and inform style decisions.

---

## 1. Agent & Reasoning

**Definition**: Systems involving autonomous agents, multi-step reasoning, planning, tool use, or decision-making pipelines.

**Keywords**: agent, reasoning, planning, chain-of-thought, tool use, decision tree, reward model, reinforcement learning, policy, state machine, memory, retrieval-augmented generation (RAG), multi-agent, workflow orchestration, feedback loop, reward shaping, RLHF, scaffolding

**Visual conventions**: Illustrative/cartoony style, rich iconography, agent personas, thought bubbles, tool icons, sequential decision nodes, feedback arrows

**Typical structures**: Multi-step pipelines with branching, feedback loops, agent-environment interaction diagrams, tool-calling sequences

---

## 2. Vision & Perception

**Definition**: Systems processing visual data — images, video, 3D, point clouds, or multi-modal inputs involving vision.

**Keywords**: image, video, 3D, point cloud, segmentation, detection, recognition, tracking, depth estimation, optical flow, visual grounding, image generation, diffusion, GAN, NeRF, visual question answering, multi-modal, CLIP, vision transformer, ViT, feature extraction, backbone, FPN, RoI

**Visual conventions**: Spatial/geometric style, image thumbnails as nodes, feature map visualizations, spatial transformations shown explicitly, heavy use of actual image examples

**Typical structures**: Encoder-decoder architectures, feature pyramids, multi-scale processing, attention maps overlaid on images, spatial transformation flows

---

## 3. Generative & Learning

**Definition**: Systems focused on generation (text, code, audio, molecules) or core learning algorithms (self-supervised, contrastive, meta-learning).

**Keywords**: generation, language model, text generation, code generation, diffusion model, VAE, autoregressive, self-supervised, contrastive learning, meta-learning, few-shot, zero-shot, pretraining, fine-tuning, adapter, LoRA, PEFT, knowledge distillation, curriculum learning, active learning, embedding, latent space, tokenizer, decoder, sampling

**Visual conventions**: Flow-oriented style, latent space visualizations, probability distributions, generation pipelines, training vs inference paths distinguished

**Typical structures**: Left-to-right generation flows, encoder-bottleneck-decoder, training pipeline with loss computation, parallel pretraining/fine-tuning branches

---

## 4. Science & Applications

**Definition**: Domain-specific applications in science, engineering, medicine, climate, materials, or other specialized fields.

**Keywords**: molecular, protein, drug discovery, climate, weather, medical imaging, pathology, genomics, materials science, robotics, autonomous driving, satellite, geospatial, energy, simulation, physics-informed, scientific computing, bioinformatics, chemistry, healthcare, EHR, clinical

**Visual conventions**: Minimalist style, domain-appropriate icons and notation, scientific diagrams (molecular structures, organ diagrams, geographic maps), conservative color palettes

**Typical structures**: Domain pipeline (data acquisition → preprocessing → model → domain output), comparison panels, multi-modal fusion with domain-specific inputs
