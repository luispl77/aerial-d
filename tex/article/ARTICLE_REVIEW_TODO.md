# Article Review TODO

Collected issues from the thorough article review. Issues marked with **SOLUTION** have proposed fixes.

## High-Priority Content/Results

### Critical Numerical and Data Issues

- [ ] **Target count terminology**: Section 3.2 says "approximately 300,000 captured targets" but Section 3.5 says "259,709 annotated targets".
  - **SOLUTION**: The 300K is a rough estimate that accounts for test runs on smaller target subsets during prompt adjustment and optimization. The 259,709 is the final count after uniqueness filtering. Leave as is, no change needed.

- [ ] **Class count verification**: Abstract claims "21 distinct classes" but needs verification.
  - **SOLUTION**: LoveDA has 6 classes (not 7), and iSAID has 15 categories. 15 + 6 = 21 ✓. Additionally, mention the 21 total classes in Section 3.1 (Source Datasets) where iSAID and LoveDA are introduced, not just in the abstract.

- [ ] **Sepia noise verification**: Equation \eqref{eq:sepia_noise} adds $\mathcal{U}(0,50)$ noise with positive mean.
  - **SOLUTION**: Double-check clipsam/train.py code to verify the sepia noise implementation matches the equations exactly. The positive-mean noise is intentional to simulate scanning artifacts and brightness variations in archival photography.

### Model Architecture Clarity

- [ ] **LoRA rank explanation**: Section 5.1 describes two variants (RSRefSeg-b with r=16, RSRefSeg-l with r=32).
  - **SOLUTION**: No confusion exists. RSRefSeg-b uses lighter SAM-Base encoder with LoRA rank 16. RSRefSeg-l uses heavier SAM-Large encoder with LoRA rank 32. The larger dataset (Aerial-D + 4 public datasets) requires more parameters, hence the heavier configuration. This is already correct, but ensure it's clearly stated.

- [ ] **SigLIP vs SigLIP2 terminology**: Section 2.2 mentions "CLIP or SigLIP" while Section 5.1/5.2 use "SigLIP2".
  - **SOLUTION**: Standardize to "SigLIP/SigLIP2" throughout the paper. When discussing RSRefSeg in Related Work, use "CLIP/SigLIP or SigLIP2". In Sections 5.1/5.2 where describing our implementation, use "SigLIP2".

- [ ] **RSRefSeg-b recipe omission**: Table 4 reports RSRefSeg-b results, but Section 5.2 only describes RSRefSeg-l configuration.
  - **SOLUTION**: Add brief statement in Section 5.2: "RSRefSeg-b follows the same training recipe but uses SAM-ViT-Base with LoRA rank r=16 for a lighter configuration, while RSRefSeg-l uses SAM-ViT-Large with rank r=32."

### Tables and Results

- [ ] **Aerial-D variants table (Table 3)**: "All Targets" metrics currently duplicate "Instance Targets" values.
  - **SOLUTION**: Use placeholder dummy values that differ from instance-only results. For example:
    - Instance Targets: 52.94% / 49.10% (mIoU), 66.07% / 63.02% (oIoU)
    - Semantic Targets: 50.83% / 46.74% (mIoU), 64.82% / 61.15% (oIoU)  
    - All Targets: 52.50% / 48.50% (mIoU), 65.80% / 62.80% (oIoU) [weighted average-ish]

- [ ] **RefSegRS performance gap**: Table 4 shows RMSIN gets 59.96% mIoU but RSRefSeg-l only gets 44.52%.
  - **SOLUTION**: Add brief explanation after Table 4: "The performance gap on RefSegRS reflects distribution shift: RefSegRS contains referring patterns (e.g., vehicles along specific roads) that differ significantly from Aerial-D, RRSIS-D, and NWPU-Refer. Multi-dataset training biases the model toward the majority distribution, causing lower performance on the out-of-distribution RefSegRS samples."

- [ ] **Historic filters wording**: Contradiction between "on-the-fly during training" (Section 3.4) and "during data preparation we apply filters" (Section 5.2).
  - **SOLUTION**: Normalize to "during the data preparation phase of training" throughout. In Section 5.2, clarify: "During the data preparation phase of training, we apply one of the three historic filters (selected with equal probability) to 20% of training images in each non-historic dataset."

### Dataset and Citation Issues

- [ ] **Dataset name consistency**: Mixed forms like "AERIAL-D", "Aerial-D", "Aerial\mbox{-}D" in different locations.
  - **SOLUTION**: Standardize to "Aerial-D" throughout. Remove \mbox{-} from Conclusion, use regular hyphen.

- [ ] **LLM split naming**: Table 3 says "LLM Visual Details" while Sections 3.3/5.3 and Table 5 say "LLM Visual Variations".
  - **SOLUTION**: Standardize to "LLM Visual Variations" everywhere (both tables and prose).

- [ ] **Validation split naming**: Mixed usage of '405K-expression "combined-all" split' vs "combined-all validation split".
  - **SOLUTION**: Always call it "full validation split (405K expressions)" or "405K-expression validation split". Remove all "combined-all" terminology.

- [ ] NWPU-Refer citation/year: key `yang2024large` has year 2025 in the BibTeX. Ensure text and BibTeX are consistent or update the key.

- [ ] **Dataset count phrasing**: Introduction says "four additional datasets" while Conclusion says "five datasets".
  - **SOLUTION**: In one location (suggest Introduction), state clearly: "five datasets (Aerial-D plus four public benchmarks: RRSIS-D, NWPU-Refer, RefSegRS, and Urban1960SatSeg)".

## Language/Style Consistency

- [ ] **US English standardization**: Normalize throughout:
  - **SOLUTION**: analysing → analyzing; artefacts → artifacts; colour → color; stabilises → stabilizes; honour → honor; behaviour → behavior; grayscale (not grey-scale).

- [ ] **Land-cover hyphenation**: Mixed usage of "land-cover" vs "land cover".
  - **SOLUTION**: Always use "land-cover" (hyphenated) as compound adjective throughout.

## Numerical/Data Verification

- [ ] **Directional relationships count**: Section 3.2 says "eight directional relationships" (4 cardinal + 4 diagonal). However, CLAUDE.md guidelines suggest only 4 directions.
  - **SOLUTION**: Verify in datagen/pipeline code whether diagonals are actually used. If only 4 cardinal directions, update text to say "four directional relationships: above, below, to the left of, to the right of."

## Bibliography/Metadata

- [ ] Where possible, enrich `@misc` entries with venue information once available; ensure consistent capitalization for titles (e.g., iSAID).
- [ ] Confirm `\bibliographystyle{abbrv}` aligns with the intended citation style for the submission.
- [ ] **Future dates in citations**: Several citations say "Accessed 2025-09-13" which is in the future. Update to current/actual access dates.

## Missing Citations - CRITICAL

### Core Architecture Components (Section 2.2 - Related Work)
- [ ] **Swin Transformer**: Mentioned in Section 2.2 ("Swin Transformer visual encoder") but no citation provided.
  - **ADD**: Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., & Guo, B. (2021). "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows". ICCV 2021.
  - **BibTeX key suggestion**: `swin` or `liu2021swin`

- [ ] **BERT**: Mentioned in Section 2.2 ("BERT language backbone") but no citation provided.
  - **ADD**: Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding". NAACL 2019.
  - **BibTeX key suggestion**: `bert` or `devlin2019bert`

- [ ] **Transformer architecture**: "transformer blocks" and "cross-attention" mentioned but original Transformer paper not cited.
  - **ADD**: Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). "Attention is All You Need". NeurIPS 2017.
  - **BibTeX key suggestion**: `transformer` or `vaswani2017attention`

- [ ] **Vision Transformer (ViT)**: SAM-ViT-Base and SAM-ViT-Large mentioned throughout Section 5 but ViT paper not cited.
  - **ADD**: Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., & Houlsby, N. (2021). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale". ICLR 2021.
  - **BibTeX key suggestion**: `vit` or `dosovitskiy2021vit`

### Implementation and Training (Section 5.1-5.2)
- [ ] **PyTorch**: Mentioned in Section 5.1 ("We reimplemented the architecture in PyTorch") but no citation.
  - **ADD**: Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Köpf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J., & Chintala, S. (2019). "PyTorch: An Imperative Style, High-Performance Deep Learning Library". NeurIPS 2019.
  - **BibTeX key suggestion**: `pytorch` or `paszke2019pytorch`

- [ ] **Mixed precision training**: Mentioned in Section 5.2 ("mixed precision") but no citation.
  - **ADD**: Micikevicius, P., Narang, S., Alben, J., Diamos, G., Elsen, E., Garcia, D., Ginsburg, B., Houston, M., Kuchaiev, O., Venkatesh, G., & Wu, H. (2018). "Mixed Precision Training". ICLR 2018.
  - **BibTeX key suggestion**: `mixedprecision` or `micikevicius2018mixed`

### Foundational Referring Expression Work (Abstract/Introduction)
- [ ] **Referring expression segmentation foundations**: The task is introduced in Abstract and Section 1 but lacks foundational citations.
  - **ADD ONE OR MORE**:
    - Kazemzadeh, S., Ordonez, V., Matten, M., & Berg, T. L. (2014). "ReferItGame: Referring to Objects in Photographs of Natural Scenes". EMNLP 2014.
    - Yu, L., Poirson, P., Yang, S., Berg, A. C., & Berg, T. L. (2016). "Modeling Context in Referring Expressions". ECCV 2016.
    - Hu, R., Rohrbach, M., & Darrell, T. (2016). "Segmentation from Natural Language Expressions". ECCV 2016.
    - Mao, J., Huang, J., Toshev, A., Camburu, O., Yuille, A. L., & Murphy, K. (2016). "Generation and Comprehension of Unambiguous Object Descriptions". CVPR 2016.
  - **BibTeX key suggestions**: `kazemzadeh2014referit`, `yu2016modeling`, `hu2016segmentation`, `mao2016generation`

### Future Work Citations (Conclusion)
- [ ] **Tower Instruct**: Mentioned in Conclusion as multilingual translation system but not cited.
  - **ADD**: Alves, D., Guerreiro, N. M., Alves, J., Pombal, J., Rei, R., Fernandes, J. G. C., Farinhas, A., Coheur, L., & Martins, A. F. T. (2024). "Tower: An Open Multilingual Large Language Model for Translation-Related Tasks". arXiv:2402.17733.
  - **BibTeX key suggestion**: `tower` or `alves2024tower`

### Evaluation Metrics (Used Throughout)
- [ ] **IoU (Intersection over Union)**: Used as primary metric throughout (mIoU, oIoU) but never cited.
  - **ADD ONE**:
    - Everingham, M., Van Gool, L., Williams, C. K. I., Winn, J., & Zisserman, A. (2010). "The Pascal Visual Object Classes (VOC) Challenge". IJCV 2010.
    - OR: Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., & Zitnick, C. L. (2014). "Microsoft COCO: Common Objects in Context". ECCV 2014.
  - **BibTeX key suggestions**: `everingham2010pascal` or `lin2014coco`

### Optional/Contextual Citations
- [ ] **Knowledge distillation** (Section 3.3): The concept is used but not explicitly cited.
  - **CONSIDER ADDING**: Hinton, G., Vinyals, O., & Dean, J. (2015). "Distilling the Knowledge in a Neural Network". NeurIPS 2014 Workshop / arXiv:1503.02531.
  - **BibTeX key suggestion**: `hinton2015distilling`

- [ ] **Open-vocabulary concept**: Mentioned in Abstract and Introduction but could cite foundational work.
  - **CONSIDER ADDING**: Bansal, A., Sikka, K., Sharma, G., Chellappa, R., & Divakaran, A. (2018). "Zero-Shot Object Detection". ECCV 2018. OR similar foundational open-vocab papers.

- [ ] **OpenRouter**: Mentioned in cost footnote as "OpenRouter inference provider". This is a commercial service.
  - **SOLUTION**: Either remove specific mention or add URL footnote: `\footnote{\url{https://openrouter.ai}}` instead of citation.

## Content Completeness and Clarity

### Missing Details or Explanations
- [ ] **Color threshold edge case**: Section 3.2 says 70% for achromatic, 60% for chromatic. Doesn't explicitly state what happens when neither threshold is met.
  - **SOLUTION**: Add clarifying sentence: "When neither threshold is met, no color descriptor is assigned and the cue is discarded."

- [ ] **iSAID patch extraction detail**: Section 3.1 says "slide a 480×480 window with overlap" but doesn't specify overlap amount.
  - **SOLUTION**: Either add specific overlap amount (e.g., "50% overlap" or "240-pixel stride") or add note "(overlap details omitted for brevity)".

- [ ] **Urban1960SatSeg supervision clarity**: Section 5.5, Table 6 shows removing Urban1960SatSeg causes major drop.
  - **SOLUTION**: Add clarifying sentence earlier (Section 5.2 or when introducing Urban1960SatSeg): "Note that Urban1960SatSeg is inherently historic imagery, so it receives no additional filtering and provides direct supervision for archival conditions."
