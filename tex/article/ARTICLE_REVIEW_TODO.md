# Article Review TODO

Collected issues from the thorough article review. Excludes template-compliance items and general “clarity checks” per request.

## High-Priority Content/Results

### Critical Numerical Inconsistencies
- [ ] **Expression count mismatch**: Abstract states "1,522,523 referring expressions" but Table 2 (tab:llm_enhancement_stats) shows "Total: 1,523K". Verify which is correct (1,523K rounded = 1,523,000 ≠ 1,522,523).
- [ ] **Sample count arithmetic error**: Section 5.3, Table 5 lists individual splits as 371K + 364K + 382K = 1,117K, but "Combined All" row shows 1,118K samples. Fix math or explain the extra 1K.
- [ ] **Target count terminology confusion**: Section 3.2 says "approximately 300,000 captured targets" but Section 3.5 says "259,709 annotated targets". Clarify that 300K is pre-filtering and 259,709 is post-uniqueness-filtering.
- [ ] **Class count verification**: Abstract claims "21 distinct classes" but iSAID has 15 categories and LoveDA has 7 classes (15 + 7 = 22, not 21). Verify if there's overlap or if one class was excluded.

### Model Architecture Clarity
- [ ] **LoRA rank confusion**: Section 5.1 says RSRefSeg-b uses "r=16" and RSRefSeg-l uses "r=32"; Section 5.2 says combined model uses "r=32"; Section 5.3 says ablation uses RSRefSeg-b with "r=16". Clarify which model (b vs l) is used for which experiments, as Section 5.2 mentions both but defaults to RSRefSeg-l.
- [ ] **SigLIP vs SigLIP2 terminology**: Section 2.2 mentions "CLIP or SigLIP" for RSRefSeg, but Section 5.1/5.2 use "SigLIP2" explicitly. Standardize to SigLIP2 throughout, including in Related Work when discussing RSRefSeg.
- [ ] **RSRefSeg-b recipe omission**: Table 4 reports RSRefSeg-b results, but Section 5.2 describes only the RSRefSeg-l training configuration. Add the lighter model's training details (backbone size, LoRA ranks, learning rate, etc.) for reproducibility.

### Tables and Results
- [ ] Aerial-D variants table: fix "All Targets" metrics (Table `tab:aeriald_variants`) — they currently duplicate the "Instance Targets" values; "All" should aggregate instance + semantic and differ.
- [ ] **RefSegRS performance gap**: Table 4 shows RMSIN gets 59.96% mIoU but RSRefSeg-l only gets 44.52%. Acknowledge this gap in text and explain why (e.g., RMSIN was trained only on RefSegRS, while yours is multi-dataset).
- [ ] Historic filters wording: resolve contradiction between "on-the-fly during training" (Section 3.4) and "during data preparation we apply filters to 20% of images" (Section 5.2). Choose one accurate description and use consistently. Also clarify that each image gets ONE randomly chosen filter, not all three.

### Dataset and Citation Issues
- [ ] Dataset name consistency: standardize to a single form (e.g., "Aerial-D") throughout; avoid mixed forms like "AERIAL-D" in the dataset comparison table and "Aerial\mbox{-}D" in Conclusion.
- [ ] **LLM split naming**: Table 3 labels the third split "LLM Visual Details" while Sections 3.3/5.3 and Table 5 refer to "LLM Visual Variations". Pick one name and use it consistently in both prose and tables.
- [ ] NWPU-Refer citation/year: key `yang2024large` has year 2025 in the BibTeX/bbl. Ensure the text consistently refers to the correct year or update the key.
- [ ] **Dataset count phrasing**: Introduction says "four additional datasets" (beyond Aerial-D) while Conclusion says "five datasets" (total). While technically consistent, consider stating "five datasets (Aerial-D plus four public benchmarks)" in one place for clarity.

## Language/Style Consistency
- [ ] Choose US or UK English and standardize globally. Examples to normalize:
  - analysing → analyzing; artefacts → artifacts; colour → color; stabilises → stabilizes; honour → honor; behaviour → behavior; grayscale vs grey-scale.
- [ ] Unify the hyphenation of the dataset name (e.g., always "Aerial-D"). If using a non-breaking hyphen (`Aerial\mbox{-}D`), apply it consistently everywhere (currently inconsistent in Conclusion).
- [ ] **Land-cover hyphenation**: Mixed usage of "land-cover" (Section 3.1, 3.5, Abstract) vs "land cover" (Introduction). Standardize on one form throughout (recommend "land-cover" as compound adjective).
- [ ] **Number formatting consistency**: Uses both comma-separated (1,522,523) and K notation (1,523K) throughout. While acceptable, consider being more consistent within individual sections.
- [ ] **Validation split naming**: Section 5.2 uses '405K-expression "combined-all" split' with quotes, Section 5.3 uses "combined-all validation split" without quotes. Minor formatting inconsistency.

## Figures, Tables, Cross-References

### Table Formatting and Accuracy
- [ ] Cost table footnote placement: confirm `\caption{...\protect\footnotemark}` + `\footnotetext{...}` renders correctly and at the intended location (Table `tab:cost_comparison`).
- [ ] **Cost calculation verification**: Section 5.4 footnote has very detailed token counts. Verify these numbers are accurate based on actual API usage logs.
- [ ] Boldface in result tables: verify that bolded entries actually correspond to best-in-column and that metric definitions (mIoU, oIoU) are consistent across tables.
- [ ] **Table sizing**: Several tables use `\resizebox{\textwidth}{!}{...}` to fit. Verify all tables render properly and text remains legible at reduced size.
- [ ] **Blue color accessibility**: Historic scores shown in blue via `\textcolor{blue}{...}`. Ensure this is accessible; consider also using different formatting like italics for print versions.

### Figure Verification
- [ ] Category distribution figures: validate that the distributions in `group_category_distribution.png` and `instance_category_distribution.png` reflect the reported counts/splits in text and tables.
- [ ] **Figure existence and quality check**: Verify all referenced figures exist and are high quality:
  - 6samples.png
  - rule_based_generation.png
  - example_group.png
  - filters.png
  - rsrefseg.png
  - expression_wordcloud.png
  - distillation.png
  - 3llm.png
  - group_category_distribution.png
  - instance_category_distribution.png

## Technical/LaTeX Preamble Notes
- [ ] Replace deprecated `subfigure` package with `subcaption` (or remove `subfigure` if not used) to avoid legacy behavior.
- [ ] Consider relying on `titlesec` alone and removing `sectsty` to avoid package warnings about redefined commands; check headings remain as intended after changes.
- [ ] Review underfull hbox/vbox warnings; adjust paragraph breaks or float placements (or minor spacing) where layout visibly suffers.
- [ ] **Equation formatting**: Verify equations 1-6 for historic filters compile without errors and display properly.

## Numerical/Data Consistency
- [ ] Confirm the "21 distinct classes" statement (Abstract) with the final category list used in figures/tables.
- [ ] Verify the preprocessing and model input sizes narrative: source patches (480×480) vs encoder inputs (SigLIP2 384×384, SAM 1024×1024) match the actual training pipeline.
- [ ] Ensure statements about training on Aerial-D only using "LLM Visual Variations" (combined run) are consistently reflected where training mixes are described.
- [ ] **Sepia noise mean bias**: Equation \eqref{eq:sepia_noise} adds $\mathcal{U}(0,50)$ noise, which has a positive mean and brightens images. Either shift this to zero-mean noise or explain why the asymmetric noise is intentional.

### Additional Data Verification Needed
- [ ] **Summary statistics to cross-check**:
  - Total expressions: 1,522,523 or 1,523K? (verify exact number)
  - Total targets: 259,709 (verify)
  - Total patches: 37,288 (verify)
  - Instance expressions: 1,278,453 (verify: 1,278,453 + 244,070 = 1,522,523 ✓)
  - Semantic expressions: 244,070 (verify)
  - Validation expressions: 405K (verify against train 1,118K)
- [ ] **Directional relationships count**: Section 3.2 says "eight directional relationships: above, below, to the left of, to the right of, and the four diagonal directions" (4 cardinal + 4 diagonal = 8). However, CLAUDE.md guidelines suggest only 4 directions. Verify if diagonals are actually used in the implementation.

## Bibliography/Metadata
- [ ] Where possible, enrich `@misc` entries with venue information once available; ensure consistent capitalization for titles (e.g., iSAID).
- [ ] Confirm `\bibliographystyle{abbrv}` aligns with the intended citation style for the submission; otherwise switch styles and recompile.
- [ ] **Future dates in citations**: Several citations say "Accessed 2025-09-13" which is in the future. Update to actual access dates or current dates.

## Content Completeness and Clarity

### Missing Details or Explanations
- [ ] **Color threshold edge case**: Section 3.2 says 70% for achromatic, 60% for chromatic. Doesn't explicitly state what happens when neither threshold is met (though it implies color is discarded). Could be clearer.
- [ ] **iSAID patch extraction detail**: Section 3.1 says "slide a 480×480 window with overlap" but doesn't specify overlap percentage/amount. Consider adding this detail or noting it's intentionally omitted for brevity.
- [ ] **Urban1960SatSeg supervision clarity**: Section 5.5, Table 6 shows "No Filters + No Urban1960SatSeg" causes major drop. Should clarify earlier in the paper that Urban1960SatSeg is ALWAYS used with inherent historic properties since it's already historic imagery.
- [ ] **Historic filter application phrasing**: Section 5.2 says "apply filters to 20% of training images in each non-historic dataset" - should explicitly clarify that each selected image gets ONE randomly chosen filter (not all three applied sequentially).

### Abstract and Introduction
- [ ] **Abstract length check**: The abstract is quite dense and long. Verify it doesn't exceed typical conference/journal limits (usually 150-250 words). Consider if any content can be condensed.
