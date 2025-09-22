## AerialSeg: Open‑Vocabulary Aerial Image Segmentation with Referring Expressions

> 📝 **Documentation under construction** - The codebase is complete, but README files and documentation are being finalized.

[![Project Page](https://img.shields.io/badge/Project%20Page-visit-blue)](https://luispl77.github.io/aerialseg)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-orange)](https://huggingface.co/datasets/luisml77/aerial-d)
[![Paper](https://img.shields.io/badge/Paper-Coming%20Soon-lightgrey)](#)

![AerialSeg dataset example](docs/dataset.png)

AerialSeg is an open‑source framework for segmenting aerial images from natural‑language prompts. It includes:
- Aerial‑D: an automatic dataset pipeline for building referring expressions over aerial imagery
- ClipSAM: a SigLIP+SAM model for referring segmentation
- LLM tooling: Gemma3/OpenAI utilities for enhancing expressions and fine‑tuning

Use the root `requirements.txt` for dependencies.

---

## Aerial‑D Generation Pipeline
First, we build the dataset. We take aerial images and automatically write short descriptions for each object, then keep only the ones that clearly point to a single target. When needed, we also produce a "historic" version to stress‑test robustness. To reproduce the fully automatic pipeline from the paper, see `datagen/README.md` and [Section 4](https://luispl77.github.io/aerialseg#section-4).

---

## Model Training and Testing (RSRefSeg)
Next, we teach the model to follow text and segment the right thing. The RSRefSeg architecture is trained on Aerial‑D and can also be trained jointly with RRSISD, RefSegRS, NWPU‑Refer, and Urban1960SatBench to test transfer. To reproduce the results reported in the paper, follow `clipsam/README.md` and see [Table 3](https://luispl77.github.io/aerialseg#table-3) and [Table 6](https://luispl77.github.io/aerialseg#table-6).

---

## LLM Fine‑Tuning Pipeline
Finally, we polish the language. After the rule‑based pipeline, the descriptions are rewritten by language models, which makes them more natural and visually informative. The `llm/` folder has the code to train the Gemma 3 model (QLoRA) used for this large‑scale enhancement, plus O3‑based augmentation. To recreate these results, use `llm/README.md` and the paper’s LLM section ([link](https://luispl77.github.io/aerialseg#llm-finetuning)).

---

## Citations
If you use this repository, please cite the dataset and (when available) the thesis/paper.

```bibtex
@dataset{aerial-d-2024,
  title={AERIAL-D: Referring Expression Instance Segmentation in Aerial Imagery},
  author={Luis M. Lopes and contributors},
  year={2024},
  publisher={Hugging Face},
  url={https://huggingface.co/datasets/luisml77/aerial-d}
}
```

---

## Acknowledgments
- iSAID and LoveDA datasets
- Hugging Face Transformers, PyTorch, OpenCV, Flask
- SAM and SigLIP authors
- Google Gemma and OpenAI models used for language enhancement

---

## Contributing
Issues and pull requests are welcome. Please open an issue to discuss substantial changes.

## Contact
For questions, please contact: [maarnotto@gmail.com](mailto:maarnotto@gmail.com) or open an issue on the repository.
