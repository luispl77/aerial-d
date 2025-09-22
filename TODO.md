Dissertation writing: 

- Cover - add the Examination Committee and correct month and year
- 1.2 Document Structure - do not say AerialSeg [FIXED]
- 2 - Introduction - root text does not cover 2.1 [FIXED]
- 2.1 - x1, x2 and all other indices on the figure should be properly formated as indices, figure 2.1 and 2.2 [FIXED]
- 3 - root does not mention 3.2 [FIXED]
- 3.1.1 - citations missing after iSAID and LoveDA [FIXED]
- 3.1.1 - no mention of figure 3.1 or 3.2 [FIXED]
- 3.1.2 - missing citations after RefSegRS, RRSIS-D, and many others. Citations should not be at the end of paragraphs like that [FIXED]
- 3.1.2 - no mention of figure 3.3 in the text [FIXED]
- 3.1.3 - citations missing for UrbanSatSeg1960 [FIXED]
- 3.1.3 - missing mention to figure 3.4 and to all tables in text
- 3.2 - missing citations for CLIP and SigLIP, and others. 
- 3.2 - missing citation to RSRefSeg , citations go directly after the first mentions of those resources
- [FIXED] 4 - Aerial-D, not AerialD
- [FIXED] 4.1 - Aerial-D not AerialD
- [FIXED] 4.2 - Repeated explanation of LoveDA conversion to instance segmenation due to poor transition between those paragraphs
- [FIXED] 4.2 - missing citation to DBSCAN
- [FIXED] 4.3 - Large Language Models not in upper case and missing acronym after.
- [FIXED] 4.3 - missing citation to GPT-5 and o3
- [FIXED] 4.3 - could cost thousands of dollars, not hundreds, mention to the ablation study section which shows those costs
- [FIXED] 4.3 - citations need to appear after the keyword is mentioned at all times, for instance QLoRA 
- [FIXED] 4.4 - equation 4.6 is hanging alone
- [FIXED] 4.5 - Aerial-D not AerialD, also on figure 4.6 caption
- 5 - do not say AerialSeg
- 5 - root is poorly written, we should start with a carry over of the dataset we have just built in the previous chapter and how we will now use it to experiment, in addition to other datasets, to train and evaluate models. Remove the strange prompt to every figure table etc, clearly an error
- 5.1 - missing citations to PyTorch and even LoRA and others. It is not fine tuned for aerial imagery out of the box, the LoRA matrices are what we train as per the RSRefSeg paper, plus the attention prompter. Consider revisiting the description of RSRefSeg in related work to better explain the role of attnprompter
- 5.2 - we now use SAM-ViT-Base, not Large. "The model" , not "The checkpoint"
- 5.2 - poor explanation of choice for Unique Only set. It is because we are trying to not overwhelm the other datasets. Mention to ablation studies showing that the Unique Expressions contain a lot of signal as shown in the generalization capability of a model trained on Unique Only. 
- 5.3 -  poorly written introduction to section. "Our main experiment" or "We first begin our experimentations with training a single model on all 5 datasets, while testing it also on all of them". 
- 5.3 percentages overflowing the paper
- 5.3 -  the model does not match or exceed, it has comparable results to the results reported in those papers
- 5.4 - missing root of the section
- 5.4.1 - do not mention the change in SAM model since we are now only using Base and rank 16. 
- 5.4.1 - inconsistency with percentages being bold in this section but not on the others
- 5.4.2 -  no reminder of the LLM enhancement phase and how we can choose a different LLM etc, therefore we do ablation to see the impact of switching LLMs etc
- 5.4.2 -  table 5.3 is huge
- 5.4.2 - because infrence no longer depends on a comercial api is wrong, its because the o3 model is much larger and requires a lot more compute to run compared to gemma3
- 5.4.2 - footnote might not be allowed in dissertation
- 5.4.3 -  table needs update to the results
- [FIXED] 6 - do not say AerialSeg
- [FIXED] 6 - root should preface the next two sections, "we finalize the work by concluding over the accomplishements, and also looking ahead into the future to extensions of this work and directions" something on those lines
- [FIXED] 6.2 - such as GPT-5 eg o3 is strange, which is it
- [FIXED] 6.2 - gemini 2.5 citation is wrong, pixel level understanding is not advartised like that or in those words
- Appendix A - missing reference to it on LLM enhancement section. Missing also description of complex dual image prompting tactic on LLM enhancement section
- missing list of acronyms
- We need to stick all figures to the top of the page
Code: 

root README.md: 
- Unnatural start to first subsection. Should begin with "the core of the work is to build Aerial-D , and on datagen folder is where all the tools are, from"
- Unnatural second subsection. "We implement the RSRefSeg architecture (github link) in pytorhc from scratch. the clipsam folder contains model.py describing the architecture, train.py to train, test.py etc etc"
- The Aerial-D creation pipeline includes a llm enhancement step, as described in the paper, in order to generate rich refering expressions. the llm folder contains the code to fine tune gemma 3 as well as obtain the data for it with o3

datagen/README.md
- Remove deepgloble as it is no longer part of the dataset
- missing mentioning bash scripts that download all the datasets
- only rule_viewer.py is functional currently
- missing mention to how to run the last step of the pipeline, the LLM enhancement

llm/README.md
- remove the gemma 3 enhancement scripts mention, as the last step of the pipeline takes care of that
- focus on first the o3 synthetic generation script
- then we mention how to train on the folder taht the o3 outputed

clipsam/README.md is mostly fine



