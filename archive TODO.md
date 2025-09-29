# Completed Tasks Archive

## Article writing

* We are using different terms all the time, Reffering instance segmentation, reffering expression segmentation. we should stick with one term, probably reffering expression segmentation
* missing explanation right at the start of introduction of what that term comprises of
* remove term "geographi complexities" as its vague. We can substitute it for something else.
* "the biggest dataset" "the first fully automatic pipeline" lets not use these terms that are risky since they require scrutiny to know if they are true
* "dataset represents pipline" no, was constructed with pipeline
* "bechmarks" no, datasets
* missing a phrase on abstract and introduction highlighting the historical capabilities as a special feature of the work
* "We introduce" no, lets not use "we" since the work was done by one person. "This work" or "This work introduces" 

## Dissertation writing:

* 1.2 Document Structure - do not say AerialSeg
* 2 - Introduction - root text does not cover 2.1
* 2.1 - x1, x2 and all other indices on the figure should be properly formated as indices, figure 2.1 and 2.2
* 3 - root does not mention 3.2
* 3.1.1 - citations missing after iSAID and LoveDA
* 3.1.1 - no mention of figure 3.1 or 3.2
* 3.1.2 - missing citations after RefSegRS, RRSIS-D, and many others. Citations should not be at the end of paragraphs like that
* 3.1.2 - no mention of figure 3.3 in the text
* 3.1.3 - citations missing for UrbanSatSeg1960
* 4 - Aerial-D, not AerialD
* 4.1 - Aerial-D not AerialD
* 4.2 - Repeated explanation of LoveDA conversion to instance segmenation due to poor transition between those paragraphs
* 4.2 - missing citation to DBSCAN
* 4.3 - Large Language Models not in upper case and missing acronym after.
* 4.3 - missing citation to GPT-5 and o3
* 4.3 - could cost thousands of dollars, not hundreds, mention to the ablation study section which shows those costs
* 4.3 - citations need to appear after the keyword is mentioned at all times, for instance QLoRA
* 4.4 - equation 4.6 is hanging alone
* 4.5 - Aerial-D not AerialD, also on figure 4.6 caption
* 5 - do not say AerialSeg
* 5 - root is poorly written, we should start with a carry over of the dataset we have just built in the previous chapter and how we will now use it to experiment, in addition to other datasets, to train and evaluate models. Remove the strange prompt to every figure table etc, clearly an error
* 5.1 - missing citations to PyTorch and even LoRA and others. It is not fine tuned for aerial imagery out of the box, the LoRA matrices are what we train as per the RSRefSeg paper, plus the attention prompter. Consider revisiting the description of RSRefSeg in related work to better explain the role of attnprompter
* 5.2 - we now use SAM-ViT-Base, not Large. "The model" , not "The checkpoint"
* 5.2 - poor explanation of choice for Unique Only set. It is because we are trying to not overwhelm the other datasets. Mention to ablation studies showing that the Unique Expressions contain a lot of signal as shown in the generalization capability of a model trained on Unique Only.
* 5.3 -  poorly written introduction to section. "Our main experiment" or "We first begin our experimentations with training a single model on all 5 datasets, while testing it also on all of them".
* 5.3 percentages overflowing the paper
* 5.3 -  the model does not match or exceed, it has comparable results to the results reported in those papers
* 5.4 - missing root of the section
* 5.4.1 - do not mention the change in SAM model since we are now only using Base and rank 16.
* 5.4.1 - inconsistency with percentages being bold in this section but not on the others
* 5.4.2 -  no reminder of the LLM enhancement phase and how we can choose a different LLM etc, therefore we do ablation to see the impact of switching LLMs etc
* 5.4.2 -  table 5.3 is huge
* 5.4.2 - because infrence no longer depends on a comercial api is wrong, its because the o3 model is much larger and requires a lot more compute to run compared to gemma3
* 5.4.2 - footnote might not be allowed in dissertation
* 5.4.3 -  table needs update to the results
* 6 - do not say AerialSeg
* 6 - root should preface the next two sections, "we finalize the work by concluding over the accomplishements, and also looking ahead into the future to extensions of this work and directions" something on those lines
* 6.2 - such as GPT-5 eg o3 is strange, which is it
* 6.2 - gemini 2.5 citation is wrong, pixel level understanding is not advartised like that or in those words