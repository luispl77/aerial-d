# TODO - Development Tasks

## Tasks That Need To Be Completed

### Article writing
Related Work:
* "as well as benchmarks that test language" innacurate, as it suggests there are datasets and benchmarks, when there are only datasets. A more accurate addition to the first phrase would be simply replacing benchmarks with datasets.
* missing citation on CLIP and SigLIP
* talking about RMSIN, we should mention the core techers they are using for the three module. for instance, ARC is convolution, while the IIM contains a transformer block, CIM uses cross attention. we should alsomention that while this uses specialized networks for the vision backbone they use a swin transformer and a bert for language backbone. SO in reality, this seciton 2.2 Architectures for RRSIS needs a major rewrite because there is no such thing as "specialized networks" , everythin uses backbones. so we need to drop the "two families" thing. the other angle that needs rewriting is mentioning all these names IIM or CIM or ARC. this is not relevant. when describing each architecture, we should instead focus on backbones and how information is fused together: cross attention, convolution, and the general flow of the information while focusing not on random terms but the concepts that are employed. 
* in fact, MRSNet is very similar to RMSIN, same backbones but the feature interaction is slightly different. needs more research to understand the brief highlihgt of difference
* RSRefSeg is different, as it uses CLIP with SAM as the backbones, and has a lightweight fusion using convulutional layers to prompt SAM using the features from the CLIP text and vision encoders. It forms sparse prompts and dense prompts, and training is done on those convolutional layers as well as added LoRA matrices to the layers in CLIP and SAM vision encoder to adapt to the domain as well as extract important domain specific features that layer allow the model to segment the region
* again on the overview we remove the innacura specialized network paths. 
* going back to the datasets related work analysis, on RRSIS-D i think we forgot to mentino the source of data is DIOR, lets include that
* we also forgot to mention RSRefSeg achieves SOTA benchmark IoU scores on RRSIS-Dand this is also proof of its effectiveness for the RRSIS task. 
* on the end of the overview, there is an unsupported claim to strong generalization and "modest adaptation" lets not say these things, lets just focus on the supporting RRSIS-D results and strong / modern backbone utilization compared to the other architectures
* also replcae "comprehensive benchmark" with comprehensive dataset again
Experiments:
* "we first describe the model architecture and training setup used"
* on Model Architecture, it has too much fluf, the point is to just say that we chose to utilize an existing proven model architecture, and the one we chose for the experiments was RSRefSeg due to stong reported results on RRSIS-D already. We also were able to adapt this archtiecture completely into our own PyTorch implementation adn verify those same results on RRSIS-D indenpendely, which confirmed the trust in the architecture to proceed with experiments. the rest of the seciton is good actually, just this littly clarification
* we can include the GPU utilized was RTX A6000, one. we can specify the resultion of siglip was 384x384 . we already say that actually later, but best to place it early next to siglip. 
* there is a problem across the paper with terminology being inconsistent on the Unique Only and Enhanced Only. these mean LLM Language Variations and LLM Visual Variations expressions, and we need to utilize these two terms only. 
* "Aerial-D uses the combined-all validation set" this needs to be more clear, something like it uses all the 405k expressions. however we do break that down into two, smantic and instance 
* we removed the threshold pass metrics so we need to remove that from text too
* "by supervision type" we need another term, by target type maybe, semantic targets (land cover and all objects of a category) adn instance types (target single instances and groups of instances) as well as the full evaluation on all 405k targets.
* "awaiting the placeholder entires" remove this now. explain difference between l and b are the rank, b stands for base and uses rank 16, l stands for large and uses rank 32, also b uses sam vit base, l uses sam vit large
* on the Table 3, these are meant to be used as baseline values for future work evaluated on Aerial-D of other models 
* on the scores of table 4, we dont need to specify numbers on our description, just highlight the key thing: the model matches or surpasses performance compared to existing models evaluated on those same datasets, in some cases, while maintaining robust performance on historic image filter handling and all 5 datasets evaluated in total. 
* "the pending run" remove that obviously
* "the combined model blends rule with llm expressions" wrong, it usees only LLM Visual Variations expressions, and this is exactly motivated and shown on this ablatino study where the model is trained only on aerial-d, 4 different models, and in fact the LLM Visual Variations have quite a bit generalizaiton capability due to the richness of the language, which makes for a higher signal subset of aeriald that requires less data to converge as shown by the epoch numbers until validation starts to rise
* "gemma 3 frequently hallucinates" important to also highlight that it also fails to follow insttructions to do both the required tasks correctly in addition to to producing that correct output schema and producing artifacts such as mentionning the red bounding box which is meant as a guide to the target and not to be mentioned in the final expression
* on the historic ablation, we utilize the base b model for comparison, ebcause it rains faster
* "each block now seperates" remove this obviosly, "now" lol
Conclusion and Future Work:
* "benchmarks" become datasets
* on the future work for multilingual, we can also mention more generally we can use specialized translation models to produce multilingual expression translations from the dataset, while keeping the full process still fully automatic. thes models can be LLMs with the proven gemma 3 distilation pipeline being able to be adapted for this task, or a dedicated translation model such as tower instruct
* for the gemini 2.5 "exhibit image geneation" is innacurate, rather the model is trained to produce complete segmentation masks of objects and bounding boxes, with remarkable generalizaiont capabilities inherent of foundation LLMs, can be used to for instance take every image of Aerial-D and produce additional instance segmentations , expanding the mask and object/target variety in the dataset to be unrestricted, much like we unrestrict the language
* on the conclusion side, emphasize the importance of Aerial-D in enabling a harder benchmark for RRSIS and in general referring expression segmentaiton in aerial pphotos, allowing for model work and experimentation with it 
### Dissertation writing:

* Cover - add the Examination Committee and correct month and year
* 3.1.3 - missing mention to figure 3.4 and to all tables in text
* 3.2 - missing citations for CLIP and SigLIP, and others.
* 3.2 - missing citation to RSRefSeg , citations go directly after the first mentions of those resources
* Appendix A - missing reference to it on LLM enhancement section. Missing also description of complex dual image prompting tactic on LLM enhancement section
* missing list of acronyms
* We need to stick all figures to the top of the page
* 

### Code:

#### root README.md:
* Unnatural start to first subsection. Should begin with "the core of the work is to build Aerial-D , and on datagen folder is where all the tools are, from"
* Unnatural second subsection. "We implement the RSRefSeg architecture (github link) in pytorhc from scratch. the clipsam folder contains model.py describing the architecture, train.py to train, test.py etc etc"
* The Aerial-D creation pipeline includes a llm enhancement step, as described in the paper, in order to generate rich refering expressions. the llm folder contains the code to fine tune gemma 3 as well as obtain the data for it with o3

#### datagen/README.md
* Remove deepgloble as it is no longer part of the dataset
* missing mentioning bash scripts that download all the datasets
* only rule_viewer.py is functional currently
* missing mention to how to run the last step of the pipeline, the LLM enhancement

#### llm/README.md
* remove the gemma 3 enhancement scripts mention, as the last step of the pipeline takes care of that
* focus on first the o3 synthetic generation script
* then we mention how to train on the folder taht the o3 outputed

#### clipsam/README.md is mostly fine

## Completed Tasks

### Article writing
Introduction:
* missing command after "group-level expressions"
* "A critical component to developing" here we say models for RRSIS, but more generally its models for RRSIS and referring segmentation of aerial photos
* "compared with prior RRSIS datasets" lands wrong, is it the large-scale? we should explicitly mention something like , significantly larger than some of the prior RRSIS benchmarks
* "each sample includes a hisstoric counterpart" innacurate, the historic filters are applied on the fly. Therefore, we remove completely this phrase.
* "datasets from instance segmentation datasets" becomes instance/semantic segmentaion datasets
* "applying the full toolchain" innacurate, we apply only the historic image transformations to other datasets, excepet for one which we apply some of the rule based tools. therefore, the accurate way of saying is not "full tool chain" but "some of our tools such as the historic transformations" and it is not across all data either

Aerial-D Dataset Construction:
* "our distillation approach" is incomplete, what is is is a multimodal LLM expression generation component which includes a cost and efficient preceding distillation step on the LLM to allow us to generate expressions on a large scale of targets
* the source datasets order of explanation of the image resizing and sliding windo is innacurate. we should say for LoveDA we simply resize to 480x480, etc, because images are just square images already, while iSAID has uniquely high resultion and varying resolution images, which require a sliding window approach with that same 480x480 resultion to get all patches that have instances into our starting images and annotations. 
* we say "buildings and water are amenable" this is incomplete and vague writing, the point is that because those are usually isolated and bounded instance like compoennets, we do a transformation through connnected component analysis on LoveDA on those two categories, creating a instance dataset from those semantic categories , while the others are kept with semantic lables such as "all agricultural land in the image".
* "describe these targets objects in natural language using only what we know from their bboxes etc"
* HSV missing acronym expansion, color values
* lacking a bit more information on color selection. a simple mention to threshold values that aim to select a color with high confidence that its legit while dropping instances that show multi hues and are not evident which color they are
* "we target the enhancement from two approaches" we can be a bit more accurate and say we prompt the LLM with two different tasks
* "we provide bounding box annotations" more accurately, we overlap the bounding box or boxes of the target to help the LLM find it more easily. We also use other tricks to focus the attention of the LLM, providing two images, the first the entire original imagea and a second close up view on the target which helps with small targets and looking at surrounding features. we also utilize a mask overlay for land cover categories such as vegetation, because bounding boxes are not adequate to help locate these regions.
* on the distillation side, we also note that the method also makes the student model Gemma3 avoid common pitfalls related to lack of instructions follwoing, for instance, mentioning the bounding box in the reffering expression, adhering to the task schema reliably, and outputing consistent styles of referring expressions, as well as the obvious reduced hallucination, partly due to the LoRa matrices being applied on the SigLIP vision encoder that Gemma3 itself uses for its multimodal backbone. when we mention the comparison with the base Gemma3 model, its important to highlight that this model is completely unusable out of the box because it does not even follow the required 2 task system wihtout finetuning due to its limited size, making the distallation necessary
* we may need to reveiw the historic filter agumentation to clarify if its accruate with the code and also th explanation. 
* i forgot a crucial aspect about the necessity of the rule based component, is in identifying targets that are unambiguous right out of the box. without this compoenent, the ambiguity problem of the expressions would make for a very unreliable dataset wher the expressions pointed to multiple targets, affecting trainng and model performance. its the selection of the targets by the rule based system that serves as a filter over ambiguous and hard to distinguish objects that might be present particularly in high density scenes. 
* on the comparison with other datasets, lets also describe and clarify the number of referring instances/groups versus the number of semantic categories. these two numbers are also shown in the comparison table

