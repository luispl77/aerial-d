# TODO - Development Tasks

## Tasks That Need To Be Completed

### Article writing
* We are using different terms all the time, Reffering instance segmentation, reffering expression segmentation. we should stick with one term, probably reffering expression segmentation
* missing explanation right at the start of introduction of what that term comprises of
* remove term "geographi complexities" as its vague. We can substitute it for something else.
* "the biggest dataset" "the first fully automatic pipeline" lets not use these terms that are risky since they require scrutiny to know if they are true
* "dataset represents pipline" no, was constructed with pipeline
* "bechmarks" no, datasets
* missing a phrase on abstract and introduction highlighting the historical capabilities as a special feature of the work
* "We introduce" no, lets not use "we" since the work was done by one person. "This work" or "This work introduces" 
* Table 1 , 3 and 6 are the only tables in the article. Need to renumber
* Table 1: Aerial-D has both reffering instance segmentation and reffering semantic segmentation in it. We should sperate the two and count them seperately, which makes for a more fair comparison with the other datasets
* Table 2: same thing here, we can evaluate the model on both tasks seperately rather than presenting just one unified number
* Table 3: We are going to add another experiment where we train on only the 4 datasets without urbansatseg1960 , effectively removing all exposure to historical imagery
* Table 2: Include other results from other models of other works on the existing datasets for comparison
* Table 2: move the legent of each column one line down so that is right above the numbers. instead of N/A , put the value in the middle of both orig and hist columns


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
