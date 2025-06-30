# TODO 13/05:

## Done

- Clean up datasets in repo
- Get the rule based generation pipeline running for smaller 150 patches again
- Modify it to run for the full thing, including adding a parameter in case we want partial, no parameter means full. see how long it will take
- Need to do the train test val split. To do that, we use the splits iSAID already has. Inside the dataset folder, the script should create the 3 folders.
- Add multiprocessing using all cores to speed up 1_create_patches.py
- Modify rule_viewer.py in order to view new splits
- Add multiprocessing using all cores to speed up step 2,3,4
- Time print to run_pipeline
- Improve rule_viewer to handle large datasets
- Run full dataset now that we have the multiprocessing 48 cores on all steps
- Create a script on utils for seeing the final metrics of the dataset
- change directory names from dataset to aeriald (created 
- Minor: generate index file, eg json, for the dataset (no need, already split in folders)
- Investigate why it skips the first 60 patches (does it?)
- Why are there less patches than anticipated? (now I know , it because some patches have no objects)
- Add patch visualization to report
- Improvement in size rule: we are not rejecting instances that are cutoff screen, which make them artificially smaller
- Improvement in near center/center uniqueness check: we are treating both as seperate, which is ambiguous in cases where we have instances in both
- Improvement in cutoff condition: best would be to consider cutoff instances but not use them for expressions, avoiding cases where we have 2 of the same instance , one cutoff more than 50 percent, yet we consider the expression for the other instance unique (no easy solution)
- Minor: add rejection of patches that are more than half black pixels (some iSAID images have black borders)
- Minor: consider revisiting 3x3 position grid, adding margins between boundaries that reject position or add "near" qualifier (not sure there is a solution here)
- Improvement: some objects don't get a color, but are very similar to others. this causes ambiguity. Need to calibrate color detection
- Change step 1 in order to use new splits directory structure
- Change step 3 in order to use new splits directory structure
- Feature: group colors rule (not needed)
- Modify app.py for new directory structure
- Improvement: change prompt to focus on mixing up language, to have more varied vocabulary
- Improvement: change schema to generate "unrelated" expressions, that is, attempts of the LLM to generate an unique expression rather than enhancing  rule-based generated ones
- Feature: add unique_reasoning field before the unique expressions output (you know what, lets not do that, when we will have reasoning models that do that by default beforehand, there is really no need.)
- Update schema to remove description before language variations
- Bring back description and reduce temperature (did not reduce temp)
- Improvement: Probably should discard including cutoff objects in masks, but keep them for the rules (lets not do that, easier to just modify the rejection)
- Improvement: Use class specific rejections (consider the non-cut off part, if its larger than a threshold we keep it too)
- Update report with new changes
- Improvement: size rule "largest" fails when a potentially larger object is cutoff from the image (actually solving this breaks other situations and is very rare, so not worth the hassle)
- Improvement: cut off situations still are not very good when the object is small. Need to have a min instance area for cutoff objects (P0036 patch 7)
- Improvement: border ambiguity is a problem with the 3x3 grid. There cannot be any cases of a instance having a different "unique" expression for moving up one pixel
- Improvement: in fact, any borderline situation must be impossible. Under no circumstance will moving the instance one pixel in any direction produce a new unique expression (are these issues really a problem, when the fact instances cannot overlap means they will never be on pixel from each other?)
- Improvement: what if we are very stric with colors? Like verifying if sorrounding instances abosolute are NOT of the same color, playing it safe, etc.  (not needed as see yet)
- rule viewer side by side like app.py
- fix issue with polygon masks not being extracted when they have more than one polygon
- check what ID image was that plane in that was failing. check ID image of the double polygon. regenerate dataset there
- need to set input folder for the dataset in llm pipeline as argument
- Change prompt to generate more natural expressions, give examples in prompt
- For the enhanced expressions, try having it pick a reference rule based expression and expand it rather than starting from the beggining
- zip up full dataset
- move it to cfs
- start training again both
- create a mode that allows me to run rule_viewer.py picking random objects out of the entire dataset
- Make a small test compilation of good/bad samples from the enhanced expressions, to see the signal/noise ratio
- need to be able to create small datasets with random crop selection and seed, with finding a small dataset that has all the problems we mention below
## dataset fixes

- found another problem with cutoff objects: the bounding box of the cutoff object is cutoff, when what we should  be doing it comupting a new bounding box after the object itself is already cutoff, which avoids these big bounding boxes where the object itself is barealy visible
- check that polygons are being done correctly now when outside
- fix the problem with the cutoff objects. using only the annotations of a cutoff object while not using the boundariess
- absolute posisioning: add strict threshold

## Not done


- relative positioning: add strict threshold
- color: add stricter boundaries to avoid color ambiguity


LLM:

- As per the docs, placing the schema in the prompt is detrimental with structured outputs enabled. consider removing it


notes about call:
- start with showing the training run results, the full dataset 2 runs (fusion and clipsam), quantitative and qualitative. 
- then the random expression filter run.
- is filtering the expressions on a object a good idea (to include more signal), and if so, would a random selection be enough, leaving on the annotations all the expressions regardless
- show 40 run manual classification of expressions. show why its improved (prompt for more natural language, and for unique expressions, prompt to go off of a existing rule one)
- about annotations format (is XML good enough since we are loading the expressions directly to memory, or do we need to change it)
- next architecture steps, should i test fusion SAM, even if inital qualitative results dont show great advantage towards fusion encoder? should I try out BEIT-3 architecture, similar to Francisco?