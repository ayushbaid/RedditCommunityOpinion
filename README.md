# RedditCommunityOpinion
Analysing opinion of reddit communities on different topics using NLP

# Setup instructions

The following instructions have been tried on a linux distribution (Ubuntu 19.10 )

- install the conda environment from the file ```environment.yml```
- download the dataset from https://drive.google.com/open?id=1bl_bENMk-QCxRW21e399BlL4SiGpLAuJ and extract into the dataset folder
- configure the phrases in ```config/questions_large.txt```
- Activate the conda environment
- Move to the code directory: ```cd code```
- Install the current module: ```pip install -e .```
- GPT2:
    - For training: run the command ```python helper/train_gpt2.py```
    - For deploying for moderation ```python helper/deploy_gpt2.py```
- ULMFit2:
    - For training: run the command ```python models/ulmfit.py```
    - For deploying: run the command ```python helper/ulmfit_results.py```