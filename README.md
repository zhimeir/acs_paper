# acs_paper
This repository contains the code for reproducing the numerical results in [ACS: An interactive framework for conformal selection](https://arxiv.org/pdf/2507.15825).

## Simulations
The code for replicating the simulation results is in the `simulation` folder, and has been tested in Python 3.11.4. 
The scripts are written to carry out experiments with different configurations and random seeds, indexed by `task_id`.
For example, `task_id=1` refers to the experiment under Setting 1, noise level 0.3, seed group 1, base learner SVR, calibration size 200.  
- To carry out one run (with `task id=1`) of the experiments with base learners (Section 5.2), run the following code in the terminal under `/acs_paper`:
```
python3 base_learner.py 1
```
- To carry out one run (with `task id=1`) of the experiments with adaptive model selection (Section 5.3), run the following code in the terminal under `/acs_paper`:
```
python3 model_selection.py 1
```
- To carry out one run (with `task id=1`) of the experiments with extra labels (Section 5.4), run the following code in the terminal under `/acs_paper`:
```
python3 extra_label.py 1
```
- To carry out one run (with `task id=1`) of the experiments with diversity-aware selection (Section 5.5), run the following code in the terminal under `/acs_paper`:
```
python3 diversity.py 1
```
- To carry out one run (with `task id=1`) of the experiments investigating the effece of calibration data size (Appendix C), run the following code in the terminal under `/acs_paper`:
```
python3 calib_size.py 1
```
The results can be found in the `results` folder.


## Real data application: LLM deployment

All reproducible code for Section 6 lives in the foundation_models folder:

- `/qa`: Section 6.1 Question Answering

- `/cxr`: Section 6.2 Chest X‑Ray Report Generation



### Question answering (```/qa```)

Answer generation and scores extraction follow the implementations in <https://github.com/yugjerry/conformal-alignment> and <https://github.com/zlin7/UQ-NLG>, for which commands can be found in `/qa/run`.

This folder contains the implementation with two base LLMs--[**OPT-13B**](https://huggingface.co/facebook/opt-13b) and [**LLaMA-2-13B-chat**](https://llama.meta.com/llama-downloads/)--on two QA datasets--[**TriviaQA**](https://nlp.cs.washington.edu/triviaqa/) and [**CoQA**](https://stanfordnlp.github.io/coqa/).

- To reproduce the results in Figure 9, run the following code under `/qa/run`:

```
cd qa/run
python3 ../fdr_ada_aug.py --task_id $id
```

Here `id` specifies the index of the experiment configuration. For example, `id=1` refers to the setting with `model='llama-2-13b-chat-hf'`, `data='triviaqa`, `N=100`, and `split=1`, where `split/10` controls the splitting ratio of the training set (see Experimental setup in Section 6).



- To reproduce the results in Figure 10, run the following code under `/qa/run`:

```
cd qa/run
python3 ../fdr_ada_ensemble.py --task_id $id
```

## Chest X-ray report generation (```/cxr```)

Chest X-ray (CXR) report generation and scores extraction follow the implementation in <https://github.com/yugjerry/conformal-alignment>  and commands can be also found in `/cxr/run`.

For Vision-language model fine-tuning, we follow ```cxr/vlm_finetune.ipynb```, in which we use a [**Vision Transformer** (ViT) pretrained on ImageNet-21k](https://huggingface.co/google/vit-base-patch16-224-in21k) as the image encoder and **GPT2** as the text decoder.

[**MIMIC-CXR**](https://www.nature.com/articles/s41597-019-0322-0) dataset needs access. See the [PhysioNet project page](https://physionet.org/content/mimic-cxr/2.0.0/).


- To reproduce the results in Figure 11, run the following code under `/cxr/run`:

```
cd cxr/run
python3 ../fdr_ada_aug.py --task_id $id
```

- To reproduce the results in Figure 12, run the following code under `/cxr/run`:

```
cd cxr/run
python3 ../fdr_ada_ensemble.py --task_id $id
```

## Real data application: Diverse drug discovery

Code for reproducing our drug discovery experiments can be found in the folder `drug_discovery`. The notebook `drug_data_preprocess.ipynb` cleans the data and, in particular, generates the file `cleaned_drug_data.csv` as well as the directory (and its contents) `permutations`. The script `diversity_driver.py` can then be run to produce our results; it accepts one integer command line argument, between 0 and 5999 inclusive, which governs the random seed, quantile, and value of alpha at which to to run both ACS (with diversity-aware ordering) and CS. Results are stored in `dti_similarity_results` and the script `collate_results.py` should be used to collate all results (once done running) into CSVs in `csvs_to_plot`.


