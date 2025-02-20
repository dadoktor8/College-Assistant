# College-Assistant

A College-Assistant tool powered by deepseek.
Currently deep seek has been finetuned to summarize simple paragraphs, acheived through prompt engineering and regex magic :)!

## Future plans:

```
---

1. Implement a RAG Pipeline for document summariztion.
2. Increase the performance of the model in order to output faster.
3. Clean the website.
4. Better structure the file-system
```

## How to use:

```
1. In main.py add the directory for your deepseek model --> Currently only using gguf format --> DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf
2. Config the prompt as required
3. You are good to go!
```

## Project Organization

```
College-Assistant/
├── LICENSE
├── README.md
├── Makefile                     # Makefile with commands like `make data` or `make train`
├── configs                      # Config files (models and training hyperparameters)
│   └── model1.yaml
│
├── data
│   ├── external                 # Data from third party sources.
│   ├── interim                  # Intermediate data that has been transformed.
│   ├── processed                # The final, canonical data sets for modeling.
│   └── raw                      # The original, immutable data dump.
│
├── docs                         # Project documentation.
│
├── models                       # Trained and serialized models.
│
├── notebooks                    # Jupyter notebooks.
│
├── references                   # Data dictionaries, manuals, and all other explanatory materials.
│
├── reports                      # Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures                  # Generated graphics and figures to be used in reporting.
│
├── requirements.txt             # The requirements file for reproducing the analysis environment.
└── src                          # Source code for use in this project.
    ├── __init__.py              # Makes src a Python module.
    │
    ├── data                     # Data engineering scripts.
    │   ├── build_features.py
    │   ├── cleaning.py
    │   ├── ingestion.py
    │   ├── labeling.py
    │   ├── splitting.py
    │   └── validation.py
    │
    ├── models                   # ML model engineering (a folder for each model).
    │   └── model1
    │       ├── dataloader.py
    │       ├── hyperparameters_tuning.py
    │       ├── model.py
    │       ├── predict.py
    │       ├── preprocessing.py
    │       └── train.py
    │
    └── visualization        # Scripts to create exploratory and results oriented visualizations.
        ├── evaluation.py
        └── exploration.py
```
