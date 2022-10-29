# pokemon_trainer_hiring_evaluation

## Tools used in this project
* [Poetry]

## Project structure
```bash
.
├── data            
│   ├── processed                   # data after processing
│   ├── raw                         # raw data
├── .gitignore                      # ignore files that cannot commit to Git
├── notebooks                       # store notebooks
├── pyproject.toml                  # dependencies for poetry
├── README.md                       # project description
├── src                             # store source code
│   ├── __init__.py                 # make src a Python module 
│   ├── evaluate.py                 # evaluate model performance 
│   ├── process.py                  # process data before training model
│   ├── train_model.py              # train model
│   └── utils.py                    # utils
└── tests                           # store tests
    ├── __init__.py                 # make tests a Python module 
    └── test_utils.py               # tests for utils.py

```

## Set up the environment
```
poetry install
```

## Run model
```
python main.py
```