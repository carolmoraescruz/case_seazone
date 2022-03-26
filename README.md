# Case Seazone Data Analyst

This project encompasses a real data case for a junior data scientist job opportunity at Seazone. 
Seazone uses OTAs (Online Travel Agencies) like Airbnb, Booking and Expedia, as well as its own
website, to find guests interested in booking a short stay - usually lasting less than a month
per reservation. Each property is advertised as a “Listing”. For this analysis, I used two databases
related to the listings and the booked rooms to answer a few business questions. The description of the project can be found in [references/seazone_challenge.pdf](https://github.com/carolmoraescruz/case_seazone/blob/3b29e43b79367196c2745b76e73c25d4c485761e/references/seazone_challenge.pdf) and the final report is the file [references/final_report.pdf](https://github.com/carolmoraescruz/case_seazone/blob/3b29e43b79367196c2745b76e73c25d4c485761e/references/final_report.pdf)
## Installation

The project requires Python 3.7.4 or greater. It can be installed directly from GitHub. 

```bash
git clone https://github.com/carolmoraescruz/case_seazone.git
cd case_seazone
virtualenv venv --python=python3
source venv/bin/activate
pip install -r requirements.txt
deactivate
```

## Usage

```bash
source venv/bin/activate
python main.py
```


## Project Organization

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    └── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        ├── data           <- Scripts to download or generate data
        │   └── make_dataset.py
        │
        ├── features       <- Scripts to turn raw data into features for modeling
        │   └── build_features.py
        │
        ├── models         <- Scripts to train models and then use trained models to make
        │   │                 predictions
        │   ├── predict_model.py
        │   └── train_model.py
        │
        └── visualization  <- Scripts to create exploratory and results oriented visualizations
            └── visualize.py
    


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
