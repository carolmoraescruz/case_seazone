# Case Seazone Data Analyst

This project encompasses a real data case for a junior data analyst job opportunity at Seazone.
Seazone uses OTAs (Online Travel Agencies) like Airbnb, Booking and Expedia, as well as its own
website, to find guests interested in booking a short stay - usually lasting less than a month
per reservation. Each property is advertised as a “Listing”. For this analysis, I used two databases
related to the listings and the booked rooms to answer a few business questions. The description of the project can be found in [references/seazone_challenge.pdf](https://github.com/carolmoraescruz/case_seazone/blob/master/references/seazone_challenge.pdf) and the final report is the file [reports/final_report.pdf](https://github.com/carolmoraescruz/case_seazone/blob/master/reports/final_report.pdf).


## Running Application

### Basics Dependencies

- The project requires Python 3.7.4 or greater.
- It is assumed that the `virtualenv` library is properly installed. If it is not, just run:

```bash
pip install virtualenv
```

1. The first step is to clone the repository to local machine

```bash
git clone https://github.com/carolmoraescruz/case_seazone.git
```

2. Set a virtual environment to install the requirements in their respective versions

```bash
virtualenv -p python3 venv
```

3. Activate the virtual environment

- Windows

```bash
venv\Scripts\activate.bat
```

- Linux and MacOS

```bash
source . venv/bin/activate
```

4. Install the dependencies, running

```bash
pip install -r requirements.txt
```

5. Run the application:

```bash
python main.py
```

## Screenshot
![Code screenshot](https://github.com/carolmoraescruz/case_seazone/blob/b08ab969f6227e0761db19fc8e614defac2a1e81/reports/figures/code_screenshot.png)

## Project Organization

    ├── LICENSE
    ├── README.md
    ├── main.py
    ├── data
    │   ├── external
    │   ├── interim
    │   ├── processed
    │   └── raw
    |       ├── daily_revenue.csv
    │       └── listings-challenge.csv
    ├── docs
    ├── models
    ├── references
    │   └── seazone_challenge.pdf
    ├── reports
    │   ├── figures
    │   └── final_report.pdf
    ├── requirements.txt
    └── src
        ├── __init__.py
        ├── commons.py
        ├── data
        │   └── make_dataset.py
        ├── features
        │   └── build_features.py
        ├── models
        │   ├── preprocessing.py
        │   └── train_model.py
        ├── reports
        │   └── reports.py
        └── visualization
            └── visualize.py

---

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
