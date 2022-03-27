# Case Seazone Data Analyst

This project encompasses a real data case for a junior data analyst job opportunity at Seazone. 
Seazone uses OTAs (Online Travel Agencies) like Airbnb, Booking and Expedia, as well as its own
website, to find guests interested in booking a short stay - usually lasting less than a month
per reservation. Each property is advertised as a “Listing”. For this analysis, I used two databases
related to the listings and the booked rooms to answer a few business questions. The description of the project can be found in [references/seazone_challenge.pdf](https://github.com/carolmoraescruz/case_seazone/blob/master/references/seazone_challenge.pdf) and the final report is the file [reports/final_report.pdf](https://github.com/carolmoraescruz/case_seazone/blob/master/reports/final_report.pdf).

## Installation

The project requires Python 3.7.4 or greater. It can be installed directly from GitHub. 

For Windows:
```bash
git clone https://github.com/carolmoraescruz/case_seazone.git
cd case_seazone
virtualenv venv
venv/Scripts/activate.bat
pip install -r requirements.txt
deactivate
```

For Linux:
```bash
git clone https://github.com/carolmoraescruz/case_seazone.git
cd case_seazone
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
deactivate
```

## Running Application

```bash
source venv/bin/activate
python main.py
```

## Project Organization

    ├── LICENSE
    ├── README.md
    ├── main.py
    ├── data
    │   ├── external       
    │   ├── interim        
    │   ├── processed      
    │   └── raw
    |       ├── daily_revenue.csv             
    │       └── listings-challenge.csv            
    ├── docs               
    ├── models             
    ├── references               
    │   └── seazone_challenge.pdf
    ├── reports
    │   ├── figures            
    │   └── final_report.pdf        
    ├── requirements.txt
    └── src                
        ├── __init__.py
        ├── commons.py    
        ├── data           
        │   └── make_dataset.py
        ├── features       
        │   └── build_features.py
        ├── models         
        │   ├── preprocessing.py
        │   └── train_model.py
        ├── reports       
        │   └── reports.py
        └── visualization  
            └── visualize.py
    


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
