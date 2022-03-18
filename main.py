# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import seaborn as sns


def main():
    df_listings = pd.read_csv('data/raw/listings-challenge.csv')
    df_daily_revenue = pd.read_csv('data/raw/daily_revenue.csv')

    print(df_listings.shape)
    print(df_daily_revenue.shape)

if __name__ == '__main__':
    main()
