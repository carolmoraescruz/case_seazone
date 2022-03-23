import holidays

BRAZILIAN_HOLIDAYS = holidays.Brazil()


def is_holiday(date):
    return int(date in BRAZILIAN_HOLIDAYS)
