import datetime
import json
from urllib import parse
import pandas as pd
import regex as re
import sqlalchemy
from dateutil.relativedelta import relativedelta


# Date Functions

def coerce_str_len(string, out_len = 2, fill = '0', fill_dir = 'left'):
    """
    Coerces a string to a determined length.
    :param str string: String to be coerced.
    :param int out_len: Desired output length of string.
    :param str fill: String used to fill output string.
    :param str fill_dir: Direction to fill string. Possible values: {'left', 'right'}
    :return: A string of length out_len.
    :rtype: str
    """
    filler = [fill for i in range(out_len - len(string))]
    filler = ''.join(filler)

    if fill_dir == 'left':
        newstr = filler + string
    elif fill_dir == 'right':
        newstr = string + filler

    else:
        raise ValueError('fill_dir debe ser uno de los siguientes valores: {left, right}')

    return newstr


datedict = {
    'jan': 1,
    'feb': 2,
    'mar': 3,
    'apr': 4,
    'may': 5,
    'jun': 6,
    'jul': 7,
    'aug': 8,
    'sep': 9,
    'oct': 10,
    'nov': 11,
    'dec': 12,
    'january': 1,
    'february': 2,
    'march': 3,
    'april': 4,
    'may': 5,
    'june': 6,
    'july': 7,
    'august': 8,
    'september': 9,
    'october': 10,
    'november': 11,
    'december': 12,
    'ene': 1,
    'feb': 2,
    'mar': 3,
    'abr': 4,
    'may': 5,
    'jun': 6,
    'jul': 7,
    'ago': 8,
    'sep': 9,
    'oct': 10,
    'nov': 11,
    'dic': 12,
    'enero': 1,
    'febrero': 2,
    'marzo': 3,
    'abril': 4,
    'mayo': 5,
    'junio': 6,
    'julio': 7,
    'agosto': 8,
    'septiembre': 9,
    'octubre': 10,
    'noviembre': 11,
    'diciembre': 12
}


def fill_datepart(datestr, datepart):
    """
    Standardizes a date part for YYYY-MM-DD format.
    :param str datestr: String to standardize
    :param str datepart: Part of date datestr represents. Possible values: {'day', 'month', 'year'}
    :return: A standard two-digit representation of the original date string.
    :rtype: str
    """
    datestr = str(datestr)

    if datepart == 'month':
        if datestr in datedict.keys():
            datestr = str(datedict[datestr])

        datestr = coerce_str_len(datestr)
        # datestr = '0' + datestr
        # datestr = datestr[-2:]

    elif datepart == 'day':
        datestr = coerce_str_len(datestr)
        # datestr = '0' + datestr
        # datestr = datestr[-2:]

    return datestr


def fill_dateparts(datelist, format = '%Y%m%d'):
    """
    Standardizes every element of a list of date parts for YYYY-MM-DD format.
    :param datelist: List or iterable of date parts to standardize
    :param str format: String specifying the order and format of the date parts
    :return: A list of standardized date parts
    :rtype: list
    """
    if format == '%Y%m%d':

        datelist[0] = fill_datepart(datelist[0], 'year')
        datelist[1] = fill_datepart(datelist[1], 'month')
        datelist[2] = fill_datepart(datelist[2], 'day')

    elif format == '%m%d%Y':

        datelist[2] = fill_datepart(datelist[2], 'year')
        datelist[0] = fill_datepart(datelist[0], 'month')
        datelist[1] = fill_datepart(datelist[1], 'day')

    elif format == '%d%m%Y':
        datelist[2] = fill_datepart(datelist[2], 'year')
        datelist[1] = fill_datepart(datelist[1], 'month')
        datelist[0] = fill_datepart(datelist[0], 'day')

    elif format == '%Y%d%m':
        datelist[0] = fill_datepart(datelist[0], 'year')
        datelist[2] = fill_datepart(datelist[2], 'month')
        datelist[1] = fill_datepart(datelist[1], 'day')

    elif format == '%Y%m':
        datelist[0] = fill_datepart(datelist[0], 'year')
        datelist[1] = fill_datepart(datelist[1], 'month')

    elif format == '%m%Y':
        datelist[0] = fill_datepart(datelist[0], 'month')
        datelist[1] = fill_datepart(datelist[1], 'year')

    return datelist


def date_from_str(datestr, return_datetime = False, format = '%Y%m%d'):
    """
    Obtains a date/datetime from a string.
    :param str datestr: String to be parsed as date
    :param bool return_datetime: Should date be returned in datetime format?
    :param str format: Format in which datestr should be interpreted
    :return: A date or datetime object interpreted from datestr
    :rtype: date or datetime
    """
    datestr = str(datestr).lower()

    datelist = re.findall('[a-z0-9]+', datestr)

    if len(datelist) > 1:
        datelist = fill_dateparts(datelist, format)

    date = datetime.datetime.strptime(''.join(datelist), format)
    if not return_datetime:
        date = date.date()

    return date


def ymd(datestr, return_datetime = False):
    """
    Parse dates in year, month, day format.
    :param str datestr: String to be parsed as date
    :param bool return_datetime: Should date be returned in datetime format?
    :return: A date or datetime object interpreted from datestr
    :rtype: date or datetime
    """
    return date_from_str(datestr, return_datetime = return_datetime, format = '%Y%m%d')


def mdy(datestr, return_datetime = False):
    """
    Parse dates in month, day, year format.
    :param str datestr: String to be parsed as date
    :param bool return_datetime: Should date be returned in datetime format?
    :return: A date or datetime object interpreted from datestr
    :rtype: date or datetime
    """
    return date_from_str(datestr, return_datetime = return_datetime, format = '%m%d%Y')


def dmy(datestr, return_datetime = False):
    """
    Parse dates in day, month, year format.
    :param str datestr: String to be parsed as date
    :param bool return_datetime: Should date be returned in datetime format?
    :return: A date or datetime object interpreted from datestr
    :rtype: date or datetime
    """
    return date_from_str(datestr, return_datetime = return_datetime, format = '%d%m%Y')


def ydm(datestr, return_datetime = False):
    """
    Parse dates in year, day, month format.
    :param str datestr: String to be parsed as date
    :param bool return_datetime: Should date be returned in datetime format?
    :return: A date or datetime object interpreted from datestr
    :rtype: date or datetime
    """
    return date_from_str(datestr, return_datetime = return_datetime, format = '%Y%d%m')


def ym(datestr, return_datetime = False):
    """
    Parse dates in year, month format. Assumes day = 1.
    :param str datestr: String to be parsed as date
    :param bool return_datetime: Should date be returned in datetime format?
    :return: A date or datetime object interpreted from datestr
    :rtype: date or datetime
    """
    return date_from_str(datestr, return_datetime = return_datetime, format='%Y%m')


def my(datestr, return_datetime = False):
    """
    Parse dates in month, year format. Assumes day = 1
    :param str datestr: String to be parsed as date
    :param bool return_datetime: Should date be returned in datetime format?
    :return: A date or datetime object interpreted from datestr
    :rtype: date or datetime
    """
    return date_from_str(datestr, return_datetime = return_datetime, format='%m%Y')


def eomonth(date, n = 0):
    """
    Calculates the date of the last day of the month that is n months away from the provided date.
    :param date date: Reference date
    :param int n: Number of months before (n < 0) or after (n > 0) reference date. If n = 0, the last day of the month
    of the provided date is returned
    :return: Date of the last day of the month n months away from reference date
    :rtype: date
    """
    return date + relativedelta(months = n) + relativedelta(day = 31)


def date_add_day(date, n = 1, weekdays = False, holidays = []):
    """
    Calculates the date that is n days away from the provided date, with the option of omitting weekends.
    :param date date: Reference date
    :param int n: Number of days before (n < 0) or after (n > 0) reference date
    :param bool weekdays: Whether to omit weekends or not when calculating the date
    :param list holidays: List containing the dates of holidays to omit when calculating the date
    :return: Date n days away from the provided date
    :rtype: date
    """
    if len(holidays) > 0:
        for i in range(len(holidays)):
            holidays[i] = ymd(holidays[i])

    if n >= 0:
        sign = 1
    else:
        sign = -1

    if weekdays:
        for i in range(abs(n)):
            date = date + (sign * relativedelta(days = 1))
            while date.weekday() in [5, 6] or date in holidays:
                date = date + (sign * relativedelta(days = 1))
    else:
        for i in range(abs(n)):
            date = date + sign * relativedelta(days=1)

    return date


def date_add_weekday(date, n = 1, holidays = []):
    """
    Add or subtract weekdays to a given date, considering specified holidays.

    :param date date: The input date to be modified
    :param int n: The number of weekdays to add or subtract (positive values add days, negative values subtract days)
    :param list[str] holidays: A list of holiday dates as strings in the format 'YYYY-MM-DD' to be skipped when
    adding or subtracting days
    :return: The resulting date after adding or subtracting the specified number of weekdays, considering the provided
    holidays
    :rtype: date
    """
    return date_add_day(date, n = n, weekdays = True, holidays = holidays)


def yearsbetween(startdate, enddate):
    """
    Calculates the number of years between two dates.
    :param date startdate: Start date
    :param date enddate: End date
    :return: The number of years elapsed between the two dates
    :rtype: int
    """
    r = relativedelta(enddate, startdate)
    return r.years


def monthsbetween(startdate, enddate):
    """
    Calculates the number of months between two dates.
    :param date startdate: Start date
    :param date enddate: End date
    :return: The number of months elapsed between the two dates
    :rtype: int
    """
    r = relativedelta(enddate, startdate)
    return r.years * 12 + r.months


def daysbetween(startdate, enddate):
    """
    Calculates the number of days between two dates.
    :param date startdate: Start date
    :param date enddate: End date
    :return: The number of days elapsed between the two dates
    :rtype: int
    """
    # r = relativedelta(enddate, startdate)
    r = enddate - startdate
    return r.days


def date_to_str(date, format = '%Y-%m-%d'):
    """
    Returns a string representation of the provided date
    :param  date date: Reference date
    :param str format: A string containing the desired output format of the provided date
    :return: A string representation of the provided date
    :rtype: str
    """
    return datetime.datetime.strftime(date, format)


# Importing / Exporting Data funcs

def load_params(file = 'params.JSON'):
    """
    Imports a params.JSON file.
    :param str file: Name / path of file
    :return: A dictionary containing the parameters imported from the file
    :rtype: dict
    """
    with open(file, 'r') as f:
        params = json.load(f)
    f.close()

    return params


def connect_sql(driver = 'SQL Server', server = 'BMCSD230', database = 'RiesgosBMC', fast_executemany = True):
    """
    Establishes a connection to a SQL Server Database
    :param str driver: Driver to be used to establish the connection
    :param str server: Server to connect to
    :param str database: Database to connect to
    :param bool fast_executemany: sqlalchemy's fast_executemany parameter. Speeds up querying
    :return: An sqlalchemy connection engine.
    :rtype: sqlalchemy.engine
    """
    engine = sqlalchemy.create_engine(
        'mssql+pyodbc:///?odbc_connect={}'.format(
            parse.quote_plus(
                'Driver={' + driver + '};'
                'Server=' + server + ';' 
                'Database=' + database + ';'
                'Trusted_Connection=yes;')
        ),
        fast_executemany = fast_executemany
    )

    return engine


def get_from_sql(engine, query, close_con = True, **kwargs):
    """
    Get data from SQL
    :param engine: An sqlalchemy.engine object.
    :param str query: Query to use to obtain data from the database
    :param bool close_con: Whether to close the connection after obtaining the data
    :param kwargs: Keyword arguments to the pandas.read_sql_query function
    :return: A Pandas dataframe containing the queried data
    :rtype: pandas.DataFrame
    """
    df = pd.read_sql_query(query, engine, **kwargs)
    if close_con:
        engine.dispose()

    return df


def df_to_sql(
        df,
        engine,
        dest_table,
        if_exists = 'fail',
        schema = 'temp',
        index = False,
        close_con = True,
        **kwargs
):
    """
    Write a dataframe to a SQL database
    :param df: A pandas.DataFrame
    :param engine: An sqlalchemy.engine object
    :param str dest_table: The table to write the data to
    :param str if_exists: How to behave if the table already exists. Possible values: {'fail', 'replace', 'append'}
        'fail': Raise a ValueError.
        'replace': Drop the table before inserting new values.
        'append': Insert new values to the existing table.
    :param str schema: Schema to which to write the table
    :param bool index: Whether to write the dataframe's index as a column names index_label
    :param bool close_con: Whether to close the connection after writing the table
    :return: None
    """
    df.to_sql(dest_table, engine, if_exists = if_exists, index = index, schema = schema, **kwargs)
    if close_con:
        engine.dispose()


def infer_ext(path):
    """
    Infers the extension given a file name.
    :param str path: Path to the file.
    :return: The inferred file extension
    :rtype: str
    """
    return path.split(sep = '.')[-1]


def get_from_dict(dictionary, key, if_fail = ''):
    """
    Get the value for a specified key from a dictionary. Return a default value if the key is not present.

    :param dict dictionary: The dictionary to retrieve the value from
    :param key: The key to look for in the dictionary
    :type key: str or int
    :param if_fail: The default value to return if the key is not found in the dictionary
    :type if_fail: Any
    :return: The value corresponding to the provided key in the dictionary or the default value if the key is not found
    :rtype: Any
    """
    if key in dictionary.keys():
        return dictionary[key]

    else:
        return if_fail


def df_to_fwf(df, filename, file_lengths):
    """
    Write a DataFrame to a fixed-width formatted (FWF) text file.

    Parameters:
    - df: pandas DataFrame
        The DataFrame to be written to the file.
    - filename: str
        The name of the file to be created or overwritten.
    - file_lengths: list of int or str
        A list specifying the width of each column in the output file.
        If an element is an integer, it represents the fixed width for the corresponding column.
        If an element is a string containing a dot ('.'), it indicates a float column with decimal places.
        The integer part represents the width, and the decimal part represents the number of digits after the decimal point.
    """
    with open(filename, 'w', encoding='latin1') as f:
        for _, row in df.iterrows():
            line = ''
            for col, length in zip(row, file_lengths):
                if '.' in str(length):  # Check if length is a float with decimal places
                    split_length = length.split('.')
                    length = int(split_length[0])
                    digits = int(split_length[1])
                    col = round(col, digits)
                else:
                    length = int(length)

                # Format the column value and append it to the line
                line += str(col).ljust(length)

            # Write the formatted line to the file
            f.write(line + '\n')

    # Close the file after all lines have been written
    f.close()
