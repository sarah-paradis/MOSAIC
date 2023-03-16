# -*- coding: utf-8 -*-
"""
Created on Monday 22nd of March 2021

@author: Sarah Paradis

Python module to conduct a quality check of data prior to adding it to the MOSAIC database

1. Open Excel or other spreadsheet

2. Match column names with variables names.

3. Check the metadata. Log missing data as warnings.

4. Check that data is within limits for each variables. Log warnings.

5. Save checked data in new folder
"""
import numbers
import warnings
import numpy as np
import pandas as pd
import os
from crossref_commons.retrieval import get_entity
from crossref_commons.types import EntityType, OutputType
from pangaeapy.pandataset import PanDataSet
import logging
import sys
import datetime
from varname import nameof
import geopandas as gpd
from shapely.geometry import Point
import shapely.speedups
import rasterio
import shutil
from quality_check_config import locator_information_dict, dict_core_range, dict_core_analyses_range, \
    dict_sample_analyses_range, dict_author_metadata_range, locator_information_dict_new
from mosaic.general_funcs.database_connect import connection, database
from mosaic.general_funcs.general import print_with_time, columns_in_database

logger = logging.root
logger.setLevel(logging.INFO)

if not logger.hasHandlers():
    # Set the logging file
    file_handler = logging.FileHandler('quality_check.log')
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s",
                                                datefmt="%d-%b-%y %H:%M:%S"))  # Data to be logged
    file_handler.setLevel(logging.INFO)  # Save all the "info" messages into the log
    logger.addHandler(file_handler)

    # Set the stream handler (console messages)
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s",
                                                  datefmt="%d-%b-%y %H:%M:%S"))
    stream_handler.setLevel(logging.WARNING)  # Print out only warning and error logs into the console
    logger.addHandler(stream_handler)


# 1. Open Excel or other spreadsheet
def read_df(file, article_author_metadata=None, geopoints_cores=None,
            core_analyses=None, sample_analyses=None, file_dir=None):
    """
    Reads a file (either excel or open office) as a DataFrame
    If several files need to be opened, specify the directory where they are stored
    Specify the sheet names where data is stored.
    Can only read Excel and OpenOffice files in the format of the template! (with the specified sheetnames)

    :returns dataframes for article_author, geopoints_cores, core_analyses, sample_analyses
    """
    if file_dir is not None:
        file = os.path.join(file_dir, file)
    if file.endswith('.xlsx') or file.endswith('.xls') or file.endswith('.ods'):
        with print_with_time('Opening file ' + file):
            df_lists = []
            if article_author_metadata:
                df_article_author = _read_article_author_metadata(file, article_author_metadata)
                df_lists.append(df_article_author)
            if geopoints_cores:
                df_geopoints = _read_geopoints(file, geopoints_cores)
                df_lists.append(df_geopoints)
            if core_analyses:
                df_core_analyses = _read_core_analyses(file, core_analyses)
                df_lists.append(df_core_analyses)
            if sample_analyses:
                df_sample_analyses = _read_sample_analyses(file, sample_analyses)
                df_lists.append(df_sample_analyses)
    else:
        raise TypeError('File not recognized as a .xls .xlsx or .ods')
    for df in df_lists:
        df.replace(r'[^\x00-\x7F]+', " ", regex=True, inplace=True)
    logger.info("\n")  # New empty line every time a new document is opened. Simplifies reading the log file.
    return df_lists


def _read_article_author_metadata(file, article_author_metadata):
    if article_author_metadata:
        with warnings.catch_warnings():
            # UserWarning is warning that the Excel sheet has several dtaa validations. Ignore
            warnings.simplefilter("ignore", category=UserWarning)
            df_article_author = pd.read_excel(file, sheet_name=article_author_metadata, dtype=str,
                                              skiprows=[0])
            df_article_author.dropna(how='all', inplace=True)
    else:
        df_article_author = None
    return df_article_author


def _read_geopoints(file, geopoints_cores):
    if geopoints_cores:
        with warnings.catch_warnings():
            # UserWarning is warning that the Excel sheet has several data validations. Ignore
            warnings.simplefilter("ignore", category=UserWarning)
            df_geopoints = pd.read_excel(file, sheet_name=geopoints_cores, skiprows=[0],
                                         dtype={'core_name': str,
                                                'sampling_year': 'Int64',
                                                'sampling_month': 'Int64',
                                                'sampling_day': 'Int64'})
            #            df_geopoints.dropna(axis='columns', how='all', inplace=True)
            df_geopoints.dropna(how='all', inplace=True)
        return df_geopoints
    else:
        raise Exception(f'Missing geopoints data from {file}')


def _read_core_analyses(file, core_analyses):
    if core_analyses:
        with warnings.catch_warnings():
            # UserWarning is warning that the Excel sheet has several dtaa validations. Ignore
            warnings.simplefilter("ignore", category=UserWarning)
            df_core_analyses = pd.read_excel(file, sheet_name=core_analyses, skiprows=[0], dtype={'core_name': str})
            if not df_core_analyses.empty:
                arrays = [df_core_analyses.columns.to_list(),
                          df_core_analyses.iloc[0, :].to_list(),
                          df_core_analyses.iloc[1, :].to_list()]
                df_core_analyses.columns = pd.MultiIndex.from_arrays(arrays)
                df_core_analyses.drop(index=[0, 1], inplace=True)
                df_core_analyses.dropna(axis='columns', how='all', inplace=True)
                df_core_analyses.dropna(how='all', inplace=True)
                df_core_analyses.reset_index(drop=True, inplace=True)
    else:
        df_core_analyses = None
    return df_core_analyses


def _read_sample_analyses(file, sample_analyses):
    if sample_analyses:
        with warnings.catch_warnings():
            # UserWarning is warning that the Excel sheet has several dtaa validations. Ignore
            warnings.simplefilter("ignore", category=UserWarning)
            df_sample_analyses = pd.read_excel(file, sheet_name=sample_analyses, skiprows=[0],
                                               dtype={'core_name': str})
            if not df_sample_analyses.empty:
                arrays = [df_sample_analyses.columns.to_list(),
                          df_sample_analyses.iloc[0, :].to_list(),
                          df_sample_analyses.iloc[1, :].to_list(),
                          df_sample_analyses.iloc[2, :].to_list(),
                          df_sample_analyses.iloc[3, :].to_list()]
                df_sample_analyses.columns = pd.MultiIndex.from_arrays(arrays)
                df_sample_analyses.drop(index=[0, 1, 2, 3], inplace=True)
                df_sample_analyses.dropna(axis='columns', how='all', inplace=True)
                df_sample_analyses.dropna(how='all', inplace=True)
                df_sample_analyses.reset_index(drop=True, inplace=True)
                df_sample_analyses.columns.rename(['column', 'method', 'method_details',
                                                   'material_analyzed', 'raw_calculated'], inplace=True)
                df_sample_analyses.exclusivity_clause = df_sample_analyses.exclusivity_clause.astype(bool)
    else:
        df_sample_analyses = None
    return df_sample_analyses


# 2. Match column names with variables names.
def match_variables(conn=connection):
    """Matches the column names of the excel file for the Database column names"""
    # Do a list of all the variables names of the file to be imported
    # Extract column names from database
    cursor = conn.cursor()
    cursor.execute(f"SHOW tables FROM {database}")
    db_tables = [db_table[0] for db_table in cursor.fetchall()]  # Extract all the database table names into a list
    db_columns = []
    for db_table in db_tables:
        cursor.execute(f"SHOW columns FROM {db_table}")
        # Extract all the database column names of one table into a list
        db_columns_temp = [db_column[0] for db_column in cursor.fetchall()]
        db_columns.extend(db_columns_temp)  # Combine all the database column names into one list
    # Open tkinter GUI of both lists, put in asterisks the required fields

    # Click on matching column names to create a dictionary of each consecutive click
    # {file_column: db_column}

    # for column in df.columns.values:
    #     if column not in db_columns:
    #         print('not in columns')
    # column_dict = dict(zip(df.columns.values, list_new_columns))
    # df.rename(columns={column_dict}, inplace=True)
    # logger.info(f'Successfully matched {df.shape[1]} columns of all data')
    # return df


# 3. Check the metadata. Log missing data as warnings.
def check_metadata_multiple_df(filename, df_list, required_fields_list, duplicate_row_in_column_list=None):
    """

    :param filename:
    :param df_list: List of DataFrames to be checked.
    :param required_fields_list: List of required fields for each DataFrame (list of lists)
    :param duplicate_row_in_column_list: List of column (or columns) that can not have duplicate entries for each
                                    DataFrame
    """
    assert isinstance(df_list, list) and isinstance(required_fields_list, list), \
        TypeError(f'Variables df_list and required_field_list are not lists')
    assert len(df_list) == len(required_fields_list), \
        ValueError(f'Variables df_list and required_fields_list do not have the same length')
    for i in range(len(df_list)):
        check_metadata(filename=filename, df=df_list[i], required_fields=required_fields_list[i], exceptions=None)
    if duplicate_row_in_column_list:
        assert len(df_list) == len(duplicate_row_in_column_list), \
            ValueError(f'Variables df_list and required_fields_list do not have the same length')
        for i in range(len(df_list)):
            check_metadata(filename=filename, df=df_list[i], required_fields=required_fields_list[i],
                           exceptions=duplicate_row_in_column_list[i])


def check_metadata(filename: str, df: pd.DataFrame, required_fields: list, exceptions=None,
                   duplicate_row_in_column=None):
    """
    Checks metadata of a Dataframe:
    Checks that all the required fields are not empty.
    Checks that the column names are in the database, and if there is a missing column not found in the database.
    Checks that there are no duplicate column names (data entered twice).
    If relevant, checks if a column has duplicate entries.
    :param filename: Name of the filename
    :param df: Dataframe to be checked
    :param required_fields: Required columns that need to be filled. Elements in a list
    are AND and elements in a tuple are OR
    :param exceptions: Columns that do not need to be checked if they are in the database
    :param duplicate_row_in_column: Column that can not have duplicate values (rows)
    """
    if not df.empty:
        # Check if there are duplicate columns in the filename
        _check_duplicate_columns(filename=filename, df=df)
        # Check that all the columns are in the database.
        _check_columns_database(filename=filename, exceptions=exceptions, df=df)
        # Check if all the required fields are filled (not empty)
        _check_metadata_required_fields(filename=filename, df=df, required_fields=required_fields)
        # Check if there are duplicate entries in column
        if duplicate_row_in_column:
            _check_duplicate_rows(filename=filename, df=df, column_name=duplicate_row_in_column)


def _check_metadata_required_fields(filename, df, required_fields):
    """

    :param filename:
    :param df: Dataframe to be checked.
    :param required_fields: list of required fields. If there is a subset where at least one of the elements is
                            required, provided it as a tuple
    :return:
    """
    assert isinstance(filename, str) or isinstance(df, pd.DataFrame) or isinstance(required_fields, list), \
        TypeError(f'Incorrect data type of filename, df, or required_fields')
    with print_with_time(f'Checking that {required_fields} are in {filename}'):
        for i, row in df.iterrows():
            # Extract columns name that don't have NaN values
            if row.index.nlevels == 1:
                columns = [column for column in row.index.get_level_values(0) if pd.notna(row[column])]
            else:
                columns = [column for column in row.index.get_level_values(0) if pd.notna(row[column][0])]
            # Assert that the columns that don't have NaN values are in the required fields
            assert _evaluate_required_fields(columns, required_fields), \
                LookupError(
                    f'Row {i + df.columns.nlevels + 2} has missing required variables {required_fields} in {filename}')


def _check_required_field(filename, df, required_field):
    assert required_field in df.columns, ValueError(f'{required_field} is not in columns of {nameof(df)}')
    cnt = 0
    for i, data in df[required_field].iteritems():
        if pd.isna(data):  # Check if each row has the required variables
            logger.warning(f'Missing required variables {required_field} for row {i} of {nameof(df)} in {filename}.')
        else:
            cnt += 1
    if cnt == df.shape[0]:
        logger.info(f'Successfully checked {cnt}/{df.shape[0]} rows of {required_field} from {nameof(df)} '
                    f'in {filename}.')
    else:
        logger.warning(f'Successfully checked {cnt}/{df.shape[0]} rows of {required_field} from {nameof(df)} in '
                       f'{filename}.\n {df.shape[0] - cnt} rows have missing {required_field} data.')
    return cnt


def _check_duplicate_columns(filename, df):
    assert isinstance(filename, str) or isinstance(df, pd.DataFrame), \
        TypeError(f'Incorrect data type of filename or df')
    with print_with_time(f'Checking for duplicate columns in {filename}'):
        # When opening an Excel/csv as a DataFrame, any duplicate column header will be labeled as "duplicated_column".N
        # In order to check if there is a duplicate column in each DataFrame, first extract all columns that have a '.'
        if df.columns.nlevels == 1:
            possible_duplicates = [column for column in df.columns if '.' in column]
            rename_mapper = {possible_duplicate: possible_duplicate.split('.')[0] for possible_duplicate in
                             possible_duplicates}
            df.rename(columns=rename_mapper, inplace=True)
        else:
            possible_duplicates = [column for column in df.columns.get_level_values(0) if '.' in column]
            rename_mapper = {possible_duplicate: possible_duplicate.split('.')[0] for possible_duplicate in
                             possible_duplicates}
            df.rename(columns=rename_mapper, inplace=True)
        duplicate_columns = df.columns[df.columns.duplicated()].to_list()
        if duplicate_columns:
            logger.warning(f'Duplicate columns {duplicate_columns} detected in filename {filename}')
            raise ValueError(f'Duplicate columns {duplicate_columns} detected in filename {filename}')


def _check_columns_database(filename, df, exceptions=None):
    if exceptions is None:
        exceptions = []
    assert isinstance(filename, str) or isinstance(df, pd.DataFrame), \
        TypeError(f'Incorrect data type of filename or df')
    with print_with_time(f'Checking if columns in {filename} are in the Database'):
        # Do a list of all the variables names of the file to be imported.
        file_columns = df.columns.get_level_values(0).to_list()  # Returns column names even if it is a MultiIndex
        # Extract column names from database
        db_columns = columns_in_database()
        # Check if columns in the DataFrame are in the database
        missing_columns = [file_column for file_column in file_columns if file_column not in db_columns and
                           file_column not in exceptions]
        assert not missing_columns, \
            KeyError(f'Columns {missing_columns} of from filename {filename} are not in the database.')


def _check_duplicate_rows(filename, df, column_name):
    with print_with_time(f'Checking if {column_name} has duplicate entries in {nameof(df)} of {filename}'):
        assert column_name in df.columns, \
            ValueError(f'Variable column_names {column_name} is in DataFrame {nameof(df)}')
        duplicated_rows = df[column_name][df[column_name].duplicated()]
        if not duplicated_rows.empty:
            logger.warning(f'{column_name} has duplicate entries in {nameof(df)} of {filename}.')
            raise ValueError(f'{column_name} has duplicate entries in {nameof(df)} of {filename}.')


def check_article_author_data(filename, df_article_author, doi='doi', author_lastname='author_lastname',
                              author_firstname='author_firstname', dfs_replace_values=None):
    """
    Check that each data entry has a doi (searchable either in Cross-ref or PANGAEA) or author information.
    Prints out warnings in the log file when there is missing data.

    :param dfs_replace_values:
    :param filename: filename that is being checked
    :param df_article_author: pandas dataframe holding the data
    :param doi: column name with the doi information
    :param author_lastname: column name with author's lastname
    :param author_firstname: column name with author's firstname
    :return: returns True if all data is correct (has DOI or author information), or False if there is missing
    information in a row
    """
    cnt = 0
    with print_with_time(f'Checking that article and author metadata is available in {filename} \n'):
        if not df_article_author.empty:
            for i, row in df_article_author.iterrows():
                # Check if there is an entry_user_name given for that value
                assert pd.notna(row['entry_user_name']), \
                    ValueError(f'No data provided of entry_user_name for row {i} in {filename}')
                if doi not in row or pd.isna(row[doi]):  # Check if each row has a DOI
                    # If it doesn't have a DOI, check that at least it has an author - it may be umpublished data
                    logger.warning(f'No DOI provided for row {i} of {filename}.')
                    # Check if the empty DOI at least has author information
                    if pd.isna(row[author_firstname]) and pd.isna(row[author_lastname]):
                        logger.error(f'No author nor DOI provided for row {i} of {filename}.')
                    else:  # There is author information in the empty DOI row
                        logger.debug(f'No DOI available for row {i} but there is an author')
                        cnt += 1  # Add 1 to the counter because there is author information
                    # Checking the data types of the values
                    _check_article_author_data_types(filename, df_article_author, row)
                else:
                    # If there is a DOI, extract the article metadata through Cross-Ref or PANGAEA.
                    try:
                        # Try to access the Cross-Ref database of DOI
                        cross_ref_data = get_entity(row[doi], EntityType.PUBLICATION, OutputType.JSON)
                        title = cross_ref_data['title'][0]  # Extract title information
                        try:
                            # Sometimes the journal information is not given in Cross-Ref
                            journal = cross_ref_data['container-title'][0]  # Extract journal information
                        except IndexError:
                            # No data retrieved of journal, so this field is left the same as before
                            journal = df_article_author.loc[i, 'journal']
                        year = cross_ref_data['issued']['date-parts'][0][0]  # Extract year information
                        authors = [(d["given"], d['family']) for d in cross_ref_data['author']]
                        first_author = authors[0]
                        first_author_name = first_author[0]
                        first_author_lastname = first_author[1]
                    except ValueError:
                        # Did not access Cross-Ref successfully
                        logger.warning(f'No valid DOI found in Cross-Ref for {doi} in row {i} of {filename}, '
                                       f'checking in PANGAEA database.')
                        try:
                            # Try to access the PANGAEA database
                            ds = PanDataSet(row[doi])
                            logger.info(f'DOI {row[doi]} found in PANGAEA database in row {i} of {filename}.')
                            title = ds.title  # Extract title information
                            journal = 'PANGAEA'  # Extract journal information
                            year = ds.year  # Extract year information
                            authors = [(author.firstname, author.lastname) for author in ds.authors]
                            first_author = authors[0]
                            first_author_name = first_author[0]
                            first_author_lastname = first_author[1]
                        except:
                            # If not successfully accessed either Cross-Ref nor PANGAEA, raise an error in the log
                            logger.warning(f'No valid DOI found in Cross-Ref or PANGAEA for '
                                           f'{row[doi]} in row {i} of {filename}.')
                            title = row['title']
                            journal = row['journal']
                            year = row['year']
                            first_author_name = row['author_firstname']
                            first_author_lastname = row['author_lastname']
                    # Use data extracted from cross-ref or PANGAEA
                    except ConnectionError:
                        # Couldn't connect to the Crossref-commons server
                        logger.warning(f"Couldn't connect to Crossref to check the metadata of {doi}.")
                        title = row['title']
                        journal = row['journal']
                        year = row['year']
                        first_author_name = row['author_firstname']
                        first_author_lastname = row['author_lastname']
                    # Checking the data types of the values
                    _check_article_author_data_types(filename, df_article_author, row)
                    dict_columns_replace = {'title': title, 'journal': journal, 'year': year,
                                            'author_firstname': first_author_name,
                                            'author_lastname': first_author_lastname}
                    # Replace all entries in dataframes to these new values
                    if dfs_replace_values is not None:
                        for df in dfs_replace_values:
                            for column, value in dict_columns_replace.items():
                                if df.columns.nlevels == 1:
                                    if column in df.columns.values:
                                        # Find the entries that have the exact value as those provided by the sheet
                                        # ARTICLE_AUTHORS
                                        df[column] = df[column].replace({df_article_author[column][i]: value})
                                else:
                                    if column in df.columns.get_level_values(0):
                                        df[column] = df[column].replace({df_article_author[column][i]: value})
                    # Replace entries of the current dataframe (ARTICLE_AUTHOR) with data extracted from cross_ref
                    for column, value in dict_columns_replace.items():
                        if column in df_article_author.columns:
                            df_article_author.loc[i, column] = value
                    cnt += 1  # If successfully accessed, add 1 to the counter
            logger.info(f'Successfully checked {cnt}/{df_article_author.shape[0]} rows of '
                        f'from article_author_metadata in {filename}.')
            return cnt == df_article_author.shape[0]
        else:
            logger.warning(f'No metadata of article_author to check for {filename}')
            return None


def _check_article_author_data_types(filename, df, row):
    dict_columns = {}
    for column in row.index:
        if column in dict_author_metadata_range:
            true_false = _check_variable_range_in_df(filename, df, column,
                                                     **dict_author_metadata_range[column])
            dict_columns[column] = true_false
    correct_columns = [k for k, v in dict_columns.items() if True]
    incorrect_columns = [k for k, v in dict_columns.items() if False]
    assert len(correct_columns) == len(df.columns), _log_string(f'Incorrect columns for ARTICLE_AUTHOR: '
                                                                f'{incorrect_columns}')


# 4. Check that data is within limits for each variables. Log warnings.
def _check_variable_range_in_multiindex_df(filename, df, variable_column, dtype=None, min_value=None, max_value=None,
                                           data_list=None, in_column=None, common_min_value=None, common_max_value=None,
                                           compare_column=None, compare_columns=None, astype=None, method=None,
                                           value_len=None, additional_columns=None, raster=None,
                                           add_missing_data=False, method_details=None):
    cnt = 0
    cnt_not_common = 0
    for i, data in df[variable_column].iterrows():
        data_value = data[0]
        if pd.notna(data_value):
            if raster:
                if not _input_yes(f'Do you want to compare data provided by user and data from {raster}?'):
                    raster = None
            cnt_not_common += _check_variable_range(filename=filename, data=data_value, df=df, i=i,
                                                    variable_column=variable_column, dtype=dtype, min_value=min_value,
                                                    max_value=max_value, data_list=data_list, in_column=in_column,
                                                    common_min_value=common_min_value,
                                                    common_max_value=common_max_value, compare_column=compare_column,
                                                    compare_columns=compare_columns, astype=astype,
                                                    value_len=value_len, additional_columns=additional_columns,
                                                    raster=raster, method=method, method_details=method_details)
            _add_column(df, i, raster)
            cnt += 1
        elif add_missing_data:
            data = _extract_geopoint_data(variable_column=variable_column, raster=raster, df=df, i=i, dtype=dtype)
            cnt_not_common += _check_variable_range(filename=filename, data=data, df=df, i=i,
                                                    variable_column=variable_column, dtype=dtype, min_value=min_value,
                                                    max_value=max_value, data_list=data_list, in_column=in_column,
                                                    common_min_value=common_min_value,
                                                    common_max_value=common_max_value, compare_column=compare_column,
                                                    compare_columns=compare_columns, astype=astype,
                                                    value_len=value_len, additional_columns=additional_columns,
                                                    method=method, method_details=method_details)
            cnt += 1
    multiindex_columns = (variable_column,) + df[variable_column].columns[0]
    column_len = df[multiindex_columns].count()  # Variable column length (not nan)
    assert cnt == column_len, AssertionError(logger.warning(f'Note that {cnt}/{column_len} rows did not complete '
                                                            f'quality check of {variable_column} in {filename}.'))
    logger.info(f'Successfully checked {cnt}/{column_len} rows of {variable_column} in {filename}.')
    if cnt_not_common > 0:
        logger.warning(f'Note that {cnt_not_common} rows are not within the common data values of {variable_column} '
                       f'({common_min_value}, {common_max_value}) in '
                       f'{filename}. Please double-check.')
    return cnt == column_len


def _check_variable_range_in_df(filename, df, variable_column, dtype=None, min_value=None, max_value=None,
                                data_list=None, in_column=None, common_min_value=None, common_max_value=None,
                                compare_column=None, compare_columns=None, astype=None,
                                value_len=None, additional_columns=None, raster=None,
                                add_missing_data=False):
    cnt = 0
    cnt_not_common = 0
    for i, data in df[variable_column].iteritems():
        if pd.notna(data):
            if raster:
                if not _input_yes(f'Do you want to compare data provided by user and data from {raster}?'):
                    raster = None
            cnt_not_common += _check_variable_range(filename=filename, data=data, df=df, i=i,
                                                    variable_column=variable_column, dtype=dtype, min_value=min_value,
                                                    max_value=max_value, data_list=data_list, in_column=in_column,
                                                    common_min_value=common_min_value,
                                                    common_max_value=common_max_value, compare_column=compare_column,
                                                    compare_columns=compare_columns, astype=astype,
                                                    value_len=value_len, additional_columns=additional_columns,
                                                    raster=raster)
            _add_column(df, i, raster)
            cnt += 1
        elif add_missing_data:
            data = _extract_geopoint_data(variable_column=variable_column, raster=raster, df=df, i=i, dtype=dtype)
            cnt_not_common += _check_variable_range(filename=filename, data=data, df=df, i=i,
                                                    variable_column=variable_column, dtype=dtype, min_value=min_value,
                                                    max_value=max_value, data_list=data_list, in_column=in_column,
                                                    common_min_value=common_min_value,
                                                    common_max_value=common_max_value, compare_column=compare_column,
                                                    compare_columns=compare_columns, astype=astype,
                                                    value_len=value_len, additional_columns=additional_columns)
            cnt += 1
    column_len = df[variable_column].count()  # Variable column length (not nan)
    assert cnt == column_len, AssertionError(logger.warning(f'Note that {cnt}/{column_len} rows did not complete '
                                                            f'quality check of {variable_column} in {filename}.'))
    logger.info(f'Successfully checked {cnt}/{column_len} rows of {variable_column} in {filename}.')
    if cnt_not_common > 0:
        logger.warning(f'Note that {cnt_not_common} rows are not within the common data values of {variable_column} '
                       f'({common_min_value}, {common_max_value}) in '
                       f'{filename}. Please double-check.')
    return cnt == column_len


def _check_material_analyzed_calculated_metadata(df, filename):
    for material_analyzed in df.columns.get_level_values('material_analyzed').to_list():
        if pd.notna(material_analyzed):
            _check_variable_range(filename=filename, data=material_analyzed,
                                  df=df, i=-2,
                                  variable_column='material_analyzed')
    for raw_calculated_data in df.columns.get_level_values('raw_calculated').to_list():
        if pd.notna(raw_calculated_data):
            _check_variable_range(filename=filename, data=raw_calculated_data,
                                  df=df, i=-2,
                                  variable_column='raw_calculated_data')


def _check_variable_range(filename, data, df, i, variable_column, dtype=None, min_value=None, max_value=None,
                          data_list=None, in_column=None, common_min_value=None, common_max_value=None,
                          compare_column=None, compare_columns=None, astype=None, value_len=None,
                          additional_columns=None, raster=None, method=None, method_details=None):
    cnt_not_common = 0
    row_number = i + df.columns.nlevels + 2
    if pd.notna(dtype):
        if dtype == bool:
            assert data == False or data == True, _log_string(f'Row {row_number} of {variable_column} ({data}) in '
                                                              f'{filename} is not correct data type.')
        else:
            if dtype == numbers.Number:
                try:
                    data = float(data)
                except:
                    _log_string(f'Row {row_number} of {variable_column} ({data}) in '
                                f'{filename} is not correct data type.')
            elif dtype == str:
                data = str(data)
                assert isinstance(data, dtype), _log_string(f'Row {row_number} of {variable_column} ({data}) in '
                                                            f'{filename} is not correct data type.')
            else:
                assert isinstance(data, dtype), _log_string(f'Row {row_number} of {variable_column} ({data}) in '
                                                            f'{filename} is not correct data type.')
    if pd.notna(min_value) and pd.notna(data):
        assert data >= min_value, _log_string(f'Row {row_number} of {variable_column} ({data}) in '
                                              f'{filename} is smaller than {min_value}.')
    if pd.notna(max_value) and pd.notna(data):
        assert data <= max_value, _log_string(f'Row {row_number} of {variable_column} ({data}) in '
                                              f'{filename} is larger than {max_value}.')
    if data_list:
        assert data in data_list, _log_string(f'Row {row_number} of {variable_column} ({data}) in '
                                              f'{filename} is not in {data_list}.')
    if in_column is not None:
        assert data in in_column.values, _log_string(f'Row {row_number} of {variable_column} ({data}) in '
                                                     f'{filename} is not in {in_column.name}.')
    if compare_column:
        # Check if the data to be compared to is there (not NaN)
        if ('columns' in compare_column.keys() and _data_in_df(df=df, i=i, columns=compare_column['columns'])) or \
                ('column' in compare_column.keys() and _data_in_df(df=df, i=i, column=compare_column['column'])):
            # Checks if the dataframe has multiindex
            if df.columns.nlevels == 1:
                # Checks if the data is NOT greater than the compared value
                if compare_column['type'] == '>':
                    assert data >= abs(df.loc[i, compare_column['column']]), \
                        _log_string(f'Row {row_number} of {variable_column} ({data}) in {filename} is not greater than '
                                    f'{compare_column}.')
                # Checks if the data is NOT smaller than the compared value
                elif compare_column['type'] == '<':
                    assert data <= abs(df.loc[i, compare_column]), \
                        _log_string(f'Row {row_number} of {variable_column} ({data}) in {filename} is not smaller than '
                                    f'{compare_column} ({df[compare_column][i]}).')
            else:
                # Checks if the data is NOT greater than the compared value
                if compare_column['type'] == '>':
                    compared_value = abs(df.loc[i, compare_column['column']][0])
                    assert data >= compared_value, \
                        _log_string(f'Row {row_number} of {variable_column} ({data}) in {filename} is not greater than '
                                    f'{compare_column} ({compared_value}).')
                # Checks if the data is NOT smaller than the compared value
                elif compare_column['type'] == '<':
                    compared_value = abs(df.loc[i, compare_column['column']][0])
                    assert data <= compared_value, \
                        _log_string(f'Row {row_number} of {variable_column} ({data}) in {filename} is not smaller than '
                                    f'{compare_column} ({compared_value}).')
                elif compare_column['type'] == 'avg':
                    # Extracts the value of each column that needs to be compared into a list
                    lst_value = [df.loc[i, column_data][0] for column_data in compare_column['columns'] if
                                 column_data in df.columns.get_level_values('column') and
                                 pd.notna(df.loc[i, column_data][0])]
                    # Extracts the name of each column
                    lst_name = [column_data for column_data in compare_column['columns']]
                    # Find the number of decimal places in the data
                    decimal_places = len(str(data).split('.')[-1])
                    average = round(sum(lst_value) / len(lst_value), decimal_places)
                    assert round(data, decimal_places) == average, \
                        _log_string(f'Row {row_number} of {variable_column} ({round(data, decimal_places)}) in '
                                    f'{filename} is not the average of {lst_name} ({average}).')
    if compare_columns is not None:
        # Checks if the dataframe has multiindex
        if df.columns.nlevels == 1:
            if df.columns.isin(compare_columns).any():
                # Extracts the value of each column that needs to be compared into a list
                lst_value = [df.loc[i, compare_column] for column_data in compare_columns if
                             column_data in df.columns and
                             pd.notna(df.loc[i, compare_column])]
                if lst_value:
                    if compare_column['type'] == 'avg':
                        # Extracts the name of each column
                        lst_name = [df[column_data].name for column_data in compare_columns]
                        # Find the number of decimal places in the data
                        decimal_places = len(str(data).split('.')[-1])
                        average = round(sum(lst_value) / len(lst_value), decimal_places)
                        assert round(data, decimal_places) == average, \
                            _log_string(f'Row {row_number} of {variable_column} ({round(data, decimal_places)}) in '
                                        f'{filename} is not the average of {lst_name} ({average}).')
                    else:
                        raise
        else:
            for compare_column in compare_columns:
                # Only go through this quality check if at least one of the columns that needs to be quality checked
                # (other than the column in question) is there
                if [column for column in compare_column['columns'] if column in df.columns.get_level_values('column')
                                                                      and not variable_column]:
                    if compare_column['type'] == 'sum':
                        data_sum = sum([df.loc[i, column][0] for column in compare_column['columns'] if
                                        column in df.columns.get_level_values('column')])
                        if isinstance(compare_column['value'], numbers.Number):
                            ref_value = compare_column['value']
                        elif compare_column['value'] in df.columns.get_level_values('column'):
                            ref_value = df.loc[i, compare_column['value']][0]
                        else:
                            break
                        comparison = abs(data_sum - ref_value)
                        if comparison > compare_column['tolerance']:
                            raise _log_string(f'The sum of {variable_column} and {compare_column["columns"]} in row '
                                              f'{row_number} ({data_sum}) in {filename} does not equal '
                                              f'{ref_value} +- {compare_column["tolerance"]}.')
                    else:
                        raise
    if pd.notna(astype):
        if astype == 'num-num':
            try:
                assert len([float(value) for value in data.split('-')]) == 2, \
                    _log_string(f'Row {row_number} of {variable_column} in {filename} does not have the '
                                f'format {astype}.')
            except:
                raise TypeError(_log_string(f'Row {row_number} of {variable_column} ({data}) in {filename} does not '
                                            f'have the format {astype}.'))
    if pd.notna(value_len):
        assert len(data) <= value_len, _log_string(f'Row {row_number} of {variable_column} ({data}) in '
                                                   f'{filename} has a greater length than '
                                                   f'{value_len}.')
    if additional_columns:
        assert _evaluate_required_fields(df.columns, additional_columns) and \
               all([pd.notna(df.loc[i, additional_column][0]) for additional_column in additional_columns]), \
            _log_string(f'Row {row_number} of {variable_column} in {filename} is missing complementary data of '
                        f'{additional_columns}.')
    if method:
        method_in_data = df[variable_column].columns.get_level_values('method')[0]
        if pd.isna(method_in_data):  # If the field is left blank, assume it's _unknown and replace it in the DataFrame
            method_in_data = '_unknown'
        assert method_in_data in method, _log_string(f'Method ({method_in_data}) not in {method} for variables '
                                                     f'{variable_column}.')
    if method_details:
        method_details_in_data = df[variable_column].columns.get_level_values('method_details')[0]
        if pd.notna(method_details_in_data):
            assert isinstance(method_details_in_data, str), \
                _log_string(f'Method_detail ({method_details_in_data}) of {variable_column} in {filename} '
                            f'is not a string.')
            assert len(method_details_in_data) <= 250, \
                _log_string(f'Method_detail ({method_details_in_data}) of {variable_column} in {filename} has '
                            f'more than 100 characters.')
    if pd.notna(raster):
        # # Compares data provided by the user with data extracted from the raster
        if _input_yes(f'Do you want to compare data provided by user and data from {raster}?'):
            raster_data = _extract_geopoint_data(variable_column=variable_column, raster=raster, df=df, i=i,
                                                 dtype=dtype)
            perc_dif = (raster_data - data) / data * 100
            if abs(perc_dif) > 10:
                logger.warning(f'\nRow {i} of {variable_column} ({data}) in {filename} is {int(perc_dif)} % '
                               f'different than data extracted from raster ({raster_data}).')
    if pd.notna(common_min_value) and data < common_min_value:
        logger.info(f'Row {row_number} of {variable_column} ({data}) in {filename} is {data}, smaller than '
                    f'{common_min_value}, is this correct?')
        cnt_not_common = 1
    if pd.notna(common_max_value) and data > common_max_value:
        logger.info(f'Row {row_number} of {variable_column} in {filename} is {data}, larger than '
                    f'{common_max_value}, is this correct?')
        cnt_not_common = 1
    return cnt_not_common


def _log_string(text):
    logger.error(text)
    return text


def _add_column(df, i, raster):
    if raster:
        df.at[i, raster['column_metadata']] = 1


def check_core_range(filename, df, latitude='latitude', longitude='longitude'):
    with print_with_time(f'Checking that all core description values of {filename} are within range \n'):
        if not df.empty:
            dict_columns = {}  # Create empty dictionary to store column name and number of good entries
            # Create an empty column (NaN) for the following variables
            if any(column not in df.columns.values for column in ['water_depth_m']):
                for column in ['water_depth_m']:
                    df[column] = np.nan
            for column in list(df.columns.values):
                if column in dict_core_range:
                    true_false = _check_variable_range_in_df(filename, df, column, **dict_core_range[column])
                    dict_columns[column] = true_false
                else:
                    if column not in ['core_name', 'country_research_vessel']:
                        logger.warning(f'Column name {column} could not be quality checked.')
            # Add additional data (sea, ocean, EEZ, Longhurst province, MARCATS)
            _check_geopoint_location(filename=filename, df=df,
                                     latitude=latitude, longitude=longitude,
                                     locator_information_dict=locator_information_dict_new)
            _check_geopoint_location_OLD(filename=filename, df=df,
                                         latitude=latitude, longitude=longitude,
                                         locator_information_dict=locator_information_dict)
            incorrect_columns = [k for k, v in dict_columns.items() if False]
            return incorrect_columns
        else:
            logger.warning(f'No geopoints to check for {filename}')
            return None


def _extract_geopoint_data(variable_column, raster, df, i, dtype):
    with print_with_time(f'Retrieving {variable_column} data of row {i + 3} from raster {raster["file_dir"]}'):
        shapely.speedups.enable()  # Speeds up shapely functions
        raster_data = rasterio.open(raster["file_dir"])
        # Extract coordinates of that core
        longitude = df.loc[i, 'longitude']
        latitude = df.loc[i, 'latitude']
        point = Point(longitude, latitude)
        # Extract the index of that point from the raster
        row, col = raster_data.index(point.x, point.y)
        data_raster = raster_data.read(raster['band'])[row, col]
        data_raster = raster['fn'](data_raster)
        if isinstance(dtype, (tuple, list)):
            dtype = dtype[0]
        data_raster = dtype(data_raster)
        df.at[i, variable_column] = data_raster
        df.at[i, raster['column_metadata']] = raster['column_metadata_id']
        return data_raster


def _check_geopoint_location(filename, df, locator_information_dict, latitude='latitude', longitude='longitude'):
    # Convert DataFrame into a GeoDataFrame to quickly check if points are on land or not.
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs='epsg:4326')
    shapely.speedups.enable()  # Speeds up shapely functions
    # Open shapefiles to double_check data
    for locator, file in locator_information_dict.items():
        with print_with_time(f'Checking that all cores of {filename} are within {locator} \n'):
            locator_gdf = gpd.read_file(file['file_dir'])
            for idx_data, row_data in gdf.iterrows():
                # Check if the data_point is in a polygon
                data_point_in_locator = locator_gdf.contains(row_data.geometry)
                if data_point_in_locator.iloc[0]:
                    logger.error(f'Data point of row {idx_data + 3} ({row_data[latitude], row_data[longitude]}) '
                                 f'of {filename} not found in any polygon of {locator}.')


def _check_geopoint_location_OLD(filename, df, locator_information_dict, latitude='latitude', longitude='longitude'):
    # Convert DataFrame into a GeoDataFrame to quickly check if points are on land or not.
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs='epsg:4326')
    shapely.speedups.enable()  # Speeds up shapely functions
    # Open shapefiles to double_check data
    for locator, file in locator_information_dict.items():
        with print_with_time(f'Checking that all cores of {filename} are within {locator} \n'):
            locator_gdf = gpd.read_file(file['file_dir'])
            for idx_data, row_data in gdf.iterrows():
                # Check if eaach row is in a polygon
                data_point_in_locator = locator_gdf[locator_gdf.contains(row_data.geometry)]
                if not data_point_in_locator.empty:
                    # The data point is found in a polygon, exctract the name of the polygon
                    locator_name = data_point_in_locator[file['name']].iloc[0]
                    # First check if data for each locator is in dataframe (double check if it is correct)
                    if locator in row_data.index:  # Check if input file has this column
                        # Check if the data extracted is equal to the data in the input file
                        if row_data[locator] != locator_name and pd.notna(row_data[locator]):
                            # Extracted data from the shapefile doesn't match the data in the input Excel/file
                            # Prompt the user to ask if the data is correct
                            if _input_yes(f'Data extracted from {locator} ({locator_name}) does not match the data '
                                          f'added by user ({row_data[locator]}). Should we keep data from {locator}?'):
                                # Update data to that extracted from the shapefile
                                df.at[idx_data, locator] = locator_name
                                # If the user provides "o", keep the original data (do not do anything)
                    else:
                        # Input file does not have this information, add it
                        df.loc[idx_data, locator] = locator_name
                else:
                    # The point is not found in any polygon.
                    #  Check if it is required to be in a polygon (EEZ don't need to)
                    if file['necessary']:
                        if not _input_yes(
                                f'Data point of row {idx_data + 3} ({row_data[latitude], row_data[longitude]}) '
                                f'of {filename} not found in any polygon of {locator}. \n'
                                f'Is the data OK?'):
                            new_lat = input('Provide a new value for latitude:')
                            df.at[idx_data, latitude] = new_lat
                            new_long = input('Provide a new value for latitude:')
                            df.at[idx_data, longitude] = new_long
                    else:
                        df.at[idx_data, locator] = np.nan


def check_core_analyses_range(filename, df, core_metadata=None, title_metadata=None,
                              author_firstname_metadata=None, author_lastname_metadata=None):
    with print_with_time(f'Checking that all core analyses of {filename} are within range \n'):
        if not df.empty:
            dict_columns = {}  # Create empty dictionary to store column name and number of good entries
            _check_material_analyzed_calculated_metadata(df, filename)
            for column in df.columns.get_level_values(0).to_list():
                # Checking metadata
                if column == 'core_name':
                    if core_metadata is not None:
                        df['core_name'] = df['core_name'].astype(str)
                        true_false = _check_variable_range_in_multiindex_df(filename, df, variable_column=column,
                                                                            in_column=core_metadata.apply(str))
                        dict_columns[column] = true_false
                    else:
                        logger.warning(f'Missing core metadata for {nameof(df)} in {filename}')
                elif column == 'title':
                    if title_metadata is not None:
                        true_false = _check_variable_range_in_multiindex_df(filename, df, column, dtype=str,
                                                                            in_column=title_metadata)
                        dict_columns[column] = true_false
                    else:
                        logger.warning(f'Missing article metadata for {nameof(df)} in {filename}')
                elif column == 'author_firstname':
                    if author_firstname_metadata is not None:
                        true_false = _check_variable_range_in_multiindex_df(filename, df, column, dtype=str,
                                                                            in_column=author_firstname_metadata)
                        dict_columns[column] = true_false
                    else:
                        logger.warning(f'Missing author firstname metadata for {nameof(df)} in {filename}')
                elif column == 'author_lastname':
                    if author_lastname_metadata is not None:
                        true_false = _check_variable_range_in_multiindex_df(filename, df, column, dtype=str,
                                                                            in_column=author_lastname_metadata)
                        dict_columns[column] = true_false
                    else:
                        logger.warning(f'Missing author lastname metadata for {nameof(df)} in {filename}')
                elif column in dict_core_analyses_range:
                    true_false = _check_variable_range_in_multiindex_df(filename, df, column,
                                                                        **dict_core_analyses_range[column])
                    dict_columns[column] = true_false
                else:
                    logger.warning(f'Column name {column} could not be quality checked.')
            # Name of variables that have returned the quality check as False
            incorrect_columns = [k for k, v in dict_columns.items() if False]
            return incorrect_columns
        else:
            logger.info(f'No core analyses to check for {filename}')
            return None


def check_sample_analyses_range(filename, df, core_metadata, title_metadata, author_firstname_metadata,
                                author_lastname_metadata):
    with print_with_time(f'Checking that all sample analyses of {filename} are within range \n'):
        if not df.empty:
            dict_columns = {}  # Create empty dictionary to store column name and number of good entries
            _check_material_analyzed_calculated_metadata(df, filename)
            for column in df.columns.get_level_values('column').to_list():
                if column == 'core_name':
                    if core_metadata is not None:
                        df['core_name'] = df['core_name'].astype(str)
                        true_false = _check_variable_range_in_multiindex_df(filename, df, variable_column=column,
                                                                            in_column=core_metadata.apply(str))
                        dict_columns[column] = true_false
                    else:
                        logger.warning(f'Missing core metadata for {nameof(df)} in {filename}')
                elif column == 'title':
                    if title_metadata is not None:
                        true_false = _check_variable_range_in_multiindex_df(filename, df, column, dtype=str,
                                                                            in_column=title_metadata)
                        dict_columns[column] = true_false
                    else:
                        logger.warning(f'Missing article metadata for {nameof(df)} in {filename}')
                elif column == 'author_firstname':
                    if author_firstname_metadata is not None:
                        true_false = _check_variable_range_in_multiindex_df(filename, df, column, dtype=str,
                                                                            in_column=author_firstname_metadata)
                        dict_columns[column] = true_false
                    else:
                        logger.warning(f'Missing author firstname metadata for {nameof(df)} in {filename}')
                elif column == 'author_lastname':
                    if author_lastname_metadata is not None:
                        true_false = _check_variable_range_in_multiindex_df(filename, df, column, dtype=str,
                                                                            in_column=author_lastname_metadata)
                        dict_columns[column] = true_false
                    else:
                        logger.warning(f'Missing author lastname metadata for {nameof(df)} in {filename}')
                elif column in dict_sample_analyses_range:
                    true_false = _check_variable_range_in_multiindex_df(filename, df, column,
                                                                        **dict_sample_analyses_range[column])
                    dict_columns[column] = true_false
                elif column not in ['sample_name']:
                    logger.warning(f'Column name {column} could not be quality checked.')
            incorrect_columns = [k for k, v in dict_columns.items() if False]
            return incorrect_columns
        else:
            logger.info(f'No sample analyses to check for {filename}')
            return None


def save_df(df_list, df_list_name, filename, dir_output=None):
    """
    Saves DataFrame (.xls or .csv)
    :param df_list_name: list of sheetnames to save each dataframe in
    :param df_list: list of pandas dataframes to be saved
    :param filename: name of the file to be saved to. Add the extension (.csv or .xlsx)
    :param dir_output: Specify the output directory. Default is None, it will save to the working directory.
    :return:
    """
    date_save = datetime.date.today().strftime('%Y_%m_%d')
    filename_today = date_save + '_' + filename

    # Makes sure the output directory (file_dir) exists, if not, it creates it.
    if dir_output is not None:
        os.makedirs(dir_output, exist_ok=True)
        filename_today = os.path.join(dir_output, filename_today)
    if filename.endswith('.xlsx'):
        with print_with_time('Saving file as ' + filename_today):
            with pd.ExcelWriter(filename_today) as writer:
                for df, name in zip(df_list, df_list_name):
                    if df.columns.nlevels == 1:
                        df.to_excel(writer, sheet_name=name, index=False)
                    else:
                        for level in range(df.columns.nlevels):
                            # Separate each level of the columns and save them separately
                            # (Saving Multiindex causes errors)
                            df_column = pd.DataFrame(columns=df.columns.get_level_values(level))
                            df_column.to_excel(writer, sheet_name=name, merge_cells=True, index=False, startrow=level)
                        levels = df.columns.nlevels
                        # Saving the values aside (Saving Multiindex causes errors)
                        df_values = df.droplevel(list(range(1, levels)), axis=1)
                        df_values.to_excel(writer, sheet_name=name, header=False, index=False, startrow=levels)
                writer.save()
    elif filename.endswith('.csv') and len(df_list) == 1:
        df = df_list[0]
        df.to_csv(filename_today, index=None)
    else:
        raise TypeError('Data not .xls or .csv type, or passed on too many arguments for DataFrame')


def _data_in_df(df, i, columns=None, column=None):
    if column:
        if column in df.columns:
            if pd.notna(df.loc[i, column][0]):
                return True
            else:
                return False
        else:
            return False
    if columns:
        if df.columns.nlevels == 1:
            if all(df.columns.isin(columns)):
                if all([pd.notna(df.loc[i, column][0]) for column in columns]):
                    return True
                else:
                    return False
            else:
                return False
        else:
            if all(df.columns.get_level_values(1).isin(columns)):
                if all([pd.notna(df.loc[i, column][0]) for column in columns]):
                    return True
                else:
                    return False
            else:
                return False


def move_file(filename, dir_output, file_dir=None):
    # Make sure that the output directory exists
    if not os.path.exists(dir_output):
        os.mkdir(dir_output)
    # Move file to new destination
    shutil.move(os.path.join(file_dir, filename),
                os.path.join(dir_output, filename))


def _input_yes(message_input):
    res = _my_input(message_input, possible_values=('y', 'n'))
    return res == 'y'  # returns True if input is yes


def _my_input(message_input, possible_values, to_type_fn=str):
    while True:
        try:
            res = to_type_fn(input(message_input + f"  {possible_values}"))
            assert isinstance(res, type(possible_values[0])), "Error: maybe you messed up with the 'to_type_fn'?"
            if res in possible_values:
                return res
        except ValueError:
            pass
        print(f"Not a valid answer. Please choose between {possible_values}.")


def _evaluate_required_fields(input_list, target):
    """
    Evaluates as True or False if all elements in a list and at least one element in every tuple is in the input_list
    # tuple ( ) in target means OR, list [ ] means AND
    """
    if isinstance(target, list):
        return all(_evaluate_required_fields(input_list, y) for y in target)  # eval AND
    elif isinstance(target, tuple):  # eval OR
        return any(_evaluate_required_fields(input_list, y) for y in target)
    else:
        return target in input_list


example_test = ['a', 'b', 'c', 'd', ('e', 'f'), ('g', 'h'), ('i', ['j', 'k'])]
in1 = ['a', 'b', 'c', 'd', 'e', 'g', 'i']  # returns True
in2 = ['a', 'b', 'c', 'd', 'e', 'g', 'j', 'k']  # returns True
in3 = ['a', 'b', 'c', 'd', 'e', 'g', 'j']  # returns False
in4 = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'i']  # returns True?????


def test_eval():
    assert _evaluate_required_fields(in1, example_test)
    assert _evaluate_required_fields(in2, example_test)
    assert not _evaluate_required_fields(in3, example_test)
    assert _evaluate_required_fields(in4, example_test)
