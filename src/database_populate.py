# -*- coding: utf-8 -*-
"""
Created on Monday 30th of March 2021

@author: Sarah Paradis

Python module to populate MOSAIC database

1. Connect to the database

2. Read data after quality check

3. Populate Article and author metadata. Log warning if data is already in database.

4. Populate Core information. Log warning if data is already in database.

5. Populate core analyses. Log warning if data is already in database.

6. Populate sample analyses. Log warning if data is already in database.
"""
import shutil
import numpy as np
import pandas as pd
from crossref_commons.retrieval import get_entity
from crossref_commons.types import EntityType, OutputType
from pangaeapy.pandataset import PanDataSet
from requests.exceptions import SSLError
import logging
import sys
import warnings
import os
import collections

from mosaic.general_funcs import general
from mosaic.general_funcs.database_connect import connection as conn, schema
from PyQt5.QtWidgets import QWidget, QScrollArea, QTableWidget, QVBoxLayout, QTableWidgetItem
from mosaic.general_funcs.general import print_with_time, columns_table, columns_tables, query_exact_data, sql_table_information, _round_sql
import harmonization

logger = logging.root
logger.setLevel(logging.INFO)
# Allows to print the whole dataframe in the console (normally, only a few columns and rows are printed)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
# Allows the height of the dataframe to be defined by the screen's height (instead of printing the dataframe in different
# lines)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 20)
pd.options.mode.chained_assignment = None  # default='warn'


if not logger.hasHandlers():
    # Set the logging file
    file_handler = logging.FileHandler('database_populate.log')
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


# 2. Read data after quality check
def read_df(file, article_author_metadata=None, geopoints_cores=None,
            core_analyses=None, sample_analyses=None, file_dir=None):
    """
    Reads a file (either excel or open office) as a DataFrame
    If several files need to be opened, specify the directory where they are stored
    Specify the sheet names where data is stored.

    :returns dataframes for article_author, geopoints_cores, core_analyses, sample_analyses
    """
    if file_dir is not None:
        file = os.path.join(file_dir, file)
    if file.endswith('.xlsx') or file.endswith('.xlsx') or file.endswith('.ods'):
        with print_with_time(f'Opening file {file} '):
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
    return df_lists


def _read_article_author_metadata(file, article_author_metadata):
    if article_author_metadata:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            df_article_author = pd.read_excel(file, sheet_name=article_author_metadata, dtype=str)
            #df_article_author.replace("'", "''", regex=True, inplace=True)
    return df_article_author


def _read_geopoints(file, geopoints_cores):
    if geopoints_cores:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            df_geopoints = pd.read_excel(file, sheet_name=geopoints_cores, dtype={'core_name': str,
                                                                                  'sampling_year': 'Int64',
                                                                                  'sampling_month': 'Int64',
                                                                                  'sampling_day': 'Int64'            })
            df_geopoints['georeferenced_coordinates'].fillna(0, inplace=True)
            df_geopoints['georeferenced_coordinates'] = df_geopoints['georeferenced_coordinates'].astype(int)
            df_geopoints.replace("'", "''", regex=True, inplace=True)
    return df_geopoints


def _read_core_analyses(file, core_analyses):
    if core_analyses:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            df_core_analyses = pd.read_excel(file, sheet_name=core_analyses)
            if not df_core_analyses.empty:
                df_core_analyses = pd.read_excel(file, sheet_name=core_analyses, header=[0, 1, 2])
                df_core_analyses.replace("'", "''", regex=True, inplace=True)
    return df_core_analyses


def _read_sample_analyses(file, sample_analyses):
    # UserWarning is warning that the Excel sheet has several dtaa validations. Ignore
    warnings.simplefilter("ignore", category=UserWarning)
    df_sample_analyses = pd.read_excel(file, sheet_name=sample_analyses, dtype={'core_name': str})
    if not df_sample_analyses.empty:
        arrays = [list(df_sample_analyses.columns.str.rstrip('.1')),
                  df_sample_analyses.iloc[0, :].to_list(),
                  df_sample_analyses.iloc[1, :].to_list(),
                  df_sample_analyses.iloc[2, :].to_list(),
                  df_sample_analyses.iloc[3, :].to_list()]
        df_sample_analyses.columns = pd.MultiIndex.from_arrays(arrays, names=['column', 'method', 'method_details',
                                                                              'material_analyzed', 'raw_calculated'])
        # Remove the end of duplicate columns (.X)
        possible_duplicates = [column for column in df_sample_analyses.columns.get_level_values(0) if '.' in column]
        rename_mapper = {possible_duplicate: possible_duplicate.split('.')[0] for possible_duplicate in
                         possible_duplicates}
        df_sample_analyses.rename(columns=rename_mapper, inplace=True)
        df_sample_analyses.drop(index=[0, 1, 2, 3], inplace=True)
        df_sample_analyses.dropna(axis='columns', how='all', inplace=True)
        df_sample_analyses.dropna(how='all', inplace=True)
        df_sample_analyses.reset_index(drop=True, inplace=True)
        #df_sample_analyses.replace("'", "''", regex=True, inplace=True)
    return df_sample_analyses


# 3. Populate Article and author metadata. Log warning if data is already in database.
def insert_article_from_columns(df, doi_column='doi', title_column='title', year_column='year',
                                journal_column='journal',
                                author_firstname_column='author_firstname',
                                author_lastname_column='author_lastname',
                                article_table='articles', author_table='authors', authorship_table='authorship'):
    article_metadatas = {}
    for idx, row in df.iterrows():
        article_metadata = None
        if doi_column in df.columns and pd.notna(df[doi_column][idx]):
            # If DOI of that row is not NaN, insert article and author tables using DOI
            article_metadata = insert_article_from_doi(doi=df[doi_column][idx],
                                                       article_table=article_table,
                                                       authorship_table=authorship_table,
                                                       author_table=author_table)
            # insert_article_from_doi will return None if no information can be gathered from DOI (cross-ref / PANGAEA)
        if article_metadata is None:
            # If DOI of that row is empty (Nan) or its metadata couldn't be extracted,
            # insert article and author tables using remaining data
            columns_article = [doi_column, title_column, year_column, journal_column]
            dict_columns_article = {column: df[column][idx] for column in columns_article if column in df.columns
                                    and pd.notna(df[column][idx])}
            article_metadata = insert_article_author_table(article_table=article_table, author_table=author_table,
                                                           authorship_table=authorship_table, **dict_columns_article,
                                                           authors=[(df[author_firstname_column][idx],
                                                                     df[author_lastname_column][idx])])
        article_metadatas.update(article_metadata)
    return article_metadatas


def insert_article_from_doi(doi, article_table, author_table, authorship_table):
    """
    Extracts article's information from its DOI, and populates the database.
    :param doi: DOI to be searched in Cross-REF or PANGAEA
    :param article_table: Name of article table in the database where data needs to be added
    :param author_table: Name of author table in the database where data needs to be added
    :param authorship_table: Name of authorship (link between article and author tables) in the database
    :return Article metadata or None if no information can be gathered from DOI (cross-ref / PANGAEA)
    """
    with print_with_time(f'Extracting data from {doi} '):
        try:  # If source comes from a journal, metadata should be stored in cross-ref
            cross_ref_data = get_entity(doi, EntityType.PUBLICATION, OutputType.JSON)  # Extract article metadata
            doi = _cross_ref_extract(cross_ref_data, 'DOI')
            title = _cross_ref_extract(cross_ref_data, 'title')
            journal = _cross_ref_extract(cross_ref_data, 'journal')
            year = _cross_ref_extract(cross_ref_data, 'year')
            authors = _cross_ref_extract(cross_ref_data, 'authors')
            article_metadata = insert_article_author_table(article_table=article_table, author_table=author_table,
                                                           authorship_table=authorship_table, title=title, doi=doi,
                                                           year=year, journal=journal, authors=authors)
        except (ValueError, KeyError, SSLError):
            print(f'\n Data from {doi} not found in Cross-Ref. Checking in PANGAEA database.')
            try:  # If source is not in cross-ref, try searching in Pangaea
                ds = PanDataSet(doi)
                if ds.children:  # PANGAEA dataset sometimes stores sub-data in it
                    for child_doi in ds.children:
                        child_ds = PanDataSet(child_doi)
                        title = child_ds.title
                        year = child_ds.year
                        journal = 'PANGAEA'
                        doi = child_doi
                        authors = [(child_ds.authors[x].firstname, child_ds.authors[x].lastname) for x in
                                   range(len(child_ds.authors))]
                        article_metadata = insert_article_author_table(article_table=article_table,
                                                                       author_table=author_table,
                                                                       authorship_table=authorship_table, title=title,
                                                                       doi=doi, year=year, journal=journal,
                                                                       authors=authors)
                else:
                    if pd.notna(ds.title):
                        title = ds.title
                        year = ds.year
                        journal = 'PANGAEA'
                        authors = [(ds.authors[x].firstname, ds.authors[x].lastname) for x in range(len(ds.authors))]
                        article_metadata = insert_article_author_table(article_table=article_table,
                                                                       author_table=author_table,
                                                                       authorship_table=authorship_table, title=title,
                                                                       doi=doi, year=year, journal=journal,
                                                                       authors=authors)
                    else:
                        logger.warning(f'\n No data found for {doi}')
                        return None
            except:
                logger.warning(f'\n No data found for {doi}')
                return None
        return article_metadata


def insert_article_author_table(article_table, author_table, authorship_table,
                                title=None, doi=None, year=None, journal=None, authors=None):
    author_ids = []
    if title is not None or doi is not None or year is not None or journal is not None:
        article_id = _insert_article_table(title=title, doi=doi, year=year, journal=journal,
                                           sql_table=article_table)
        for (firstname, lastname) in authors:
            author_id = _insert_author_table(lastname=lastname, firstname=firstname, sql_table=author_table)
            author_ids.append(author_id)
            _insert_authorship_table(author_id=author_id, article_id=article_id, sql_table=authorship_table)
        conn.commit()
        # Extract dictionary of article_metadata (article title + id, and first author + id)
        article_metadata = {title: article_id, authors[0]: author_ids[0]}
        return article_metadata
    else:
        for (firstname, lastname) in authors:
            author_id = _insert_author_table(lastname=lastname, firstname=firstname, sql_table=author_table)
            author_ids.append(author_id)
        conn.commit()
        # Extract dictionary of article_metadata (article title + id, and first author + id)
        article_metadata = {authors[0]: author_ids[0]}
        return article_metadata


def _insert_article_table(title=None, doi=None, year=None, journal=None,
                          sql_table='articles'):
    assert title is not None or doi is not None or year is not None or journal is not None, "No data to add"
    with print_with_time(f'Inserting {title} into {sql_table} '):
        # First check if data of that article is already in the database using its DOI or title
        db_cursor = conn.cursor()
        db_cursor.execute(f'''SELECT article_id FROM {schema}."{sql_table}" WHERE article_id = 0''')
        article_id_0 = db_cursor.fetchall()
        assert article_id_0, LookupError('Missing article_id 0 from the database')
        if pd.notna(title) and "'" in title:
            # Postgresql does not accept apostrophes in a string, it needs to be replaced by a double single-quote
            title = title.replace("'", "''")
        dict_article_table = {'title': title, 'doi': doi, 'year': year, 'journal': journal}
        article_id = _query_id_null(sql_table=sql_table, id_name='article_id', dict_variable_value=dict_article_table)
        if not article_id:
            article_id = _insert_and_query_id(id_name='article_id',
                                              dict_column_values=dict_article_table,
                                              sql_table=sql_table)
        else:
            # Data is found in the database
            logger.warning(f'{title} already in {sql_table} table')
        db_cursor.close()
        return article_id


def _insert_author_table(lastname=None, firstname=None, sql_table='authors'):
    assert lastname is not None or firstname is not None, 'No data to add'
    # Add to MySQL
    with print_with_time(f'Inserting {lastname}, {firstname} into {sql_table} '):
        firstname = firstname[0]  # Extract author initial
        db_cursor = conn.cursor()
        db_cursor.execute(f'''SELECT author_id FROM {schema}."{sql_table}" WHERE author_id = 0''')
        article_id_0 = db_cursor.fetchall()
        assert article_id_0, LookupError('Missing article_id 0 from the database')
        # Replace single quotes to 2 single quotes. Postgresql does not accept apostrophes in a string
        if pd.notna(lastname) and "'" in lastname:
            lastname = lastname.replace("'", "''")
        elif pd.notna(firstname) and "'" in firstname:
            firstname = firstname.replace("'", "''")
        # Query database if there is an author_id for that author
        dict_author_table = {'author_lastname': lastname, 'author_firstname': firstname}
        author_id = _query_id_null(sql_table=sql_table,
                                   id_name='author_id',
                                   dict_variable_value=dict_author_table)
        if not author_id:
            # If no data was obtained in the Database of that author, insert it into the database
            author_id = _insert_and_query_id(id_name='author_id',
                                             dict_column_values=dict_author_table,
                                             sql_table=sql_table)
        else:
            # Data is found in the database
            logger.warning(f'{lastname}, {firstname} already in {sql_table} table')
        db_cursor.close()
        return author_id


def _insert_authorship_table(author_id, article_id, sql_table='authorship'):
    assert article_id is not None and author_id is not None, 'No data to add'
    with print_with_time(f'Inserting {article_id} and {author_id} into {sql_table} '):
        db_cursor = conn.cursor()
        # Query the database first to see if it is already there
        db_cursor.execute(f'''SELECT article_id, author_id FROM {schema}."{sql_table}" WHERE article_id = {article_id}
                            AND author_id = {author_id}''')
        authorship = db_cursor.fetchall()
        if not authorship:
            # No authorship of article_id and author_id is found in the database
            db_cursor.execute(f'''INSERT INTO {schema}."{sql_table}" (article_id, author_id)
                                VALUES ( %s, %s)''', (article_id, author_id))
        else:
            # Data is found in the database
            logger.warning(f'{article_id}, {author_id} already in {sql_table} table')
        db_cursor.close()


def _cross_ref_extract(cross_ref, data):
    """
    Extracts important data from Cross-Ref
    :param cross_ref: JSON format of Cross-Ref data
    :param data: Data to be extracted (DOI, title, journal, year, authors)
    :return: Harmonized data from Cross-Ref
    """
    try:
        if data == 'DOI':
            cross_ref_data = cross_ref[data]
        elif data == 'title':
            cross_ref_data = cross_ref[data][0]
        elif data == 'journal':
            cross_ref_data = cross_ref['container-title'][0]
        elif data == 'year':
            cross_ref_data = cross_ref['issued']['date-parts'][0][0]
        elif data == 'authors':
            cross_ref_data = [(d["given"], d['family']) for d in cross_ref['author']]
        else:
            raise ValueError
    except IndexError:
        # If there is no data to be extracted, it will raise an IndexError. In that case, return an empty data
        cross_ref_data = np.nan
    return cross_ref_data


# 4. Populate Core information. Log warning if data is already in database.
def insert_geopoints(df, sampling_campaign_table='sampling_campaign', geopoints_table='geopoints'):
    with print_with_time(f'Inserting data into {sampling_campaign_table} and {geopoints_table}'):
        core_id_name_dict = {}
        # To populate geopoints and sampling_campaign tables, extract all the column names from the database
        columns_geopoints = columns_table(sql_table=geopoints_table)
        columns_sampling = columns_tables(sql_tables=[sampling_campaign_table, 'sampling_method'])
        # Populate each row into sampling_campaign_table and geopoints_table
        for i, row in df.iterrows():
            # Convert the variables into a dictionary of data that needs to be populated in the table, which will be
            # inputed into the following functions
            sampling_campaign_dict = _variables_to_populate_dict(row, columns_sampling)
            if bool(sampling_campaign_dict):  # Check if the dictionary has entries
                # Populate sampling_campaign table
                sampling_method_id, sampling_campaign_id = _insert_sampling_campaign_table(
                    sql_table=sampling_campaign_table, **sampling_campaign_dict)
            else:
                sampling_method_id = 1  # If there are no entries, choose _unknown (sampling_method_id = 1)
                sampling_campaign_id = None
            # Create a new dictionary with the variables that will be populated into the geopoints table
            # Each row may have different variables (not NaN) so this step is performed here
            geopoints_dict = _variables_to_populate_dict(row, columns_geopoints)
            core_id, core_name = _insert_geopoints_table(geopoints_dict=geopoints_dict,
                                                         sampling_method_id=sampling_method_id,
                                                         sampling_campaign_id=sampling_campaign_id,
                                                         sql_table=geopoints_table)
            core_id_name_dict[core_name] = core_id
            conn.commit()
    return core_id_name_dict


def _insert_sampling_campaign_table(research_vessel=None, sampling_campaign_name=None,
                                    sampling_campaign_date_start=None,
                                    sampling_campaign_date_end=None, sampling_method_type=None,
                                    sql_table='sampling_campaign'):
    """
    Populates sampling_campaign table into the database
    :param research_vessel:
    :param sampling_campaign_name:
    :param sampling_campaign_date_start:
    :param sampling_campaign_date_end:
    :param sampling_method_type:
    :param sql_table:
    :return: sampling_method_id, sampling_campaign_id
    """
    assert research_vessel is not None or sampling_campaign_name is not None or sampling_campaign_date_start is not \
           None or sampling_campaign_date_end is not None or sampling_method_type is not None, 'No data to add'
    with print_with_time(f'Inserting sampling campaign {sampling_campaign_name} and research vessel '
                         f'{research_vessel} into {sql_table} '):
        sampling_campaign_dict = {'research_vessel': research_vessel,
                                  'sampling_campaign_name': sampling_campaign_name,
                                  'sampling_campaign_date_start': sampling_campaign_date_start,
                                  'sampling_campaign_date_end': sampling_campaign_date_end}
        if all(value is None or pd.isna(value) for key, value in sampling_campaign_dict.items()
               if key in ['research_vessel', 'sampling_campaign_name']):
            # If they all are None (there is no data to add)
            logger.info(f'No data to add to {sql_table} for research vessel {research_vessel} and sampling campaign'
                        f'{sampling_campaign_name}.')
            sampling_campaign_id = None
        else:
            # If there is information of the cruise, look for its id in the database first (avoid duplicates) based
            # only on the research vessel and sampling campaign name (sampling dates are often empty)
            dict_researchvessel_campaign = {'research_vessel': research_vessel,
                                            'sampling_campaign_name': sampling_campaign_name}

            sampling_campaign_id = _query_id_null(sql_table=sql_table, id_name='sampling_campaign_id',
                                                  dict_variable_value=dict_researchvessel_campaign)
            # If there is no sampling_campaign_id with that information, insert it in the database
            if not sampling_campaign_id:
                sampling_campaign_id = _insert_and_query_id(id_name='sampling_campaign_id',
                                                            dict_column_values=sampling_campaign_dict,
                                                            sql_table=sql_table)
            else:
                # There is information for that sampling_campaign_id, and maybe some needs to be updated
                sampling_campaign_dict['sampling_campaign_id'] = sampling_campaign_id
                _update_table(dict_column_values=sampling_campaign_dict, sql_table=sql_table)
        # Find sampling_method_id in the database
        if sampling_method_type:
            sampling_method_id = _query_id_null(sql_table='sampling_method', id_name='sampling_method_id',
                                                dict_variable_value={'sampling_method_type': sampling_method_type})
            if not sampling_method_id:
                sampling_method_id = 1
            assert isinstance(sampling_method_id, int) and sampling_method_id >= 1, \
                LookupError(f'No sampling_method_id found for {sampling_method_type}')
        else:
            # If there is no sampling_method_type provided, give it the default value of 1 (_unknown)
            sampling_method_id = 1
        # Evaluate whether data needs to be added to sampling nexus table
        if sampling_campaign_id is not None:
            _insert_sampling_nexus(sampling_method_id=sampling_method_id, sampling_campaign_id=sampling_campaign_id)
        else:
            logger.info(f'No data to add to sampling_nexus for {sampling_campaign_name} and {sampling_method_type}.')
    return sampling_method_id, sampling_campaign_id


def _insert_sampling_nexus(sampling_method_id, sampling_campaign_id, sql_table='sampling_nexus'):
    assert sampling_method_id is not None and sampling_campaign_id is not None, 'No data to add'
    with print_with_time(f'Inserting {sampling_method_id} and {sampling_campaign_id} into {sql_table} '):
        sampling_nexus_dict = {'sampling_method_id': sampling_method_id, 'sampling_campaign_id': sampling_campaign_id}
        # Query database if this sampling_nexus already exists
        sampling_nexus = _query_id_null(sql_table=sql_table, id_name='sampling_method_id',
                                        dict_variable_value=sampling_nexus_dict)
        if not sampling_nexus:
            _insert(dict_column_values=sampling_nexus_dict, sql_table=sql_table)
        else:
            # Data is found in the database
            logger.warning(f'Sampling_nexus {sampling_method_id}, {sampling_campaign_id} already in {sql_table} table')


def _insert_geopoints_table(geopoints_dict, sampling_method_id, sampling_campaign_id, sql_table):
    assert 'core_name' and 'latitude' and 'longitude' and 'entry_user_name' in geopoints_dict.keys(), \
        ValueError('Missing key variables core_name, latitude, longitude, or entry_user_name')
    core_name = str(geopoints_dict['core_name'])
    latitude = geopoints_dict['latitude']
    longitude = geopoints_dict['longitude']
    entry_user_name = geopoints_dict['entry_user_name']
    assert core_name is not None and latitude is not None and longitude is not None and entry_user_name is not None, \
        ValueError('No data of core_name, latitude, longitude, or entry_user_name to add')
    with print_with_time(f'Inserting {core_name} into {sql_table} '):
        # Add sampling_method_id, sampling_campaign_id to the dictionary, and create a geometry type of coordinates
        geopoints_dict.update({'sampling_campaign_id': sampling_campaign_id,
                               'sampling_method_id': sampling_method_id,
                               'geopoints_point': f'POINT({longitude} {latitude})'})
        # Query if there is an exact same entry (core_name, coordinates) in the database.
        core_id = _core_in_database(core_name=core_name, latitude=latitude, longitude=longitude, sql_table=sql_table)
        if core_id:
            # The exact same core is already in the database.
            logger.warning(f'Core name {core_name}, at ({latitude}, {longitude}) already in {sql_table} table')
            # Update variables in table (if there are variables to be updated)
            geopoints_dict['core_id'] = core_id
            _update_table(dict_column_values=geopoints_dict, sql_table=sql_table)
        if not core_id:
            # No exact core (core_name, coordinates) was found in the database
            # Query database to see if an entry is located close to the latitude and longitude (the same core
            # may have different core name, or the same core may have a different coordinate precision)
            df_nearby_cores = _nearby_cores(latitude, longitude, sql_table)
            if df_nearby_cores.empty:
                # There are no nearby cores, so populate it in the database
                core_id = _insert_and_query_id(id_name='core_id',
                                               dict_column_values=geopoints_dict,
                                               sql_table=sql_table)
            else:
                # There are nearby cores, prompt the user if the core to be added is one of these
                geopoints_df = pd.DataFrame([geopoints_dict], columns=list(geopoints_dict.keys()))
                if _input_yes(f'{core_name} (Full data): \n \n '
                              f'{geopoints_df} \n \n '
                              f'The database already has the following entries corresponding to similar coordinates '
                              f'as {core_name}. Do any of the following entries correspond to {core_name}?:'
                              f'\n \n {df_nearby_cores} \n '):
                    if len(set(df_nearby_cores.index.to_list())) == 1:
                        # There's only one nearby core, so this one needs to be updated
                        core_id = df_nearby_cores.index.to_list()[0]
                        if len(df_nearby_cores) == 1:
                            logger.warning(f'Core_name {core_name} has been assigned the same core_id '
                                           f'{core_id} as {df_nearby_cores.loc[core_id, "Core name"]}')
                        else:
                            # To print out the core_name, if the length of the DF is more than one, you need to call it
                            # using .iloc again
                            logger.warning(f'Core_name {core_name} has been assigned the same core_id '
                                           f'{core_id} as {df_nearby_cores.loc[core_id, "Core name"].iloc[0]}')
                    else:
                        core_id = _my_input(message_input='Which core_id corresponds to the core that needs to be '
                                                          'populated?',
                                            possible_values=set(df_nearby_cores.index.to_list()), to_type_fn=int)
                        if len(df_nearby_cores.loc[core_id, "Core name"]) == 1:
                            logger.warning(f'Core_name {core_name} has been assigned the same core_id '
                                           f'{core_id} as {df_nearby_cores.loc[core_id, "Core name"]}')
                        else:
                            # To print out the core_name, if the length of the DF is more than one, you need to call it
                            # using .iloc again, this is the only difference in this if/else statement
                            logger.warning(f'Core_name {core_name} has been assigned the same core_id '
                                           f'{core_id} as {df_nearby_cores.loc[core_id, "Core name"].iloc[0]}')
                    # The core is already in the database, and it has been assigned its specific core_id.
                    # Add any additional data that is not in the database for that core
                    geopoints_dict['core_id'] = core_id
                    _update_table(dict_column_values=geopoints_dict, sql_table=sql_table)
                else:  # If the new data is not any of these cores, populate it in the database
                    core_id = _insert_and_query_id(id_name='core_id',
                                                   dict_column_values=geopoints_dict,
                                                   sql_table=sql_table)
    return core_id, core_name


def _core_in_database(core_name, latitude, longitude, sql_table):
    """
    Queries the database and extracts a core_id from the sql_table that has the same core_name and coordinates
    :param core_name:
    :param latitude:
    :param longitude:
    :param sql_table:
    :return: core_id
    """
    db_cursor = conn.cursor()
    db_cursor.execute(f'''SELECT core_id FROM {schema}."{sql_table}" WHERE core_name = '{core_name}' AND
                        latitude = {format(latitude, '.8f')} AND
                        longitude = {format(longitude, '.8f')}''')
    core_id = db_cursor.fetchall()
    db_cursor.close()
    assert len(core_id) <= 1, TypeError(f'Returned more than one row for {core_name}. '
                                        f'There is a duplicate entry in the database')
    if len(core_id) == 1:
        core_id = core_id[0][0]
    return core_id


def _nearby_cores(latitude, longitude, sql_table):
    # Query the database to see if there are nearby cores, and extract important information of those cores in a
    # dataframe
    db_cursor = conn.cursor()
    latitude_round = _round_sql(latitude, 2)
    longitude_round = _round_sql(longitude, 2)
    db_cursor.execute(f'''SELECT core_id FROM {schema}."{sql_table}" WHERE ROUND(latitude, 2) = {latitude_round} AND 
                        ROUND(longitude, 2) = {longitude_round}''')
    core_id = db_cursor.fetchall()
    if core_id:
        # There are nearby cores, extract their information to prompt the user if the core that we want to
        # populate is already in the database
        # Adapt it to MySQL query format
        core_ids = ', '.join(str(sql_id[0]) for sql_id in core_id)
        # Extract main parameters of all geopoints found nearby, and all the variables analysed there
        db_cursor.execute(f'''SELECT * FROM
            (SELECT geopoints.core_id, core_name AS "Core name", ROUND(latitude, 3) AS "Latitude", 
                    ROUND(longitude, 3) AS "Longitude", TO_CHAR(sampling_date, 'YYYY (MON)') as 
                    "Year (month) sampled", water_depth_m AS "Depth", 
                    sampling_method_type
                    AS "Sampling technique", core_comment AS "comment",
                    title AS "Title", year AS "Year", journal AS "Journal", 
                    CONCAT(author_lastname, ', ', author_firstname) AS "Author", 
                    core_analysis AS "Analysis"
                FROM mosaic.geopoints
                    LEFT JOIN {schema}.sampling_method USING(sampling_method_id)
                    LEFT JOIN {schema}.core_metadata USING (core_id)
                    LEFT JOIN {schema}.core_analysis_description USING (core_analysis_id)
                    LEFT JOIN {schema}.articles ON core_metadata.core_article_id = articles.article_id
                    LEFT JOIN {schema}.authors ON core_metadata.core_contact_person_id = authors.author_id
            UNION
                SELECT geopoints.core_id, core_name AS "Core name", ROUND(latitude, 3) AS "Latitude", 
                    ROUND(longitude, 3) AS "Longitude", TO_CHAR(sampling_date, 'YYYY (MON)') as 
                    "Year (month) sampled", water_depth_m AS "Depth", 
                    sampling_method_type 
                    AS "Sampling technique", core_comment AS "comment",
                    title AS "Title", year AS "Year", journal AS "Journal", 
                    CONCAT(author_lastname, ', ', author_firstname) AS "Author", 
                    CONCAT(sample_analysis, ' (', material_analyzed,')') AS Analysis
                FROM mosaic.geopoints
                    LEFT JOIN {schema}.sampling_method USING(sampling_method_id)
                    LEFT JOIN {schema}.sample_metadata ON geopoints.core_id = sample_metadata.sample_core_id
                    LEFT JOIN {schema}.sample_material_analyzed USING (material_analyzed_id)
                    LEFT JOIN {schema}.sample_analysis_description USING (sample_analysis_id)
                    LEFT JOIN {schema}.articles ON sample_metadata.sample_article_id = articles.article_id
                    LEFT JOIN {schema}.authors ON sample_metadata.sample_contact_person_id = authors.author_id) as t
                WHERE t.core_id in ({core_ids})
                ORDER BY t.core_id, t."Year", t."Title"''')
        df_nearby_cores = pd.DataFrame(db_cursor.fetchall(), columns=[desc[0] for desc in db_cursor.description])
        df_nearby_cores.set_index('core_id', inplace=True)
    else:
        df_nearby_cores = pd.DataFrame()
    db_cursor.close()
    return df_nearby_cores


# 5. Populate core analyses
def insert_core_analyses(df, core_id_dict, article_author_metadata):
    with print_with_time('\nInserting core analysis data\n'):
        for i, row in df.iterrows():
            # Convert each row into a dataframe with the metadata
            df_row = _transform_multiidex_row_samples(row)
            if not df_row.dropna(how='all').empty:
                _insert_core_analysis_tables(df_row=df_row, core_id_dict=core_id_dict,
                                             article_author_metadata=article_author_metadata)
            conn.commit()


def _transform_multiidex_row_samples(row):
    df_row = pd.DataFrame(row).reset_index().set_index('column')
    df_row = df_row.T
    df_row.set_axis(['method', 'method_details', 'material_analyzed', 'raw_calculated', 'data'], axis=0, inplace=True)
    df_row = df_row.replace(to_replace=r'^Unnamed.', value=np.nan, regex=True)
    return df_row


def _transform_multiidex_row_cores(row):
    df_row = pd.DataFrame(row).transpose()
    df_row.set_axis(['data'], axis=0, inplace=True)
    df_row = df_row.replace(to_replace=r'^Unnamed.', value=np.nan, regex=True)
    return df_row


def _insert_core_analysis_tables(df_row, core_id_dict, article_author_metadata):
    # Create a dictionary of sql table (key) with a list of variables that will be added in it (value)
    sql_table_variables_dict = _dictionary_analysis_table(df_row=df_row, analysis_source='core')
    core_name = str(df_row['core_name']['data'])
    core_id = core_id_dict[core_name]
    assert core_id is not None, ValueError(f'No core_id found for {core_name}')
    for sql_table, columns in sql_table_variables_dict.items():
        dict_column_values = _insert_core_analysis_table(core_id=core_id, df_row=df_row, columns=columns,
                                                         sql_table=sql_table)
        # If data has been added to a core analysis table, it will return a dictionary of added values
        # If no data was added to a core analysis table, it will return None, so we first need to check if there is
        # data in that dictionary
        if dict_column_values:
            # Extract primary keys of that table (will not populate analyses of primary keys such as MAR_section_depths)
            if 'replicate' not in dict_column_values.keys():
                dict_column_values['replicate'] = 1
            core_analyzed, pks = _pks_in_table(dict_column_values=dict_column_values, sql_table=sql_table)
            not_null_variables = general.not_null_sql_table(sql_table)
            for column in columns:
                if column not in pks + not_null_variables and column in dict_column_values.keys():
                    # Add each analysis in the core_metadata as long as it is an analysis (not a primary key)
                    if pd.notna(dict_column_values[column]):
                        # Add an entry to the core_metadata table for every variables that has been analyzed (not nan)
                        _insert_core_metadata_table(df_row=df_row, core_id=core_id, core_analysis=column,
                                                    sql_table='core_metadata',
                                                    replicate=dict_column_values['replicate'],
                                                    article_author_metadata=article_author_metadata)
    return sql_table_variables_dict


def _insert_core_analysis_table(core_id, df_row, columns, sql_table):
    # Dictionary of column and value to go in that column if value is not NaN
    dict_column_values = _variables_to_populate_dict(df_row, columns)
    if dict_column_values:
        # If there are values in dict_column_values (not empty)
        # The dictionary can have empty values if that specific row doesn't have any information to be added to
        # the sql_table (maybe there is information for different tables)
        with print_with_time(f'Inserting {columns} of {df_row["core_name"]["data"]} (core_id {core_id}) '
                             f'into {sql_table}'):
            dict_column_values['core_id'] = core_id
            # Query if the primary keys (core_id and any other primary key) is already in the table
            # Example, one core_id can have several SARs with different models, tracers, section depths, etc.
            core_analyzed, pks = _pks_in_table(dict_column_values=dict_column_values, sql_table=sql_table)
            if core_analyzed.empty:
                # No data was returned from that table with those primary keys (core_id), so we can populate it
                # Insert in database
                _insert(dict_column_values=dict_column_values, sql_table=sql_table)
                return dict_column_values
            else:
                # Data for that core_id (and other primary keys) are already in the database.
                # Add missing variables and prompt the user if different variables need to be updated.
                dict_column_values_updated = _update_table(dict_column_values=dict_column_values, sql_table=sql_table)
                if dict_column_values_updated:
                    dict_column_values_updated['core_id'] = core_id
                    if 'replicate' not in dict_column_values_updated.keys():
                        dict_column_values_updated['replicate'] = 1
                return dict_column_values_updated


def _insert_core_metadata_table(df_row, core_id, core_analysis, replicate, sql_table, article_author_metadata):
    with print_with_time(
            f'Inserting core metadata of {core_analysis} for {df_row["core_name"]["data"]} (core_id {core_id})'):
        columns = df_row.columns.values
        db_cursor = conn.cursor()
        # Extract core_analysis_id
        core_analysis_id = _query_id_null(sql_table='core_analysis_description', id_name='core_analysis_id',
                                          dict_variable_value={'core_analysis': core_analysis})
        assert core_analysis_id is not None, ValueError(f'No core_analysis_id found for {core_analysis}.')
        exclusivity_clause_id = _exclusivity_clause(df_row=df_row, exclusivity_clause='exclusivity_clause')
        core_analysis_calculated = _extract_metadata_columns(df_row_series=df_row[core_analysis],
                                                             metadata_field='raw_calculated')
        # Extract core_article_id
        if 'title' in columns and pd.notna(df_row['title']['data']):
            title = df_row['title']['data']
            core_article_id = article_author_metadata[title]
        else:
            core_article_id = 0
        # Extract core_contact_person_id
        if {'author_firstname', 'author_lastname'}.issubset(set(columns)):
            core_author_lastname = df_row['author_lastname']['data']
            core_author_firstname = df_row['author_firstname']['data']
            core_author = (core_author_firstname, core_author_lastname)
            core_author_id = article_author_metadata[core_author]
        else:
            # If there is no author information in the Excel, find the first author of the paper the data belong to
            if core_article_id > 0:
                db_cursor.execute(f'''SELECT author_id FROM {schema}.authorship WHERE article_id={core_article_id}''')
                author_id = db_cursor.fetchall()
                assert author_id is not None, ValueError(f'No author_id assigned to article id {core_article_id} '
                                                         f'in authorship')
                core_author_id = author_id[0][0]
            else:
                core_author_id = 0
        # Create a dictionary of column:value that needs to be inserted into the table
        dict_column_values = {'core_id': core_id, 'core_analysis_id': core_analysis_id,
                              'core_exclusivity_clause': exclusivity_clause_id,
                              'core_analysis_calculated': core_analysis_calculated, 'core_article_id': core_article_id,
                              'core_contact_person_id': core_author_id,
                              'entry_user_name': df_row['entry_user_name']['data'],
                              'replicate': replicate}
        metadata = _query_id_null(sql_table=sql_table, id_name='core_id', dict_variable_value=dict_column_values)
        if not metadata:
            _insert(dict_column_values=dict_column_values, sql_table=sql_table)
        else:
            # Already in database, log a warning stating the values that are already in the database
            logger.warning(f'Core metadata of {dict_column_values} already in database.')
        db_cursor.close()


# 6. Populate sample analyses
def insert_sample_analyses(df, core_id_dict, article_author_metadata):
    """
    Populates the database with data of SAMPLE_ANALYSES:
    1. Populates general information of the sample (sample_name, section depth, core_id, etc.) and obtains sample_id
    2. Populates sample analysis tables:
        2.1. Populates (or updates) each table of sample_analysis (e.g. sample_composition)
        2.2. Harmonizes sample analysis.
        2.3. Populates sample metadata
    :param df: DataFrame with all the information to be populated
    :param core_id_dict: Dictionary of core_id: core_name of data that will be populated
    :param article_author_metadata: Dictionary of article_name: article_id, author_name: author_id of the data that
                                    will be populated
    :return: None
    """
    with print_with_time('\nInserting sample analysis data\n'):
        for i, row in df.iterrows():
            # Convert each row into a dataframe with the metadata
            df_row = _transform_multiidex_row_samples(row)
            sample_id, core_id = _insert_sample_table(df_row=df_row, core_id_dict=core_id_dict, sql_table='samples')
            _insert_sample_analysis_tables(df_row=df_row, sample_id=sample_id, core_id=core_id,
                                           article_author_metadata=article_author_metadata)
            conn.commit()


def _insert_sample_table(df_row, core_id_dict, sql_table):
    with print_with_time(f'Inserting data related to {df_row["sample_name"]["data"]} into {sql_table}'):
        # Construct a dictionary with all the possible columns that are found in the dataframe and in the database
        sample_table_dict = _variables_to_populate_dict(df_row, columns_table(sql_table))
        core_id = core_id_dict[str(df_row['core_name']['data'])]
        sample_table_dict['core_id'] = core_id
        # Harmonize sample depths
        sample_table_dict = harmonization.sample_depth_harmonization(sample_table_dict)
        # Query if that sample_id is already in the table using the following variables (do not use sample_name because
        # if the sample name refers to another core that is already in the database, it will not be the same and will
        # always return that it isn't in the database
        variables = ['sample_depth_upper_cm', 'sample_depth_bottom_cm', 'sample_depth_average_cm', 'core_id']
        sample_table_dict_temp = {variable: value for variable, value in sample_table_dict.items() if
                                  variable in variables and pd.notna(sample_table_dict[variable])}
        sample_id = _query_id_null(sql_table=sql_table, id_name='sample_id', dict_variable_value=sample_table_dict_temp)
        if not sample_id:
            # There is no sample_id in the table, so it needs to be added
            sample_id = _insert_and_query_id(id_name='sample_id',
                                             dict_column_values=sample_table_dict,
                                             sql_table=sql_table)
        else:
            _update_table(dict_column_values=sample_table_dict, sql_table=sql_table)
            logger.warning(f'Data of {df_row["sample_name"]["data"]} (sample_id {sample_id}) for {sql_table} already '
                           f'in database.')
        return sample_id, core_id


def _insert_sample_analysis_tables(df_row, sample_id, core_id, article_author_metadata):
    # Create a dictionary of sql table (key) with a list of variables that will be added in it (value)
    sql_table_variables_dict = _dictionary_analysis_table(df_row=df_row, analysis_source='sample')
    for sql_table, columns in sql_table_variables_dict.items():
        # Group data based on material_analyzed (primary key of all the tables) and extract value to be populated
        dict_material_analyzed_table = _dictionary_levels_table(df_row, columns, level='material_analyzed')
        for material_analyzed, df_material_analyzed in dict_material_analyzed_table.items():
            # Group data based on method
            dict_method_table = _dictionary_levels_table(df_material_analyzed, columns, level='method')
            for method, df_method in dict_method_table.items():
                # Group data based on method details
                dict_method_details_table = _dictionary_levels_table(df_method, columns, level='method_details')
                for method_details, df_method_details in dict_method_details_table.items():
                    # Group data whether it has been calculated or not
                    dict_calculated_table = _dictionary_levels_table(df_method_details, columns, level='raw_calculated')
                    for calculated, df_calculated in dict_calculated_table.items():
                        # For each group of data, extract the data in a dictionary
                        dict_column_values = _variables_to_populate_dict(df_calculated, columns)
                        dict_column_values = _insert_sample_analysis_table(sample_id=sample_id, df_row=df_row,
                                                                           columns=columns, sql_table=sql_table,
                                                                           material_analyzed=material_analyzed,
                                                                           dict_column_values=dict_column_values)
                        # If data has been added to a sample analysis table, it will return a dictionary of added values
                        # If no data was added to a sample analysis table, it will return None, so we first need to
                        # check if there is data in that dictionary
                        if dict_column_values:
                            # If there is data, populate the sample_metadata
                            # Extract primary keys of that table
                            # (will not populate analyses of primary keys such as material_analyzed)
                            sample_analyzed, pks = _pks_in_table(dict_column_values=dict_column_values,
                                                                 sql_table=sql_table)
                            for column in dict_column_values.keys():
                                if column not in pks:
                                    # Add each analysis in the sample_metadata as long as it is an analysis
                                    # (not a primary key)
                                    if column in columns_table(sql_table) and \
                                            (not isinstance(dict_column_values[column], list)
                                             and pd.notna(dict_column_values[column])):
                                        # Query if data is already in metadata table
                                        # Add an entry to the core_metadata table for every variables that has
                                        # been analyzed (not nan)
                                        _insert_sample_metadata_table(df_row=df_row, sample_id=sample_id,
                                                                      sample_analysis=column,
                                                                      core_id=core_id,
                                                                      material_analyzed_id=dict_column_values[
                                                                          'material_analyzed_id'],
                                                                      replicate=dict_column_values['replicate'],
                                                                      sample_analysis_calculated=
                                                                      dict_column_values['sample_analysis_calculated'],
                                                                      article_author_metadata=article_author_metadata,
                                                                      method=method, method_details=method_details,
                                                                      sql_table='sample_metadata')


def _insert_sample_analysis_table(sample_id, df_row, columns, sql_table, dict_column_values, material_analyzed):
    # Dictionary of column and value to go in that column if value is not NaN
    if dict_column_values:
        # If there is data to add, compliment with sample_id and material_analyzed
        with print_with_time(f'Inserting {columns} of sample {df_row["sample_name"]["data"]} '
                             f'(sample_id {sample_id}) into {sql_table} '):
            dict_column_values['material_analyzed_id'] = _material_analyzed_id(material_analyzed)
            dict_column_values['sample_id'] = sample_id
            dict_column_values['sample_analysis_calculated'] = _columns_calculated(df_row, material_analyzed)
            # Perform data harmonization if needed
            if sql_table in harmonization.harmonization_functions.keys():
                dict_column_values = harmonization.harmonization_search(sql_table, dict_column_values)
            # Query if the primary keys (sample_id and any other primary key) are already in the table
            sample_analyzed, pks = _pks_in_table(dict_column_values=dict_column_values, sql_table=sql_table)
            if sample_analyzed.empty:
                # No data was returned from that table with those primary keys (sample_id), so we can populate it
                # Insert in database
                _insert(dict_column_values=dict_column_values, sql_table=sql_table)
                dict_column_values['replicate'] = 1
                return dict_column_values
            else:
                # That sql_table has entries for that sample_id and material_analyzed.
                # Check if there is new data to be added to that table and if there are values that could be updated
                dict_column_values_updated = _update_table(dict_column_values=dict_column_values, sql_table=sql_table)
                if dict_column_values_updated:
                    dict_column_values_updated['sample_id'] = sample_id
                    dict_column_values_updated['material_analyzed_id'] = dict_column_values['material_analyzed_id']
                    dict_column_values_updated['sample_analysis_calculated'] = \
                        dict_column_values['sample_analysis_calculated']
                    if 'replicate' not in dict_column_values_updated.keys():
                        dict_column_values_updated['replicate'] = 1
                return dict_column_values_updated


def _insert_sample_metadata_table(df_row, sample_id, core_id, sample_analysis, material_analyzed_id, replicate,
                                  sample_analysis_calculated, article_author_metadata, method, method_details,
                                  sql_table):
    """
    Populates the sample metadata table
    :param df_row:
    :param sample_id:
    :param core_id:
    :param sample_analysis: name of analysis
    :param material_analyzed_id:
    :param sample_analysis_calculated: list of variables (analyses) that have been calculated
    :param article_author_metadata:
    :param sql_table:
    :return:
    """
    with print_with_time(f'Inserting sample metadata of analysis {sample_analysis} for {df_row["sample_name"]["data"]} '
                         f'(sample_id: {sample_id}, material analyzed: {material_analyzed_id})'):
        columns = df_row.keys().values
        db_cursor = conn.cursor()
        # Extract sample_analysis_id
        db_cursor.execute(f'''SELECT sample_analysis_id FROM {schema}.sample_analysis_description WHERE 
                            sample_analysis = '{sample_analysis}' ''')
        sample_analysis_id = db_cursor.fetchone()
        assert sample_analysis_id is not None, ValueError(f'No core_analysis_id found for {sample_analysis}.')
        sample_analysis_id = sample_analysis_id[0]
        # Extract exclusivity clause
        exclusivity_clause_id = _exclusivity_clause(df_row=df_row, exclusivity_clause='exclusivity_clause')
        # Extract sample_analysis_calculated.
        if sample_analysis in sample_analysis_calculated:
            sample_analysis_calculated = 1  # True
        else:
            sample_analysis_calculated = 0  # False
        # Extract sample_article_id
        if 'title' in columns and pd.notna(df_row['title']['data']):
            title = df_row['title']['data']
            sample_article_id = article_author_metadata[title]
        else:
            sample_article_id = 0
        # Extract sample_contact_person_id
        if {'author_firstname', 'author_lastname'}.issubset(set(columns)):
            sample_author_lastname = df_row['author_lastname']['data']
            sample_author_firstname = df_row['author_firstname']['data']
            sample_author = (sample_author_firstname, sample_author_lastname)
            sample_author_id = article_author_metadata[sample_author]
        else:
            # If there is no author information in the Excel, find the first author of the paper the data belong to
            if sample_article_id > 0:
                db_cursor.execute(f'''SELECT author_id FROM {schema}.authorship WHERE article_id={sample_article_id}''')
                author_id = db_cursor.fetchall()
                assert author_id is not None, ValueError(f'No author_id assigned to article id {sample_article_id} '
                                                         f'in authorship')
                sample_author_id = author_id[0][0]
            else:
                sample_author_id = 0
        # Create a dictionary of column:value that needs to be inserted into the table
        dict_column_values = {'sample_id': sample_id, 'material_analyzed_id': material_analyzed_id,
                              'replicate': replicate,
                              'sample_analysis_id': sample_analysis_id,
                              'sample_exclusivity_clause': exclusivity_clause_id, 'sample_core_id': core_id,
                              'sample_analysis_calculated': sample_analysis_calculated,
                              'sample_article_id': sample_article_id,
                              'sample_contact_person_id': sample_author_id,
                              'entry_user_name': df_row['entry_user_name']['data'],
                              'method': method,
                              'method_details': method_details}
        metadata_pks, pks_list = _pks_in_table(dict_column_values, sql_table)
        dict_column_values_pks = {key: value for key, value in dict_column_values.items() if key in pks_list}
        metadata = _query_id_null(sql_table=sql_table, id_name='sample_id', dict_variable_value=dict_column_values_pks)
        if not metadata:
            _insert(dict_column_values=dict_column_values, sql_table=sql_table)
        else:
            # Check if data is already in the database (not only the primary keys) and update it
            metadata = _query_id_null(sql_table=sql_table, id_name='sample_id', dict_variable_value=dict_column_values)
            if not metadata:
                _update_table(dict_column_values=dict_column_values, sql_table=sql_table)
            else:
                # Already in database, log a warning stating the values that are already in the database
                logger.warning(f'Sample metadata of {sample_id} already in database.')
        db_cursor.close()


def _find_analysis_table(variable, analysis_source):
    """
    Extracts the table where the variables that needs to be added is in.
    :param variable: Variable name that needs to be added
    :param analysis_source: Specify whether the analysis is from a sample ('sample) or from a core ('core')
    :return: name of the table
    """
    # Identify which type of analysis you are searching for (core or sample)
    if analysis_source == 'core':
        analysis_type_id_column = 'core_analysis_type_id'
        analysis_description_table = 'core_analysis_description'
        analysis_type = 'core_analysis_type'
        analysis = 'core_analysis'
    elif analysis_source == 'sample':
        analysis_type_id_column = 'sample_analysis_type_id'
        analysis_description_table = 'sample_analysis_description'
        analysis_type = 'sample_analysis_type'
        analysis = 'sample_analysis'
    else:
        raise ValueError(f'analysis_source was "{analysis_source}" while it should be either "core" or "sample".')
    # Variables not to check
    metadata_list = ['core_name', 'exclusivity_clause', 'title', 'author_firstname', 'author_lastname',
                     'sample_name', 'sample_depth_upper_cm', 'sample_depth_bottom_cm', 'sample_depth_average_cm',
                     'entry_user_name', 'sample_comment']
    # Query table core_analysis_type to find in which table to store the data
    db_cursor = conn.cursor()
    db_cursor.execute(f'''SELECT {analysis} FROM {schema}.{analysis_description_table}''')
    analysis_column = db_cursor.fetchall()
    analysis_column = [item for t in analysis_column for item in t]  # Convert into a list
    if variable in analysis_column:  # If the variables is in the analysis column extracted, execute the query
        db_cursor.execute(f'''SELECT {analysis_type_id_column} FROM {schema}.{analysis_description_table} WHERE 
                            {analysis} = '{variable}' ''')
        analysis_type_id = db_cursor.fetchone()  # Analysis id of each variables
        assert analysis_type_id is not None, f'Variable "{variable}" could not be found in the database.'
        analysis_type_id = analysis_type_id[0]
        # Query table core_analysis_type to find in which table to store the data
        db_cursor.execute(f'''SELECT {analysis_type} FROM {schema}.{analysis_type} WHERE {analysis_type_id_column} = 
                            {analysis_type_id}''')
        sql_table = db_cursor.fetchone()
        assert sql_table is not None, f'Variable "{analysis_type_id}" could not be found in {analysis_type}.'
        sql_table = sql_table[0]
    elif variable in metadata_list:
        sql_table = None
    else:
        logger.error(f'Variable {variable} not found in any table')
        sys.exit()
    db_cursor.close()
    return sql_table


def _insert(dict_column_values, sql_table):
    # Converts dictionary to list of tuples, eliminating any None values, and converting values into strings to
    # insert in the database
    variables_to_insert = [(column, str(value)) for column, value in dict_column_values.items() if
                           column in columns_table(sql_table) and value is not None and pd.notna(value)]
    if len(variables_to_insert) > 0:
        # Separates into columns and values
        columns, values = tuple(zip(*variables_to_insert))  # * indicates the variables is a string
        # Converts columns and values into readable variables for MySQL
        columns = ', '.join([f'"{column}"' for column in columns])
        values = ', '.join(f"'{value}'" for value in values)
        db_cursor = conn.cursor()
        db_cursor.execute(f'INSERT INTO {schema}."{sql_table}" ({columns}) VALUES ({values})')
        db_cursor.close()


def _insert_and_query_id(id_name, dict_column_values, sql_table):
    _insert(dict_column_values=dict_column_values, sql_table=sql_table)
    db_id = _query_id_null(sql_table=sql_table, id_name=id_name, dict_variable_value=dict_column_values)
    assert db_id, LookupError(f'Error querying the id {id_name} of recently added data ({dict_column_values}) '
                              f'in {sql_table}')
    return db_id


def _query_id(sql_table, id_name, dict_variable_value):
    """
    Query the table to see if the exact same values are in the database
    :param sql_table:
    :param id_name:
    :param dict_variable_value:
    :return: core_id or sample_id
    """
    # Make sure that the dictionary does not have any NaN or None values
    dict_variable_value = {column: value for column, value in dict_variable_value.items() if pd.notna(value)}
    assert dict_variable_value, ValueError(f'No data of {id_name} to query from {sql_table}')
    columns_info = sql_table_information(sql_table=sql_table)
    where_clause = []
    for column, value in dict_variable_value.items():
        # For each column, query the data type and assemble the SQL query based on that
        query_exact_data_str = query_exact_data(column=column, value=value, columns_info=columns_info,
                                                sql_table=sql_table)
        if query_exact_data_str:
            where_clause.append(query_exact_data_str)
    db_cursor = conn.cursor()
    db_cursor.execute(f'''SELECT {id_name} FROM {schema}."{sql_table}" WHERE {' and '.join(where_clause)}''')
    query = db_cursor.fetchall()
    assert len(query) <= 1, ValueError(f"Duplicate entry in {sql_table} for id {id_name} "
                                       f"({dict_variable_value}).")
    if (isinstance(query, list) or isinstance(query, tuple)) and len(query) > 0:
        query = query[0][0]
    db_cursor.close()
    return query


def _query_replicate(sql_table, dict_variable_value):
    """
    Query the table to see if the exact same values are in the database
    :param sql_table:
    :param dict_variable_value:
    :return: core_id or sample_id
    """
    # Make sure that the dictionary does not have any NaN or None values
    dict_variable_value = {column: value for column, value in dict_variable_value.items() if pd.notna(value)}
    assert dict_variable_value, ValueError(f'No data of replicate to query from {sql_table}')
    columns_info = sql_table_information(sql_table=sql_table)
    where_clause = []
    for column, value in dict_variable_value.items():
        # For each column, query the data type and assemble the SQL query based on that
        query_exact_data_str = query_exact_data(column=column, value=value, columns_info=columns_info,
                                                sql_table=sql_table)
        if query_exact_data_str:
            where_clause.append(query_exact_data_str)
    db_cursor = conn.cursor()
    db_cursor.execute(f'''SELECT replicate FROM {schema}."{sql_table}" WHERE {' and '.join(where_clause)}''')
    query = db_cursor.fetchall()
    if (isinstance(query, list) or isinstance(query, tuple)) and len(query) > 0:
        replicate_list = [replicate[0] for replicate in query]
        last_replicate = replicate_list[-1]
    else:
        last_replicate = query[-1]
    db_cursor.close()
    return last_replicate


def _query_id_null(sql_table, id_name, dict_variable_value):
    """
    Query the table to see if the exact same values are in the database
    :param sql_table:
    :param id_name:
    :param dict_variable_value:
    :return: core_id or sample_id
    """
    assert dict_variable_value, ValueError(f'No data of {id_name} to query from {sql_table}')
    columns_info = sql_table_information(sql_table=sql_table)
    where_clause = []
    for column, value in dict_variable_value.items():
        if pd.notna(value):
            # For each column, query the data type and assemble the SQL query based on that
            query_exact_data_str = query_exact_data(column=column, value=value, columns_info=columns_info,
                                                    sql_table=sql_table)
            if query_exact_data_str:
                where_clause.append(query_exact_data_str)
        else:
            # If the value is None, NaN, etc, construct a SQL query taking this into account
            null_query = f'"{sql_table}"."{column}" IS NULL'
            where_clause.append(null_query)
    db_cursor = conn.cursor()
    db_cursor.execute(f'''SELECT {id_name} FROM {schema}."{sql_table}" WHERE {' and '.join(where_clause)}''')
    query = db_cursor.fetchall()
    assert len(query) <= 1, ValueError(f"Duplicate entry in {sql_table} for id {id_name} "
                                       f"({dict_variable_value}).")
    if (isinstance(query, list) or isinstance(query, tuple)) and len(query) > 0:
        query = query[0][0]
    db_cursor.close()
    return query


def _query_notnull_analyses(sql_table, id_name, dict_variable_value, variables):
    """
    Queries the database table to see which variables are empty (NULL)
    :param sql_table: SQL table that needs to be queried
    :param id_name: Name of id (primary key) of the SQL table
    :param dict_variable_value: Dictionary of variables (column) and its value that needs to be queried (usually the id,
                                or any other primary key or related variables)
    :param variables: Variables that will be queried if they are NULL in the SQL table
    :return: ID (primary key) from the database that has NULL values in the given variables.
    """
    db_cursor = conn.cursor()
    where_clause = ' AND '.join(f"{variable} = '{value}'" for variable, value in dict_variable_value.items())
    where_notnull_clause = ' AND '.join(f'"{variable}" IS NOT NULL' for variable in variables)
    db_cursor.execute(f'''SELECT {id_name} FROM {schema}."{sql_table}" WHERE {where_clause}
                        AND {where_notnull_clause}''')
    query = db_cursor.fetchall()
    db_cursor.close()
    return query


def _update_table(dict_column_values, sql_table):
    """
    Follows a sequence of steps to determine if the variables need to be updated in the database
    1. Updates variables that are empty (NULL) in the database
    2. Queries if the data to be added is different than the data in the database
    3. Prompts the user to know which of these variables needs to be updated in the database
    :param dict_column_values: Dictionary of column and its value to be populated in the database
    :param sql_table: SQL table to be populated
    :return: None
    """
    # Make sure the values that need to be updated are not Null (two different if statements in case value is a list)
    dict_column_values = {column: value for column, value in dict_column_values.items() if
                          (not isinstance(value, list) and pd.notna(value)) or
                          (isinstance(value, list) and value is not None)}
    # Extract in a dataframe the field that needs to be updated based on their primary keys
    df_pks_in_table, pks = _pks_in_table(dict_column_values=dict_column_values, sql_table=sql_table)
    pks_dict = {pk: dict_column_values[pk] for pk in pks if pk in dict_column_values.keys() and
                pd.notna(dict_column_values[pk])}
    # Primary key query (i.e. primary_key_1 = value_1 AND primary_key_2 = value_2)
    pks_value_query = ' AND '.join(f'''"{pk}" = '{dict_column_values[pk]}' ''' for pk in pks if pk in
                                   dict_column_values.keys() and pd.notna(dict_column_values[pk]))
    dict_column_values_added = {}
    # 1. Update variables that are null in the database and returns dictionary of variables that were added
    updated_null_columns = _update_null_columns(dict_column_values=dict_column_values, sql_table=sql_table,
                                                df_sql_table=df_pks_in_table, pks=pks,
                                                pks_value_query=pks_value_query)
    if updated_null_columns:
        dict_column_values_added.update(updated_null_columns)
    # 2. Query the variables that are exactly the same in the database (those will be ignored)
    same_columns = _query_columns_exact(dict_column_values=dict_column_values, sql_table=sql_table,
                                        pks_value_query=pks_value_query)
    same_columns_dict = {column: [dict_column_values[column]] for column in same_columns}
    df_same_columns = pd.DataFrame.from_dict(same_columns_dict)
    # 3. Query the variables that are different in the database, and prompt the user if these should be updated
    # Extract a dictionary of columns that are different from those in the database
    dict_column_values_different_in_db = {column: values for column, values in dict_column_values.items() if column
                                          not in df_same_columns.columns
                                          and column not in ['geopoints_point', 'sample_analysis_calculated',
                                                             'core_name', 'sample_name']
                                          and column in columns_table(sql_table)
                                          and pd.notna(dict_column_values[column])}
    if dict_column_values_different_in_db:
        # If there are variables in the dictionary, prompt the user if they need to be updated in the database
        updated_columns = _prompt_and_update_columns(dict_column_values=dict_column_values,
                                                     dict_column_values_different_in_db=
                                                     dict_column_values_different_in_db,
                                                     sql_table=sql_table, df_pks_in_table=df_pks_in_table,
                                                     pks=pks,
                                                     pks_value_query=pks_value_query)
        if updated_columns:
            dict_column_values_added.update(updated_columns)
    else:
        # Data already in the database
        logger.warning(f'Data of {list(df_same_columns.columns.get_level_values(0))} for {pks_dict} in {sql_table} '
                       f'already in the database.')
    return dict_column_values_added


def _prompt_and_update_columns(dict_column_values, dict_column_values_different_in_db,
                               sql_table, df_pks_in_table, pks, pks_value_query):
    # Dictionary of primary keys and its values
    pks_dict = {pk: dict_column_values[pk] for pk in pks if pk in dict_column_values.keys() and
                pd.notna(dict_column_values[pk])}
    # Dictionary of variables and its values that are the same in the database
    update_col = _my_input(f'Data to be added {dict_column_values_different_in_db} has similar entries in the '
                           f'database: \n \n {df_pks_in_table} \n \n'
                           f'Should some of these variables be updated (y) or not (n), or is it a replicate (r)?',
                           possible_values=['y', 'n', 'r'])
    if update_col == 'y':
        if df_pks_in_table.shape[0] > 1:
            possible_values = [x for x in list(range(df_pks_in_table.shape[0]))]
            row_to_update = _my_input(f'Which row needs to be updated?',
                                      possible_values=possible_values, to_type_fn=int)
            pks_value_query = {pk:df_pks_in_table[pk][row_to_update] for pk in pks}
            return _update_data(dict_column_values, dict_column_values_different_in_db,
                                sql_table, df_pks_in_table.iloc[[row_to_update]], pks_dict, pks_value_query)
        else:
            return _update_data(dict_column_values, dict_column_values_different_in_db,
                                sql_table, df_pks_in_table, pks_dict, pks_value_query)
    elif update_col == 'r':
        # Data is a replicate analysis of that sample. Add it to the database using an incremental replicate entry
        # Query the database to see what was the replicate value
        replicate = _query_replicate(sql_table=sql_table,
                                     dict_variable_value=pks_dict)
        if replicate:
            replicate += 1
            dict_column_values_different_in_db['replicate'] = replicate
            # Remove replicate because it is already added in dict_column_values_different_in_db
            pks_dict = {k: pks_dict[k] for k in pks_dict.keys() if k != 'replicate'}
            new_dict = {**dict_column_values_different_in_db, **pks_dict}
            _insert(dict_column_values=new_dict, sql_table=sql_table)
            return new_dict
    else:
        logger.warning(f'No data was updated of {pks_dict} in {sql_table}.')


def _update_data(dict_column_values, dict_column_values_different_in_db,
                 sql_table, df_pks_in_table, pks_dict, pks_value_query):
    max_number_of_variables_to_populate = len(dict_column_values_different_in_db.keys())
    possible_values = [x + 1 for x in list(range(max_number_of_variables_to_populate))]
    possible_values.append(0)
    if max_number_of_variables_to_populate > 1:
        # If there's more than one variables that could be updated, prompt the user which one it should be
        number_of_variables_to_populate = _my_input(message_input=f'How many variables should be updated in the '
                                                                  f'DataFrame?',
                                                    possible_values=possible_values,
                                                    to_type_fn=int)
        if number_of_variables_to_populate == max_number_of_variables_to_populate:
            # Replace all the variables in the database
            variables_to_populate = list(dict_column_values_different_in_db.keys())
        else:
            if number_of_variables_to_populate == 0:
                # The user made a mistake saying it had to add data to the database.
                variables_to_populate = []
            else:
                # Only some variables need to be updated. Prompt the user which variables need to be populated.
                variables_to_populate = []
                possible_values = list(dict_column_values_different_in_db.keys())
                for i in range(number_of_variables_to_populate):
                    variable_to_populate = _my_input(f'{i + 1}: which variables should be updated in {sql_table}?',
                                                     possible_values=possible_values)
                    variables_to_populate.append(variable_to_populate)
                    possible_values.remove(variable_to_populate)
    else:
        # The only variables needs to be updated in the database, without prompting the user
        variables_to_populate = list(dict_column_values_different_in_db.keys())
    dict_variables_to_populate = {column: dict_column_values[column] for column in variables_to_populate}
    sql_variables_values_to_populate = ', '.join(f'''"{column}" = '{dict_column_values[column]}' ''' for column
                                                 in variables_to_populate)
    sql_pks_value_query = ' AND '.join(f'''"{column}" = '{value}' ''' for column, value in pks_dict.items())
    # Convert into DataFrame to easily show it in the Log
    dict_column_values_in_db = df_pks_in_table[variables_to_populate].to_dict(orient='list')
    if variables_to_populate:
        logger.warning(f'Updating columns {dict_column_values_in_db} to '
                       f'{sql_variables_values_to_populate} for {pks_dict} in {sql_table}.')
        _update(sql_table=sql_table,
                sql_variables_values=sql_variables_values_to_populate,
                sql_where_statement=sql_pks_value_query)
    return dict_variables_to_populate


def _update_null_columns(dict_column_values, sql_table, df_sql_table, pks, pks_value_query):
    """
    Updates only variables that are NULL in the database for the given primary keys
    :param dict_column_values: Dictionary of column and values that need to be populated
    :param sql_table: SQL table that needs to be populated
    :param df_sql_table: DataFrame of primary key that needs to be populated in the SQL table
    :param pks: List of primary keys of that SQL table
    :param pks_value_query: SQL statement to query the SQL table and update only columns from that row
    :return: None
    """
    # Extract a list of columns that have NaN values and that have values to be populated.
    # We will only update those values
    columns_null_in_db_to_update = [column for column in dict_column_values.keys() if
                                    column in columns_table(sql_table=sql_table) and
                                    column not in df_sql_table.columns or (column in df_sql_table.columns and
                                                                           df_sql_table[column].isna().any())]
    # If there are columns that are null in the database:
    if columns_null_in_db_to_update:
        # Create a query of the variables that will be populated (are NaN in the database)
        sql_variables_values = ', '.join(f'''"{column}" = '{dict_column_values[column]}' ''' for column in
                                         columns_null_in_db_to_update)
        pks_dict = {pk: dict_column_values[pk] for pk in pks if pk in dict_column_values.keys()
                    and pd.notna(dict_column_values[pk])}
        logger.warning(f'Adding values {sql_variables_values} for {pks_dict} in {sql_table}')
        _update(sql_table=sql_table, sql_variables_values=sql_variables_values, sql_where_statement=pks_value_query)
        # Return a dictionary of column: value of variables that were updated
        dict_column_values_updated = {column: dict_column_values[column] for column in columns_null_in_db_to_update}
        return dict_column_values_updated


def _query_columns_exact(dict_column_values, sql_table, pks_value_query):
    """
    Queries the database to see which columns are exactly the same as those given. Returns a list of columns that are
    exactly the same in the database
    :param dict_column_values: Dictionary of columns and values that need to be queried in the database
    :param sql_table: SQL table that needs to be queried
    :param pks_value_query: Primary key query (only want to check the columns that are exact of that primary key)
    :return: A list of columns that are exactly the same in the database
    """
    db_cursor = conn.cursor()
    columns_info = sql_table_information(sql_table=sql_table)
    columns_exact = []
    for column, value in dict_column_values.items():
        # Sometimes the value is NaN (is instance makes sure it isn't a list, it would return an error)
        # Makes sure that the column to be queried is in the database
        if (not isinstance(value, list) and pd.notna(value)) and column in columns_table(sql_table=sql_table):
            sql_query = query_exact_data(column=column, value=value, columns_info=columns_info, sql_table=sql_table)
            if sql_query:
                db_cursor.execute(f'''SELECT * FROM {schema}."{sql_table}" WHERE {sql_query} AND {pks_value_query}''')
                df_out = pd.DataFrame(db_cursor.fetchall(), columns=[desc[0] for desc in db_cursor.description])
                if not df_out.empty:
                    columns_exact.append(column)
    db_cursor.close()
    return columns_exact


def _update(sql_table, sql_variables_values, sql_where_statement):
    """
    Updates entries in SQL table
    :param sql_table: SQL table that needs to be updated
    :param sql_variables_values: SQL statement of variables and values that will be updated.
                                The format of this statement needs to be like "column_name_1 = new_value_1 AND
                                columns_name_2 = value_2"
    :param sql_where_statement: SQL statement that establishes which entries need to be updated.
                                The format of this statement needs to be like "column_name_1 = value_1 AND
                                column_name_2 = value_2". For instance: "core_id = 146 AND
    :return: None
    """
    db_cursor = conn.cursor()
    db_cursor.execute(f'''UPDATE {schema}."{sql_table}" SET {sql_variables_values} WHERE {sql_where_statement}''')
    db_cursor.close()


def _update_old(dict_column_values, id_name, sql_table, material_analyzed=None):
    """
    Updates column values
    :param dict_column_values: Dictionary of column and value that needs to be updated
    :param id_name: Name of column where the id is stored
    :param sql_table: SQL table to update
    :param material_analyzed: For sample analyses tables, one of the primary keys is "material_analyzed",
                                which needs to be taken into account
    :return: Nothing
    """
    sql_statement = ', '.join(f'''"{column}" = '{value}' ''' for column, value in dict_column_values.items() if
                              pd.notna(value))
    id_value = dict_column_values[id_name]
    db_cursor = conn.cursor()
    if not material_analyzed:
        db_cursor.execute(f'''UPDATE {schema}."{sql_table}" SET {sql_statement} WHERE {id_name} = {id_value}''')
    else:
        db_cursor.execute(f'''UPDATE {schema}."{sql_table}" SET {sql_statement} WHERE {id_name} = {id_value}
                            AND material_analyzed = '{material_analyzed}' ''')
    db_cursor.close()


def _my_input(message_input, possible_values, to_type_fn=str):
    while True:
        try:
            res = to_type_fn(input(message_input + f"  {possible_values}"))
            assert isinstance(res, to_type_fn), "Error: maybe you messed up with the 'to_type_fn'?"
            if res in possible_values:
                return res
        except ValueError:
            pass
        print(f"Not a valid answer. Please choose between {possible_values}.")


def _input_yes(message_input):
    res = _my_input(message_input, possible_values=('y', 'n'))
    return res == 'y'  # returns True if input is yes


def _pks_in_table(dict_column_values, sql_table):
    """
    Function that queries the database based on the primary keys that need to be populated
    :param dict_column_values: Dictionary of variables and values that need to be populated
    :param sql_table: SQL table that needs to be populated
    :return: DataFrame of the variables that have the same primary keys as the data that needs to be populated,
    and a list of primary keys of that table
    """
    db_cursor = conn.cursor()
    # Extract all the primary keys from the sql_table
    db_cursor.execute(f'''SELECT pk.column_name, data_type FROM
                        (SELECT column_name, kcu.table_name
                        FROM information_schema.table_constraints tco
                        right join information_schema.key_column_usage kcu 
                             on kcu.constraint_name = tco.constraint_name
                             and kcu.constraint_schema = tco.constraint_schema
                             and kcu.table_name = tco.table_name
                        where tco.constraint_type = 'PRIMARY KEY'
                        AND tco.table_name = '{sql_table}') as pk
                        INNER JOIN information_schema.columns col 
                            USING (column_name, table_name)
                        WHERE col.table_name = '{sql_table}';''')
    df_pks = pd.DataFrame(db_cursor.fetchall(), columns=['column_name', 'data_type'])
    df_pks.set_index('column_name', inplace=True)
    pks = df_pks.index.to_list()
    # Extract the values of those primary keys, if they are provided and they are not _unknown
    # (if SAR_model is _unknown, that SAR may already be in the database, specifying the model, and that
    # variables would otherwise be added to the database - it needs to be prompted by the user)
    pks_values = {pk: dict_column_values[pk] for pk in pks if pk in dict_column_values.keys() and
                  dict_column_values[pk] != '_unknown'}
    sql_query_pks_list = []
    for pk, value in pks_values.items():
        if df_pks['data_type'][pk] == 'character varying':
            sql_query_pks_list.append(f'''("{pk}" = '{value}' OR "{pk}" = '_unknown')''')
        else:
            sql_query_pks_list.append(f'''("{pk}" = '{value}')''')
    sql_query_pks = ' AND '.join(sql_query_pks_list)
    db_cursor.execute(f'''SELECT * FROM {schema}."{sql_table}" WHERE {sql_query_pks}''')
    df_pks_in_table = pd.DataFrame(db_cursor.fetchall(), columns=[desc[0] for desc in db_cursor.description])
    df_pks_in_table.dropna(axis='columns', how='all', inplace=True)
    db_cursor.close()
    return df_pks_in_table, pks


def _variables_to_populate_dict(data, columns_table):
    """
    Creates a dictionary of columns and values that need to be inputed into the table
    :param data: Data that needs to be inputed
    :param columns_table: Columns of the table that need to be populated into
    :return: Dictionary of columns and values that will be added to the database
    """
    if len(data.shape) > 1:
        # It is a Dataframe
        if data.columns.nlevels > 1:
            return _variables_to_populate_multiindex(data, columns_table)
        else:
            return _variables_to_populate_dataframe(data, columns_table)
    else:
        return _variables_to_populate_simple_index(data, columns_table)


def _variables_to_populate_simple_index(row, columns_table):
    """
    Creates a dictionary of columns and values that need to be inputed into the table
    :param row: Row of data that needs to be inputed
    :param columns_table: Columns of the table that need to be populated into
    :return: Dictionary of columns and values that will be added to the database
    """
    variables_dict = {k: v for k, v in row.items() if not pd.isna(v)}
    # Create a new dictionary with data to be added to sampling_campaign
    variables_to_populate_dict = {column: value for column, value in variables_dict.items() if
                                  column in columns_table}
    return variables_to_populate_dict


def _dictionary_levels_table(df_row, columns, level):
    columns = [column for column in columns if column in df_row.columns]
    df_columns = df_row[set(columns)]
    if df_columns.columns.nlevels == 1:
        # Convert to multiindex
        arrays = [df_columns.columns.tolist(),
                  df_columns.iloc[0, :].to_list(),
                  df_columns.iloc[1, :].to_list(),
                  df_columns.iloc[2, :].to_list(),
                  df_columns.iloc[3, :].to_list()]
        df_columns.columns = pd.MultiIndex.from_arrays(arrays, names=['column', 'method', 'method_details',
                                                                      'material_analyzed', 'raw_calculated'])
        df_columns.drop(index=['method', 'method_details', 'material_analyzed', 'raw_calculated'], inplace=True)
    # Put default values
    if level == 'material_analyzed':
        df_columns.columns = _set_level_values(df_columns.columns,
                                               level=level,
                                               values=df_columns.columns.get_level_values(level).fillna('bulk'))
    elif level == 'method':
        df_columns.columns = _set_level_values(df_columns.columns,
                                               level=level,
                                               values=df_columns.columns.get_level_values(level).fillna('_unknown'))
    elif level == 'method_details':
        df_columns.columns = _set_level_values(df_columns.columns,
                                               level=level,
                                               values=df_columns.columns.get_level_values(level).fillna('_unknown'))
    elif level == 'raw_calculated':
        df_columns.columns = _set_level_values(df_columns.columns,
                                               level=level,
                                               values=df_columns.columns.get_level_values(level).fillna('Raw_data'))
    else:
        raise ValueError
    # Group by level as dictionaries
    dict_level_table = collections.defaultdict(dict)
    for level_cat in df_columns.columns.get_level_values(level).unique():
        df_level_cat = df_columns.xs(key=level_cat, axis=1, level=level)
        dict_level_table[level_cat] = df_level_cat
    return dict_level_table


def _columns_calculated(df_row, material_analyzed):
    # Extract the column names of the specific material_analyzed that is calculated
    df_material_analyzed = df_row.loc[:, df_row.loc['material_analyzed'] == material_analyzed]
    calculated_columns = df_material_analyzed.columns[df_material_analyzed.loc['raw_calculated']
                                                      == 'Calculated_data'].to_list()
    return calculated_columns


def _variables_to_populate_multiindex(df_row, columns_table):
    variables_dict = {}
    for column in columns_table:
        if column in df_row.columns:
            data = df_row[column].loc['data'].iloc[0]
            if pd.notna(data):
                variables_dict[column] = data
    return variables_dict


def _variables_to_populate_dataframe(df_row, columns_table):
    variables_dict = {}
    for column in columns_table:
        if column in df_row.columns:
            data = df_row[column]['data']
            if pd.notna(data):
                variables_dict[column] = data
    return variables_dict


def _exclusivity_clause(df_row, exclusivity_clause='exclusivity_clause'):
    """
    Extracts whether exclusivity clause data from a row in the DataFrame
    :param df_row: Row in the DatFrame
    :param exclusivity_clause: Name of the column where exclusivity_clause is stored
    :return: exclusivity_clause_id (1 if True, 0 is False or not provided)
    """
    if exclusivity_clause in df_row.columns and df_row[exclusivity_clause]['data']:
        exclusivity_clause_id = 1  # True
    else:
        exclusivity_clause_id = 0  # False
    return exclusivity_clause_id


def _df_to_window(df):
    win = QWidget()
    scroll = QScrollArea()
    layout = QVBoxLayout()
    table = QTableWidget()
    scroll.setWidget(table)
    layout.addWidget(table)
    win.setLayout(layout)
    table.setColumnCount(len(df.columns))
    table.setRowCount(len(df.index))
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            table.setItem(i, j, QTableWidgetItem(str(df.iloc[i, j])))
    win.show()


def move_file(filename, dir_output, file_dir=None):
    # Make sure that the output directory exists
    if not os.path.exists(dir_output):
        os.mkdir(dir_output)
    # Move file to new destination
    shutil.move(os.path.join(file_dir, filename),
                os.path.join(dir_output, filename))


def _material_analyzed_id(material_analyzed, sql_table='sample_material_analyzed', id_name='material_analyzed_id'):
    # Query the database to extract the ID of the material_analyzed
    material_analyzed_id = _query_id(sql_table, id_name, {'material_analyzed': material_analyzed})
    return material_analyzed_id


def _method(df_row, variable, material_analyzed):
    if variable in df_row.columns:
        if isinstance(df_row[variable], pd.DataFrame):
            whole_column = df_row[variable].loc[:, df_row[variable].loc['material_analyzed'] == material_analyzed].squeeze()
        else:
            whole_column = df_row[variable]
        if whole_column.empty:
            whole_column = df_row[variable].loc[:, df_row[variable].loc['material_analyzed'].isna()].squeeze()
        method = whole_column.loc['method']
        if pd.isna(method):
            method = '_unknown'
        method_details = whole_column.loc['method_details']
        if pd.isna(method_details):
            method_details = None
    else:
        method = '_unknown'
        method_details = None
    return method, method_details


def _extract_metadata_columns(df_row_series, metadata_field):
    """
    Extracts the metadata (multi-index column) of material_analyzed or raw_calculated
    :param df_row_series:
    :param metadata_field:
    :return:
    """
    assert metadata_field == 'raw_calculated' or metadata_field == 'material_analyzed'
    if metadata_field == 'raw_calculated':
        if 'raw_calculated' not in df_row_series.index or df_row_series['raw_calculated'] == 'Raw_data' or \
                pd.isna(df_row_series['raw_calculated']):
            return 0  # Raw data (default)
        else:
            return 1  # Calculated data
    else:
        if 'material_analyzed' not in df_row_series.index or pd.isna(df_row_series['material_analyzed']):
            return 'bulk'  # Default value
        else:
            return df_row_series['material_analyzed']


def _extract_multiindex_dict(df, analyses_type):
    """
    Creates a dictionary of column name and material analyzed or raw/calculated
    :param df:
    :param analyses_type: 'material_analyzed' or 'raw_calculated'
    :return:
    """
    assert df.columns.nlevels > 1
    for multicolumn in df.columns:
        if multicolumn[0] in _find_analysis_table():
            pass
    if analyses_type == 'material_analyzed':
        return {multicolumn[0]: multicolumn[1] for multicolumn in df.columns if not \
            multicolumn[1].startswith('Unnamed')}
    else:
        return {multicolumn[0]: multicolumn[2] for multicolumn in df.columns if not \
            multicolumn[2].startswith('Unnamed')}


def _dictionary_analysis_table(df_row, analysis_source):
    """
    Creates a dictionary of variables and the SQL table where to find the variables in
    :param df_row:
    :param analysis_source:
    :return:
    """
    columns = df_row.columns.values
    # Create an empty dictionary where each sql table will be given a list of variables that will be added in it
    sql_table_variables_dict = collections.defaultdict(list)
    for column in columns:
        if len(df_row[column].shape) > 1:
            if pd.notna(df_row[column].loc['data', :]).any():
                sql_table = _find_analysis_table(variable=column, analysis_source=analysis_source)
                sql_table_variables_dict[sql_table].append(column)
        else:
            if pd.notna(df_row[column]['data']):
                sql_table = _find_analysis_table(variable=column, analysis_source=analysis_source)
                sql_table_variables_dict[sql_table].append(
                    column)  # Each sql table has a list of variables they will store
    # Variables that are not found in the tables of the analysis_source will be stored in the key "None". Since these
    # don't need populating (they are data of the analaysis metadata) they are eliminated from this dictionary
    sql_table_variables_dict.pop(None, None)
    return sql_table_variables_dict


def _set_level_values(midx, level, values):
    full_levels = list(zip(*midx.values))
    names = midx.names
    if isinstance(level, str):
        if level not in names:
            raise ValueError(f'No level {level} in MultiIndex')
        level = names.index(level)
    if len(full_levels[level]) != len(values):
        raise ValueError('Values must be of the same size as original level')
    full_levels[level] = values
    return pd.MultiIndex.from_arrays(full_levels, names=names)
