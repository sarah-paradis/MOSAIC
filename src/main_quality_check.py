import quality_check as qc
import os

"""
List of steps
1. Open Excel or other spreadsheet

2. Match column names with variables names.

3. Check the metadata. Log missing data as warnings.
Checks that all the required fields are not empty.
Checks that the column names are in the database, and if there is a missing column not found in the database.
Checks that there are no duplicate column names (data entered twice).
If relevant, checks if a column has duplicate entries. 

4. Check that data is within limits for each variables. Log warnings.
Add geospatial information to geopoints

5. Save checked data in new folder
"""

# 1. Open Excel or other spreadsheet
filename = 'MOSAIC_template.xlsx'
file_dir = 'new_data'

df_lists = qc.read_df(file=filename,
                      article_author_metadata='ARTICLE_AUTHOR',
                      geopoints_cores='GEOPOINTS_CORES',
                      sample_analyses='SAMPLE_ANALYSES',
                      file_dir=file_dir)
df_article_author = df_lists[0]
df_geopoints = df_lists[1]
df_sample_analyses = df_lists[2]


# Check ARTICLE_METADATA
print('Checking metadata in ARTICLE_AUTHOR')
qc.check_metadata(filename=filename, df=df_article_author, required_fields=['entry_user_name',
                                                                            (['doi'],
                                                                             ['title', 'year', 'journal',
                                                                              'author_firstname', 'author_lastname'],
                                                                             ['author_firstname', 'author_lastname'])])
# ARTICLE_METADATA needs to have entry_user_name AND (doi, OR full article metadata (if no DOI), OR author info
# (if unpublished))

# Check GEOPOINTS_CORES
print('Checking metadata in GEOPOINTS_CORES')
qc.check_metadata(filename=filename, df=df_geopoints,
                  required_fields=['entry_user_name', 'core_name', 'latitude', 'longitude'],
                  duplicate_row_in_column='core_name',
                  exceptions='country_research_vessel')
# GEOPOINTS can not have duplicate core_names, since in the data population step each core_name will be assigned a
# core_id


# Check SAMPLE_ANALYSES
print('Checking metadata in SAMPLE_ANALYSIS')
qc.check_metadata(filename=filename, df=df_sample_analyses,
                  required_fields=['entry_user_name', 'core_name', 'sample_name', 'exclusivity_clause',
                                   'author_firstname', 'author_lastname',
                                   ('sample_depth_average_cm', ['sample_depth_upper_cm', 'sample_depth_bottom_cm'])],
                  exceptions='exclusivity_clause')
# Sample analyses need to provide OR the average section depth OR both the upper and lower limit

## 4. Check that data is within limits for each variables. Log warnings
correct_columns_article = qc.check_article_author_data(filename=filename,
                                                       df_article_author=df_article_author,
                                                       dfs_replace_values=[df_sample_analyses])
if correct_columns_article:
    print('Successfully checked ARTICLE_AUTHORS')

incorrect_sample_analysis = qc.check_sample_analyses_range(filename, df_sample_analyses,
                                                           core_metadata=df_geopoints['core_name'],
                                                           title_metadata=df_article_author['title'],
                                                           author_firstname_metadata=df_article_author[
                                                               'author_firstname'],
                                                           author_lastname_metadata=df_article_author[
                                                               'author_lastname'])
print(f'Incorrect columns of sample analyses in {filename} are {incorrect_sample_analysis}')

incorrect_columns_coredata = qc.check_core_range(filename, df_geopoints)
# Adds additional information to the geopoint (geomorphological site, longhurst province, exclusive economics zone,
# seas, ocean
print(f'Incorrect columns of core data in {filename} are {incorrect_columns_coredata}')

# 5. Save data in new folder and move raw data into a checked folder
qc.save_df(df_list=[df_article_author, df_geopoints, df_sample_analyses],
           df_list_name=['ARTICLE_AUTHOR', 'GEOPOINTS_CORES', 'SAMPLE_ANALYSES'],
           filename=filename, dir_output='quality_checked')
qc.move_file(filename=filename, dir_output='new_data/quality_checked', file_dir=file_dir)
