import database_populate as dp
import os

"""
1. Read data after quality check

2. Populate Article and author metadata. Log warning if data is already in database.

3. Populate Core information. Log warning if data is already in database.

4. Populate sample analyses. Log warning if data is already in database.
"""

file_dir = 'quality_checked'

# 1. Read data after quality check
filename = '2023_03_16_Baozhi_unpublished.xlsx'

df_lists = dp.read_df(file=filename,
                      article_author_metadata='ARTICLE_AUTHOR',
                      geopoints_cores='GEOPOINTS_CORES',
                      sample_analyses='SAMPLE_ANALYSES',
                      file_dir=file_dir)
df_article_author = df_lists[0]
df_geopoints = df_lists[1]
df_sample_analyses = df_lists[2]

# 2. Populate Article and author metadata
article_metadata_dict = dp.insert_article_from_columns(df=df_article_author, doi_column='doi', article_table='articles',
                                                       author_table='authors', authorship_table='authorship')

# 3. Populate Core information
# Returns a dictionary of core_name: core_id, used to properly assign the core_id to the analyses conducted later on.
core_id_dict = dp.insert_geopoints(df=df_geopoints,
                                   sampling_campaign_table='sampling_campaign',
                                   geopoints_table='geopoints')


# 4. Populate sample analysis
dp.insert_sample_analyses(df=df_sample_analyses,
                          core_id_dict=core_id_dict,
                          article_author_metadata=article_metadata_dict)


# 5. Move file to another directory
dp.move_file(filename=filename, dir_output='quality_checked/MOSAIC', file_dir=file_dir)
