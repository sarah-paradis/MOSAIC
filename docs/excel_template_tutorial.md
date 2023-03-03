# Excel template Tutorial
MOSAIC is a collaborative database! If you have data (published and unpublished) and would like to contribute, download the Excel template and send it back to mosaic@erdw.ethz.ch. Here we explain how you can fill in the template to contribute your data to MOSAIC.  

## Workbook structure
The workbook has 4 sheets: ARTICLE_AUTHOR, GEOPOINTS_CORE, SAMPLE_ANALYSES, SET_VARIABLES. 
![Excel_template_sheets](https://user-images.githubusercontent.com/15121054/222724572-83d3a9cf-87f0-4ed1-9b4e-eb051dec7ea1.jpg)

The first three sheets refer to the main data categories that the database stores, in a hierarchical configuration: the source metadata is stored in the ARTICLE_AUTHOR sheet, all the information regarding sampling (e.g., coordinates, sampling date, sampling technique) is stored in GEOPOINTS, and the different analyses performed on the sediment samples are stored in SAMPLE_ANALYSES
![1  Hierarchy-03](https://user-images.githubusercontent.com/15121054/222724638-82771df2-ef8c-46b7-8c77-040f2ffb9718.jpg)

In each sheet, the columns that have a darker color are **mandatory**. If you don't fill them, the quality check algorithm will raise an error.

## ARTICLE_AUTHOR
Fill in all the columns.

## GEOPOINTS_CORES
Provide information about the sediment sampling. Columns in dark green are required, and the remaining columns are complimentary. 
- Core_name **Note that the core_name needs to be unique for indexing purposes. If you have two cores with the same name, modify their names so that they are unique (e.g., add the sampling year or month, name of cruise, etc.)**
- Latitude and Longitude: should be provided in WGS 84 degrees decimal.
- Georeferenced_coordinates: Defines whether the location of the sample was obtained by georeferencing a map, or whether they were provided in tabular format from the source.

## SAMPLE_ANALYSES
Provide information about the analyses performed on the samples. Columns in dark red are mandatory, and the remaining columns are complimentary.
- Geopoint Data
