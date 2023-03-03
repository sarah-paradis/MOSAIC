# Excel template Tutorial
MOSAIC is a collaborative database! If you have data (published and unpublished) and would like to contribute, download the Excel template and send it back to mosaic@erdw.ethz.ch. Here we explain how you can fill in the template to contribute your data to MOSAIC.  

## Workbook structure
The workbook has 4 sheets: ARTICLE_AUTHOR, GEOPOINTS_CORE, SAMPLE_ANALYSES, SET_VARIABLES. 
![Excel_template_sheets](https://user-images.githubusercontent.com/15121054/222724572-83d3a9cf-87f0-4ed1-9b4e-eb051dec7ea1.jpg)

The first three sheets refer to the main data categories that the database stores, in a hierarchical configuration: the source metadata is stored in the ARTICLE_AUTHOR sheet, all the information regarding sampling (e.g., coordinates, sampling date, sampling technique) is stored in GEOPOINTS, and the different analyses performed on the sediment samples are stored in SAMPLE_ANALYSES
![1  Hierarchy-03](https://user-images.githubusercontent.com/15121054/222730501-93877178-5f77-4daa-a950-7e6e17eaf822.jpg)

In each sheet, the columns that have a darker color are **mandatory**. If you don't fill them, the quality check algorithm will raise an error.

## ARTICLE_AUTHOR
Provide information about the source of the data. If you have a DOI, provide it, since it will allow the quality check and population algorithms to extract all the metadata from that source. The information of the title and author will be asked for in the SAMPLE_ANALYSES sheet. 

## GEOPOINTS_CORES
Provide information about the sediment sampling. Columns in dark green are required, and the remaining columns are complimentary. 
- **Core_name:** unique name of the retrieved sediment sample.  
**Note that the core_name needs to be unique for indexing purposes. If you have two cores with the same name, modify their names so that they are unique (e.g., add the sampling year or month, name of cruise, etc.)**
- **Latitude and Longitude:** should be provided in WGS 84 degrees decimal.
- **Georeferenced_coordinates:** Defines whether the location of the sample was obtained by georeferencing a map, or whether they were provided in tabular format from the source.

## SAMPLE_ANALYSES
Provide information about the analyses performed on the samples. Columns in dark red are mandatory, and the remaining columns are complementary.
- **Geopoint Data:** entry_user_name, core_name, sample_name.  
Choose from the drop-down menu the *core_name* that you previously specified in GEOPOINTS_CORES. This is why the core_name needs to be unique!
The *sample_name* doesn't need to be unique. 
- **Article / Author that provided data:** exclusivity_clause, title, author_firstname, author_lastname.
Choose from the drop-down menu the *title* and *author* name that you previously specified in ARTICLE_AUTHOR. This will link the information you provide in this sheet to the appropriate reference.  
![geopoint_article_author](https://user-images.githubusercontent.com/15121054/222728834-31a43d9c-f7c6-419b-a262-357f770816fd.jpg)
- **Section of sample in the core:** sample_depth_upper_cm, sample_depth_bottom_cm, sample_depth_average_cm
Provide the upper and bottom sample section slice of the core (sample_depth_upper_cm, sample_depth_bottom_cm) **and/or** the average section depth (sample_depth_average_cm). Provide as much information as possible.  
Some studies only report that they sample *surface sediment* without specifying which sampling sectioning they performed (0-0.5 cm, 0-1 cm, 0-2 cm). In these situations, put 0 in *sample_depth_average_cm*, and you can even specify this under *sample_comments*. If you have any additional information to say about the sample section, add it under *sample_comment*.  
![sample_sectioning](https://user-images.githubusercontent.com/15121054/222728924-ce8d68b0-3916-4fed-8d3f-ddb9a269c810.jpg)

## Adding sample analyses
Given the complexity of the database, the data columns are dynamic. You can modify the column names based on the analyses that you have available. For example:  
![image](https://user-images.githubusercontent.com/15121054/222732199-1f8d594c-a8e3-4a6b-9283-5a54d7150bab.png)

The first row let's you choose the category of the sample analysis: sample_composition (OC, TN, OC:TN ratio, TIC, CaCO<sub>3</sub>), sample_sediment_properties (dry bulk density, grain size, water content, porosity, surface area), sample_isotopes ($\delta$ 13C).  
![image](https://user-images.githubusercontent.com/15121054/222732318-ca3119bb-5029-4000-ba41-2f324f70d763.png)
