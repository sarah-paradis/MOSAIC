# Modern Ocean Sediment Archive and Inventory of Carbon (MOSAIC)

The Modern Ocean Sediment Archive and Inventory of Carbon (MOSAIC) is a database that compiles, but most importantly, harmonizes data from marine sediments worldwide. With this database, we aim to understand the factors that govern the distribution of organic carbon in marine sediments, such as the quantity, origin, and age of organic carbon.

## Contribute to MOSAIC
MOSAIC is a collaborative database! If you have data (published and unpublished) and would like to contribute, download the [Excel template](https://github.com/sarah-paradis/MOSAIC/raw/main/Excel_templates/MOSAIC_input_excel_file_2022.xlsx) and send it back to mosaic@erdw.ethz.ch.  
For more information on how to input data into this template, check out this [tutorial](excel_template_tutorial.md) for more information.

## MOSAIC v.2.0
We have been working hard to further expand MOSAIC, and we're ready to launch a new version of the database: **MOSAIC v.2.0!**

### What's new in MOSAIC v.2.0?
#### 1. Database structure
We have slightly updated the database's stucture, mainly modified the storage of sample metadata, so you can easily see where the data came from (the reference), how it was measured (method), if we calculated the data based on existing parameters, etc.
![mosaic_structure-01-01](https://user-images.githubusercontent.com/15121054/234294141-d1c2aef2-6e0b-42a4-9f7c-c33260d27fb1.jpg)
<em>Simple schematic representation of the database's structure</em>

#### 2. Database quality check and population workflow
One of the mistakes we identified from the previous quality check and population workflow was that it could lead to duplicate entries in the database and data published in different studies on the same samples were not properly lined together, hindering assessing relationships and patterns in the dataset. To overcome this, we modified the population workflow as shown in the different panels below:   
![02_Population_workflow_all-01](https://user-images.githubusercontent.com/15121054/234294198-14fdfbaa-483b-4b58-90bf-d8742c839e8b.jpg)
<em>Schematic diagrams of data population into MOSAIC v.2.0. Population workflow for **a)** article information, **b)** Geopoint core locations, and **c)** sample analyses. **d)** Population workflow of tables to avoid duplicates and link different analyses to the same sample. Colors in the tables indicate similarities in the values between each row (Data in MOSAIC vs. Data to be added). **e)** Database query for nearby cores. **f)** Sample analysis table grouping to populate each individual table. **g)** Example of harmonization workflow of radiocarbon analyses and sample composition.</em>

We foster open source, so come check out the code used to [quality check the data](https://github.com/sarah-paradis/MOSAIC/blob/main/src/main_quality_check.py) and its [functions](https://github.com/sarah-paradis/MOSAIC/blob/main/src/quality_check.py), as well as the code used to [populate it in the database](https://github.com/sarah-paradis/MOSAIC/blob/main/src/main_database_populate.py) and its [functions](https://github.com/sarah-paradis/MOSAIC/blob/main/src/database_populate.py).

#### 3. More variables
The number of variables stored in MOSAIC v.2.0 increased by ten-fold in comparison to the previous iteration, which only contained compositional (OC, total and organic nitrogen, CaCO<sub>3</sub>, biogenic silica), and isotopic (<sup>13</sup>C and <sup>14</sup>C) data. MOSAIC v.2.0 also includes sedimentological variables and a broad range of biomarkers!
![image](https://user-images.githubusercontent.com/15121054/225639241-b7303a14-d353-4256-b73f-0f2dd370b5c8.png)
<em>Number of variables stored in the first iteration of MOSAIC **(a)** and in MOSAIC v.2.0 **(b)**, grouped by categories: compositional (e.g., OC, TN, CaCO3, biogenic opal), isotopic (e.g., δ13C, δ15N, Δ14C), sedimentological (e.g., grain size fractions, mineral surface area, dry bulk density), and biomarkers (e.g., lignin phenols, alkanes, fatty acids, alcohols).</em> 

#### 4. Greater spatiotemporal coverage
We have quadrupled the spatiotemporal coverage of data in MOSAIC v.2.0, with sediment samples spanning from the 1950s to present. More digitalization of data collected and published prior the 1990s is necessary. Help us do this!
![07_Global_map](https://user-images.githubusercontent.com/15121054/234294282-c65b40a0-827d-40ff-af7a-4c4bbfe7771b.jpg)
<em>**a)** Spatial distribution of sampling locations stored in the first iteration of MOSAIC (red) and additional data in MOSAIC v.2.0 (blue). Note the increase in spatial coverage for MOSAIC v.2.0. **b)** Temporal distribution of the datapoints in MOSAIC v.2.0. The previous iteration of MOSAIC did not store information of the sampling year.</em> 

### How to cite MOSAIC v.2.0.
MOSAIC v.2.0 is currently being peer-reviewed. We will update this section once it is officially published, with its corresponding DOI and other citation metrics.
\n
MOSAIC is licensed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).![image](https://i.creativecommons.org/l/by/4.0/88x31.png) 

# What's next?
We just received an Open Research Data-Contribute Project: *Application Programming Interface for the Modern Ocean Sedimentary Inventory and Archive of Carbon database* (**API-MOSAIC**). With this grant, we will further expand and update the accessibility to the MOSAIC database. 
**New website and API are coming to you this 2023!**
