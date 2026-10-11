-----------------------------------------------------------------------------------------------------------------
GENERAL INFORMATION
-----------------------------------------------------------------------------------------------------------------

G01. Names of file(s) or dataset(s) that this README file describes
	
ETo_212223.csv
irrigation_212223.csv
precipitation_212223.csv
soilsampmeas_212223.csv
sensordata2021_2022_2023.xlsx
Summary_sensors.xlsx

G02. Date of creation/last update of the README file
	
13 June 2025

G03. Name and contact information of Principal Investigator
	
Marit Hendrickx - marit.hendrickx@kuleuven.be

G04. ORCID of Principal Investigator
	
https://orcid.org/0000-0003-0410-9903

G05. Institution of Principal Investigator
	
KU Leuven

G06. Contact of other person at KU Leuven that has access to the dataset
	
Jan Diels - jan.diels@kuleuven.be
https://orcid.org/0000-0002-0317-8280

G07. Description of the dataset
	
This dataset includes soil moisture measurement data from TEROS 10 sensors at 15 cm depth and soil moisture samples from the top 30 cm soil layer. The measurement data covers 18 vegetable cropping cycles across 2021-2023. Corresponding field data are provided including bulk density and water retention measurements, crop type, planting date, irrigation method, soil type, as well as local precipitation data, ETo data and irrigation data. The purpose of this dataset is to be able to simulate soil moisture and crop water use, and to compare and/or assimilate with in situ soil moisture measurements.


G08. Keywords (author defined)

soil moisture sensor data
soil moisture sample data
bulk density
soil water retention
cropping cycles
local weather data
irrigation data

G09. Thesaurus or controlled vocabulary keywords
-

G10. Thesaurus or controlled vocabulary used in this README
-

G11. Language(s) used in the dataset

English
Dutch (2 rows in Summary_sensors.xlsx: irr_method, crop)

G12. Other involved researchers/parties

BDB (Bodemkundige Dienst van België, Heverlee)
PSKW (Proefstation voor de Groenteteelt vzw, Sint-Katelijne-Waver)
Viaverda (Viaverda vzw, Kruisem)
Praktijkpunt Landbouw Vlaams-Brabant vzw (Herent)


-----------------------------------------------------------------------------------------------------------------
PROJECT INFORMATION
-----------------------------------------------------------------------------------------------------------------

P01. Project information
	
DRIP: Datagedreven Regeling van druppelirrigatie voor een duurzame Productie in de tuinbouw

P02. Project abstract
	
In het DRIP project wordt in samenwerking met groentetelers onderzocht hoe druppelirrigatie de plaats kan in nemen van bovenberegening. 
Slimme sensoren, gekoppeld aan een bodemwaterbalansmodel, zorgen er voor dat de installatie optimaal wordt aangestuurd.

Water is een cruciale productiefactor voor circa 4000 tuinbouwbedrijven die groenten telen in openlucht. 
Uitzonderlijk droge jaren, zoals 2018, die nu één keer per 20 jaar voorkomen, zullen in de toekomst eens om de vier jaar voorkomen. 
Ook de start van het groeiseizoen 2020 was opnieuw droog, waardoor irrigatie noodzakelijk was om in de vroege teelten een optimale groei te garanderen.
Opnieuw blijkt dat watergebruik door de tuinbouw sector sterk onder druk komt te staan. Het is dus noodzakelijk om de waterefficiëntie te verhogen om zo tot een duurzame rendabele bedrijfsvoering te komen. 
Door vochtsensoren te koppelen aan een bodemwaterbalansmodel en weersvoorspellingen, waarbij gebruik gemaakt wordt van data-assimilatie, ontstaat een krachtig hulpmiddel voor irrigatiebegeleiding. 
Bijkomend kan de irrigatie efficiëntie worden verhoogd door haspelirrigatie om te vormen naar druppelirrigatie, althans voor teelten waarvan het gewas pas laat in het seizoen volledig de bodem bedekt zoals prei, selder, ui, venkel, bloemkool. 
Voor deze teelten kunnen de verdampingsverliezen door evaporatie vanuit de bodem ingeperkt worden door water gericht toe te dienen in de wortelzone. 
Doorrekeningen, literatuuronderzoek en praktijkervaring duiden op een waterefficiëntiewinst van 10 tot 30% bij de toepassing van druppelirrigatie ten opzichte van haspelberegening. 
Gedurende het project gaan pionierbedrijven aan de slag met druppelirrigatie om obstakels te overwinnen zodat druppelirrigatie praktisch en economisch evenwaardig alternatief wordt voor haspelberegening.

Dit zijn de vooropgestelde doelen:

	- Uitwerken draaiboek voor correcte keuze en gebruik van bodemsensoren
	- De bodemvochtsimulatie met bodemwaterbalansmodellen automatisch kalibreren door middel van data-assimilatie, op data gegenereerd door bodemsensoren
	- Real-time en perceel-specifiek datagedreven irrigatieadvies
	- De waterbesparing en rentabiliteit van druppelirrigatie kwantificeren voor de tuinbouwer
	- Oplossingen formuleren voor
		– knelpunten bij het gebruik van druppelirrigatie
		– meerjarig gebruik van druppelslangen.
		– plaatsing en verwijdering van druppelslangen onder de grond
	- Een druppelirrigatiesysteem koppelen aan het online datagedreven adviesplatform


P03. Project funder: Name of funder, type of grant, grant number
	
Flanders Innovation & Entrepreneurship: VLAIO LA-traject HBC.2018.2201
Research Foundation Flanders: FWO fellowship 1S20822N


-----------------------------------------------------------------------------------------------------------------
FILE OVERVIEW
-----------------------------------------------------------------------------------------------------------------

F01. Number of files described by the README-file
	
6


F02. List with names of files, description, date of creation of file
	
ETo_212223.csv
	ETo data based on on-site or nearby weather station. Each column corresponds with a sensor module location.
	October 2024

irrigation_212223.csv
	Field-specific irrigation data expressed in mm. Each column corresponds with a sensor module location.
	October 2024

precipitation_212223.csv
	Field-specific precipitation data expressed in mm. Precipitation was measured on-site with a bucket-type pluviometer; missing data was filled with precipitation radar data. Each column corresponds with a sensor module location.
	October 2024

soilsampmeas_212223.csv
	Gravimetric soil moisture measurements from 30 cm soil samples with a gouge auger.
	October 2024

sensordata2021_2022_2023.xlsx
	Soil moisture sensor data from TEROS 10, both raw (mV) and converted (volumetric soil moisture content, m3 m-3). A sensor module consisted of three TEROS 10 sensors.
	October 2024

Summary_sensors.xlsx
	Overview of the sensor modules in 2021-2023, with their corresponding field data.
	June 2025


F03. File formats
	
csv
Excel


F04. Software used to generate the data
	
Python was used to download the sensor data via an API.


F05. Software necessary to open the file
	
Microsoft Office (Excel)


F06. Relationship between the files
	
The number of the sensor modules links the data in each file.


-----------------------------------------------------------------------------------------------------------------
METHODOLOGICAL INFORMATION
-----------------------------------------------------------------------------------------------------------------

M01. Date (beginning-end) and place of data collection
	
2021 to 2023
The study sites were agricultural field located in Flanders, Belgium.
Specific coordinates of the locations are confidential due to privacy concerns.


M02. Aim for which the data were collected

The aim of this field-specific measurement dataset is to support research on real-time irrigation scheduling using soil moisture sensors and modeling; specifically to calibrate and validate data-informed soil moisture simulations for vegetable crops in Flanders.


M03. Data collecting method
	
Sensors: TEROS 10, dielectric capacitance sensors
Manual soil moisture samples using a 30 cm auger, oven-dried
Undisturbed soil samples for bulk density & water retention using Kopecky rings
Water retention measurements using pressure plate method


M04. Information about data processing methods

ETo
	Daily ETo was calculated using the Penman-Monteith equation, based on weather variables from on-site or nearby weather stations.

precipitation
	Precipitation was measured on-site with a bucket-type pluviometer; missing data was filled with precipitation radar data.


M05. Information about the instrument, calibration
	
TEROS 10, dielectric capacitance sensors
The manufacturer’s calibration equation for mineral soils was applied to convert the raw sensor output in mV to soil water content (m3 m-3) (TEROS 10, 2022)
θ = -2.154 + 3.898×10^(-3)×mV - 2.278×10^(-6)×mV^2 + 4.824×10^(-10)×mV^3

We recommend to apply the pooled sensor calibration from  
Hendrickx, M. G. A., Vanderborght, J., Janssens, P., Bombeke, S., Matthyssen, E., Waverijn, A., and Diels, J.: Pooled error variance and covariance estimation of sparse in situ soil moisture sensor measurements in agricultural fields in Flanders, SOIL, 11, 435–456, https://doi.org/10.5194/soil-11-435-2025, 2025.


M06. Quality assurance procedures
	
Individual sensor data that are outside the expected range (0.01 - 1 m3 m-3) should be filtered out / deleted.


M07. Information about limitations of the dataset, information that ensures correct interpretation of the dataset

The specific coordinates of the fields are not published due to privacy concerns.


M08. People involved in the creation or processing of the dataset
	
BDB (Jarl Vaerten, Pieter Janssens, Eveline Baens)
PSKW (Noémie Hisette, Joris De Nies, Sander Bombeke)
Viaverda (Anne Waverijn, Elise Vandewoestijne)
Praktijkpunt Landbouw Vlaams-Brabant (Evi Matthyssen)


-----------------------------------------------------------------------------------------------------------------
DATA ACCESS AND SHARING
-----------------------------------------------------------------------------------------------------------------

A01. Recommended citation for the dataset
	


A02. License information, restrictions on use
	
CC-BY-4.0
Free to use, share, adapt, provided that you give appropriate credit 

A03. Confidentiality information

Specific coordinates of the locations are confidential


-----------------------------------------------------------------------------------------------------------------
DATA SPECIFIC INFORMATION (ABOUT THE DATA THEMSELVES)
-----------------------------------------------------------------------------------------------------------------

D01. Full names and definitions for columns and rows

- Summary_sensors.xlsx
year
	Year; 2021-2023

sensor
	Number of sensor module

crop
	Specific vegetable crop type (Dutch)

location
	Location (city/municipality) of the study sites with the sensor module (Kruisem ; Herent ; Sint-Katelijne-Waver ; Kinrooi)

irr_method
	Irrigation method at the study site
	'druppel': drip irrigation
	'haspel': overhead, hose reel irrigation

soil_type
	Belgian soil texture class of the study site
	Z: Sand, S: Loamy sand, P: Light sandy loam, L: Heavy sandy loam, A: Silt loam
	
bd_mean
	Mean bulk density of the study site (g cm-1)
	Based on three undisturbed soil samples (Kopecky cores) at 15 cm depth (bd_1, bd_2, bd_3; (g cm-1))

pF0_1, pF0_2, pF0_3
	Volumetric soil moisture content (m3 m-3) at pF 0 from three undisturbed soil samples (Kopecky cores) at 15 cm depth

pF2_1, pF2_2, pF2_3
	Volumetric soil moisture content (m3 m-3) at pF 2 from three undisturbed soil samples (Kopecky cores) at 15 cm depth

pF2.7_1, pF2.7_2, pF2.7_3
	Volumetric soil moisture content (m3 m-3) at pF 2.7 from three undisturbed soil samples (Kopecky cores) at 15 cm depth

pF4.2_1, pF4.2_2, pF4.2_3
	Volumetric soil moisture content (m3 m-3) at pF 4.2 from three undisturbed soil samples (Kopecky cores) at 15 cm depth

planting_date
	Serial date of the planting/sowing date


- ETo_212223.csv ; irrigation_212223.csv ; precipitation_212223.csv
year	
	Year; 2021-2023

Date
	Serial date

Other columns: number of the sensor module, used to identify different study sites. Contains daily amounts in mm (ETo ; irrigation ; precipitation)


- sensordata2021_2022_2023.xlsx
Datetime
	Timestamp: dd/mm/yyyy hh:mm:ss

Sensor
	Number of the sensor module, used to identify different study sites

Adc0 (mV), Adc1 (mV), Adc2 (mV)
	Raw mV measurement by the three sensors connected to a single sensor module

vwc0 (m3/m3), vwc1 (m3/m3), vwc2 (m3/m3)
	Volumetric soil moisture content (m3 m-3) measured by the three sensors connected to a single sensor module from converting the raw mV measurement using the manufacturer’s calibration equation (TEROS 10, 2022)

pluvio
	If a pluviometer was connected to the logger: mm measured with bucket-type rain gauge

temp
	Temperature measured in the logger


- soilsampmeas_212223.csv
Date
	Serial date

0_30(grav%)
	Gravimetric soil moisture content (%g g-1) in the 0-30cm soil layer measured from disturbed soil moisture samples using a 30 cm auger, oven-dried

30_60(grav%)
	Gravimetric soil moisture content (%g g-1) in the 30-60cm soil layer measured from disturbed soil moisture samples using a 30 cm auger, oven-dried

0_5(grav%)
	Gravimetric soil moisture content (%g g-1) in the 0-5cm soil layer measured from disturbed soil moisture samples using trowel, oven-dried

sensor
	Number of the sensor module, used to identify different study sites

year
	Year; 2021-2023


D02. Explanation of abbreviations


D03. Units of measurement

Volumetric soil moisture content (m3 m-3)
%grav: Gravimetric soil moisture content (%g g-1)
See above

D04. Symbols for missing data

[empty]


-----------------------------------------------------------------------------------------------------------------
REFERENCES
-----------------------------------------------------------------------------------------------------------------
Manufacturer’s calibration equation:
	TEROS 10: http://publications.metergroup.com/Manuals/20788_TEROS10_Manual_Web.pdf, last access: 18 May 2022.
