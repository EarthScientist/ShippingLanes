ShippingLanes
=============

# AIS Shipping Data Wrangling

Data Processing and some output metrics scripts for working with the Exact Earth 
Exact AIS raw data dump for the Aleutian and Bering Sea Islands region off the Western 
Coast of Alaska.

##### PHASE III:

    Improved Voyage slicing and cleanup of voyage line transects. 
    Adds Directionality (0-360) and Cardinality ('N','NE','E','SE','S','SW','W','NW')

    Adds some very simple methods, which appear to have decent results, for removing outliers in Voyages.  
    This method involves setting thresholds on the median absolute deviation.

These **new** tools are found in [ais_shipping_voyage_splitter_phase3].py.  Code is documented and is still in heavy development so it will be changing rapidly and unexpectedly.


[ais_shipping_voyage_splitter_phase3]:https://github.com/EarthScientist/ShippingLanes/blob/master/ais_shipping_voyage_splitter_phase3.py


#### GLOSSARY OF THE FIELDS:
--------------------------
	Base_stati = Base station ID
	COG = [AIS] Degrees (0-359); 511 = not available = default
	Call_sign = [AIS] Ship radio call sign
	Country = [AIS] Country of origin
	Destinatio = [AIS] Port of destination
	Heading = [AIS] Ship heading
	IMO_ee = [AIS] vessel identification number
	IMO_ihs = vessel identification number from IHS-join in PHASE II
	MMSI = [AIS] Unique ship identification number 
	Message_ID = [AIS] AIS message number
	Millisecon = [AIS] Timestamp milliseconds
	PortofRegi = [AIS] Ship port of registry
	ROT = [AIS]
	Region = [AIS]
	Repeat_ind = was the message repeated
	SOG = [AIS] Knots (0-62); 63 = not available = default
	ShipName = [AIS] Ship Name
	Ship_Type = Vesel type
	ShiptypeLe = Vessel type from IHS data in Phase II
	Vessel_Nam = Vessel name from IHS data in Phase II
	Voyage = [III] Unique Voyage name 
	lon_begin = [III] longitude value of the starting point of the transect
	lon_end = [III] longitude value of the ending point of the transect
	lat_begin = [III] latitude value of the starting point of the transect
	lat_end = [III] latitude value of the ending point of the transect
	time_begin = [III] time of stating point of the transect
	time_end = [III] time of ending point of the transect
	bear_begin = [III] bearing between the initial 2 points of the transect
	bear_end = [III] bearing between the ending 2 points of the transect
	dir4_begin = [III] this is the old 'simple_direction' with a 4-quadrant return of directionality between the first 2 points of the voyage
	dir4_end = [III] this is the old 'simple_direction' with a 4-quadrant return of directionality between the last 2 points of the voyage
	dir8_begin = [III] a new 8 direction version of the simple direction between the first 2 points of the voyage.
	dir8_end = [III] a new 8 direction version of the simple direction between the last 2 points of the voyage.

#### METADATA

[METADATA](https://github.com/EarthScientist/ShippingLanes/blob/master/METADATA/metadata_shipping_voyages_phase3.xml) Describing all of the data in a global way following ISO-19119 using GeoNetwork.


