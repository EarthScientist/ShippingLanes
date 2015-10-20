ShippingLanes
=============

# AIS Shipping Data Wrangling

Data Processing and some output metrics scripts for working with the Exact Earth 
Exact AIS raw data dump for the Aleutian and Bering Sea Islands region off the Western 
Coast of Alaska.

##### PHASE III:

    Improved Voyage slicing and cleanup of voyage line transects. 
    Adds Directionality (0-360) and in a more simple way:
    {   'NE':(0.0, 90.0),
        'SE':(90.0, 180.0),
        'SW':(180.0, 270.0),
        'NW':(270.0, 360.0)     }

These **new** tools are found in [ais_shipping_voyage_splitter_phase3].py.  Code is documented and is still in heavy development so it will be changing rapidly and unexpectedly.


[ais_shipping_voyage_splitter_phase3]:https://github.com/EarthScientist/ShippingLanes/blob/master/ais_shipping_voyage_splitter_phase3.py

