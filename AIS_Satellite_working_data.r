# examining the shipping lanes AIS Satellite Data

# a little function to load the data we need for the data exploration

load_packages <- function (x, repos="http://cran.r-project.org", ...){
   if (!require(x,character.only=TRUE, quietly=TRUE)){
      install.packages(pkgs=x, repos=repos, ...)
      require(x, character.only=TRUE, quietly=TRUE)
   }
   return(paste0('loaded: ',x))
}
pkg_list <- list('raster','maptools','sp','rgeos','rgdal','parallel','plyr','igraph','reshape2','ggplot2','Metrics')
sapply(pkg_list, load_packages)


# set up a working directory
setwd('/workspace/UA/malindgren/projects/ShippingLanes/AIS_Satellite/converted_data_to_shapefiles')

# read the file into R
csv <- read.csv('/workspace/UA/malindgren/projects/ShippingLanes/AIS_Satellite/downloaded_data_extracted/20100710_20100930.csv')

# check out the names of the csv header
names(csv)

# now we know that 29:30 are the cols with the lon/lat info
# convert the data into Spatial Points

pts <- SpatialPoints(csv[,29:30])

# Error in .checkNumericCoerce2double(obj) : non-finite coordinates

# Due to the error I think there are non-numeric values in one of the cols of lon/lat so lets check that

length(which(is.na(csv[,30]) == T))

# [1] 7683
# so there are a lot of missing values, lets remove those rows from the data.frame

csv <- csv[which(is.na(csv[,30]) == F),]

# rows are removed now lets try again
pts <- SpatialPoints(csv[,29:30])

# yay! it worked.  now lets bring the data into those spatialpoints

spdf <- SpatialPointsDataFrame(pts, data=csv)

# now lets write it out into a shapefile so we can examine it outside of R
writeSpatialShape(spdf, fn='AIS_Satellite_20100710_20100930')

# examining the data outside of R it shows that the data are in a WGS 1984 Geographic Reference System
# lets re-project it to the NAD 1983 ALASKA ALBERS Projection System

# set the ref system to wgs1984
spdf@proj4string<-CRS('+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs')

#project it
spdf_akalb <- spTransform(spdf, CRSobj=CRS('+init=epsg:3338'))

# now lets write it out again
writeSpatialShape(spdf_akalb, fn='AIS_Satellite_20100710_20100930_akalb')

# so now that we have the data in a spatial format and can see that the whole region is basically covered by points
# it may be time here to get the data into SpatialLines Formats

# now I will need to examine the data a bit further to see if we can replicate the date/Destination unique ID for this set of AIS data
# similar to what we have done with the last set of AIS data from the ground stations

# are there any names of the csv file that have destination in it?
grep('Dest',names(csv))

# yup there indeed are a few names that have destination in the name

# the data look to be characterized by NA values (not good) lets test to be sure
lapply(csv[,grep('Dest',names(csv))], unique)

# which returns the following:
# $Destination
# [1] 
# 1046 Levels:                         ... ZHENJIANG,CHINA     

# $Destination_ID
# [1] NA

# $Destination_ID_1
# [1] NA

# $Destination_ID_2
# [1] NA

# $Destination_ID_3
# [1] NA

# $Destination_ID_4
# [1] NA

# $Destination_ID_A
# [1] NA

# $Destination_ID_B
# [1] NA

# $Destination_indicator
# [1] NA


# so it looks like there are some problems here.  No destination identifier, so we will need to find another one

# since there are a LOT of NA's in the columns lets find ones that have all or no NA's so we can remove/focus on those 
test_na <- apply(csv,2,function(x) (length(which(is.na(x) == TRUE))-nrow(csv)))

# which returns the following:
 #                  MMSI             Message_ID       Repeat_indicator 
 #               -651260                -651260                -651260 
 #                  Time            Millisecond                 Region 
 #               -651260                -651260                     -1 
 #               Country           Base_station            Online_data 
 #               -651260                -651260                -651260 
 #            Group_code            Sequence_ID                Channel 
 #               -651260                      0                -651260 
 #           Data_length            Vessel_Name              Call_sign 
 #               -651260                -651260                -651260 
 #                   IMO              Ship_Type       Dimension_to_Bow 
 #                     0                      0                      0 
 #    Dimension_to_stern      Dimension_to_port Dimension_to_starboard 
 #                     0                      0                      0 
 #               Draught            Destination            AIS_version 
 #                     0                -651260                      0 
 #   Navigational_status                    ROT                    SOG 
 #               -647306                -651260                -651260 
 #              Accuracy              Longitude               Latitude 
 #               -651260                -651260                -651260 
 #                   COG                Heading               Regional 
 #               -651260                -651260                    -35 
 #              Maneuver              RAIM_flag     Communication_flag 
 #               -647306                -651260                    -35 
 #   Communication_state               UTC_year              UTC_month 
 #               -651260                  -3919                  -3919 
 #               UTC_day               UTC_hour             UTC_minute 
 #                 -3919                  -3919                  -3919 
 #            UTC_second          Fixing_device   Transmission_control 
 #               -651260                  -3919                  -3919 
 #             ETA_month                ETA_day               ETA_hour 
 #                     0                      0                      0 
 #            ETA_minute               Sequence         Destination_ID 
 #                     0                      0                      0 
 #       Retransmit_flag           Country_code          Functional_ID 
 #                     0                      0                      0 
 #                  Data       Destination_ID_1             Sequence_1 
 #                     0                      0                      0 
 #      Destination_ID_2             Sequence_2       Destination_ID_3 
 #                     0                      0                      0 
 #            Sequence_3       Destination_ID_4             Sequence_4 
 #                     0                      0                      0 
 #              Altitude        Altitude_sensor          Data_terminal 
 #                     0                      0                      0 
 #                  Mode            Safety_text      Non.standard_bits 
 #                   -35                      0                      0 
 #        Name_extension Name_extension_padding         Message_ID_1_1 
 #                     0                      0                      0 
 #            Offset_1_1         Message_ID_1_2             Offset_1_2 
 #                     0                      0                      0 
 #        Message_ID_2_1             Offset_2_1       Destination_ID_A 
 #                     0                      0                      0 
 #              Offset_A            Increment_A       Destination_ID_B 
 #                     0                      0                      0 
 #               offsetB             incrementB          data_msg_type 
 #                     0                      0                      0 
 #            station_ID                Z_count         num_data_words 
 #                     0                      0                      0 
 #                health              unit_flag                display 
 #                     0                    -35                    -35 
 #                   DSC                   band                  msg22 
 #                   -35                    -35                    -35 
 #               offset1             num_slots1               timeout1 
 #                     0                      0                      0 
 #           Increment_1               Offset_2         Number_slots_2 
 #                     0                      0                      0 
 #             Timeout_2            Increment_2               Offset_3 
 #                     0                      0                      0 
 #        Number_slots_3              Timeout_3            Increment_3 
 #                     0                      0                      0 
 #              Offset_4         Number_slots_4              Timeout_4 
 #                     0                      0                      0 
 #           Increment_4              ATON_type              ATON_name 
 #                     0                      0                      0 
 #          off_position            ATON_status           Virtual_ATON 
 #                     0                      0                      0 
 #             Channel_A              Channel_B             Tx_Rx_mode 
 #                     0                      0                      0 
 #                 Power      Message_indicator    Channel_A_bandwidth 
 #                     0                      0                      0 
 #   Channel_B_bandwidth         Transzone_size            Longitude_1 
 #                     0                      0                      0 
 #            Latitude_1            Longitude_2             Latitude_2 
 #                     0                      0                      0 
 #          Station_Type        Report_Interval             Quiet_Time 
 #                     0                      0                      0 
 #           Part_Number              Vendor_ID       Mother_ship_MMSI 
 #                     0                -651260                      0 
 # Destination_indicator            Binary_flag            GNSS_status 
 #                     0                      0                      0 
 #                 spare                 spare2                 spare3 
 #               -651260                      0                      0 
 #                spare4 
 #                     0 


# now we need to find out which columns have all the data
csv_dat <- csv[,as.numeric(which(abs(test_na) == nrow(csv)))]

# the above command returned these columns as having all of the data 

#        MMSI          Message_ID    Repeat_indicator                Time 
#           1                   2                   3                   4 
# Millisecond             Country        Base_station         Online_data 
#           5                   7                   8                   9 
#  Group_code             Channel         Data_length         Vessel_Name 
#          10                  12                  13                  14 
#   Call_sign         Destination                 ROT                 SOG 
#          15                  23                  26                  27 
#    Accuracy           Longitude            Latitude                 COG 
#          28                  29                  30                  31 
#     Heading           RAIM_flag Communication_state          UTC_second 
#          32                  35                  37                  43 
#   Vendor_ID               spare 
#         131                 136 


# how many columns have *some* data
# lets solve this by summing the number of cols that have all data with the number of cols that
# have no data and subtract by the total ncols
(length(which(abs(test_na) == nrow(csv))) + length(which(test_na == 0))) - ncol(csv)

# this command returns
# -18
# this means that there are 18 cols with *some* data in them, which ones are they?

# get the names of the columns with all data present and with no data present
csv_names_na <- names(csv[,which(test_na == 0)])
csv_names_dat <- names(csv[,which(abs(test_na) == nrow(csv))])

# use those names with the match() function that will allow for us to find the id's of the 
# columns that do not match the colnames of the 2 other groups,
# thus creating the some_data csv set

csv_some_dat <- csv[,which(is.na(match(names(csv), c(names(csv_dat),names(csv_na)))))]

# now that we have this new subset of the original csv columns, lets ask it how much data live in the columns

apply(csv_some_dat,2, function(x) length(which(is.na(x) == FALSE)))

# which returns the following:

#     Region  Navigational_status             Regional 
#          1               647306                   35 
#   Maneuver   Communication_flag             UTC_year 
#     647306                   35                 3919 
#  UTC_month              UTC_day             UTC_hour 
#       3919                 3919                 3919 
# UTC_minute        Fixing_device Transmission_control 
#       3919                 3919                 3919 
#       Mode            unit_flag              display 
#         35                   35                   35 
#        DSC                 band                msg22 
#         35                   35                   35 


# so there is *some* data in there, and we should probably keep those columns since they may prove useful
# we will need to make this sst


# QUESTIONS I HAVE REGARDING MOVING FORWARD
1. which columns are deemed necessary to perform the procedure.
	- There are more than one columns named to the longitude and Latitude
	- which ship destination to use? -- none seems to have what the last set of data header

2. do we have a unique idenifier for a ship that is not a name?  The names are going to be a nightmare to figure out
	as there are many names with special/forbidden characters and a lot of misspellings at first glance.
	- this actually brings up an interesting point, are we only using the ships from the previous analysis?  if so we 
		may have a better way forward.  If not it will be tough to reconcile all of this.

3. we need a smart way of determining the unique transects for a given time.  the only way we have done this in the past
	is by creating new names by combining the ship names and the date, where we had to assume that the shipping transect 
	only exists for a single day.  This is quite obviously very wrong and is something that had to be done due to the lack
	of transect identifiers.


# test of the ExactEarth historical csv 
# this calculates the percentage of missing data in each column 
missing_dat <- lapply(csv, function(x) length(which(is.na(x) == TRUE))/nrow(csv))


# possible way forward with the analysis
1. lets grab a single ship that is in both and try to link them?
	- this must involve something small since we certainly cannot use all of the data
2. put it in an sqlite database?  I am not even sure this is good practice.



