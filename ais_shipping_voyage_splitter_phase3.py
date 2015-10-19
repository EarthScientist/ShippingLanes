#!/usr/bin/python
# -*- coding: utf-8 -*-

# Marine Shipping Project -- Phase III
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# ABSI-LCC SOW Data Processing Script -- ExactEarth Data Dump
# created by: Michael Lindgren (malindgren@alaska.edu)
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def rolling_window( a, window ):
	''' 
	simple rolling window over a numpy array.  
	PANDAS provides similar, but would involve data restructuring
	'''
	shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
	strides = a.strides + (a.strides[-1],)
	return np.lib.stride_tricks.as_strided( a, shape=shape, strides=strides )
def calculate_distance( lat1, lon1, lat2, lon2, **kwargs ):
	'''
	Calculates the distance between two points given their (lat, lon) co-ordinates.
	It uses the Spherical Law Of Cosines (http://en.wikipedia.org/wiki/Spherical_law_of_cosines):

	cos(c) = cos(a) * cos(b) + sin(a) * sin(b) * cos(C)                        (1)

	In this case:
	a = lat1 in radians, b = lat2 in radians, C = (lon2 - lon1) in radians
	and because the latitude range is  [-π/2, π/2] instead of [0, π]
	and the longitude range is [-π, π] instead of [0, 2π]
	(1) transforms into:

	x = cos(c) = sin(a) * sin(b) + cos(a) * cos(b) * cos(C)

	Finally the distance is arccos(x)
	borrowed from: https://gmigdos.wordpress.com/2010/03/31/
				python-calculate-the-distance-between-2-points-given-their-coordinates/
	'''
	import math
	if ((lat1 == lat2) and (lon1 == lon2)):
		return 0
	try:
		delta = lon2 - lon1
		a = math.radians(lat1)
		b = math.radians(lat2)
		C = math.radians(delta)
		x = math.sin(a) * math.sin(b) + math.cos(a) * math.cos(b) * math.cos(C)
		distance = math.acos(x) # in radians
		distance  = math.degrees(distance) # in degrees
		distance  = distance * 60.0 # 60 nautical miles / lat degree
		# distance = distance * 1852 # conversion to meters
		distance  = round(distance)
		return distance;
	except:
		return 0
def calculate_initial_compass_bearing( pointA, pointB ):
	"""
	Calculates the bearing between two points.

	The formulae used is the following:
		θ = atan2(sin(Δlong).cos(lat2),
				  cos(lat1).sin(lat2) − sin(lat1).cos(lat2).cos(Δlong))

	:Parameters:
	  - `pointA: The tuple representing the latitude/longitude for the
		first point. Latitude and longitude must be in decimal degrees
	  - `pointB: The tuple representing the latitude/longitude for the
		second point. Latitude and longitude must be in decimal degrees

	:Returns:
	  The bearing in degrees

	:Returns Type:
	  float

	 borrowed from: https://gist.github.com/jeromer/2005586
	"""
	if (type(pointA) != tuple) or (type(pointB) != tuple):
		raise TypeError("Only tuples are supported as arguments")

	lat1 = math.radians(pointA[0])
	lat2 = math.radians(pointB[0])

	diffLong = math.radians(pointB[1] - pointA[1])

	x = math.sin(diffLong) * math.cos(lat2)
	y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)
			* math.cos(lat2) * math.cos(diffLong))

	initial_bearing = math.atan2(x, y)
	# Now we have the initial bearing but math.atan2 return values
	# from -180° to + 180° which is not what we want for a compass bearing
	# The solution is to normalize the initial bearing as shown below
	initial_bearing = math.degrees(initial_bearing)
	compass_bearing = (initial_bearing + 360) % 360
	return compass_bearing
def ais_time_to_datetime( x ):
	''' convert the string time in the MMSI tables to a python datetime object'''
	import re, collections, datetime
	date, time = x.split( '_' )
	year = int( date[:4] )
	month, day = [ int(i) for i in re.findall( '..?', date[4:] ) ]
	hour, minute, second = [ int(i) for i in re.findall( '..?', time ) ]
	return datetime.datetime( year, month, day, hour, minute, second )
def group_voyages( mmsi_group, speed_limit=0.9 ):
	''' group the data into unique voyages following stationarity and time differences between pings '''
	cur_df = mmsi_group
	
	# # sort by Time since a ship cant go 2 places at the same time.
	cur_df[ 'datetime_tmp' ] = [ ais_time_to_datetime( i ) for i in cur_df.Time ]
	cur_df = cur_df.sort_values( by='datetime_tmp' )
	
	# time
	time_diff = cur_df[ 'datetime_tmp' ].diff()
	cur_df = cur_df.drop( ['datetime_tmp'], axis=1 ) # drop the temporary column
	one_day = pd.Timedelta( 1,'D' )
	cur_df.loc[ :, 'day_breaks' ] = ( time_diff > one_day )
	
	# create a grouping variable that will be used to break up the groups:
	# http://stackoverflow.com/questions/19911206/how-to-find-times-when-a-variable-is-below-a-certain-value-in-pandas
	non_stationary = (cur_df.SOG > speed_limit)
	clusters = ( non_stationary == True ) & ( cur_df['day_breaks'] == False )
	cur_df.loc[ :, 'clusters' ] = str( cur_df[ 'MMSI' ].tolist()[0] ) + '_' + ( clusters != clusters.shift() ).cumsum().astype( str )
	cluster_groups = cur_df.groupby( 'clusters' ).filter( lambda x: ((x['SOG'] > speed_limit) & (x['day_breaks'] == False)).all() == True )
	cluster_groups.loc[ :, 'day_breaks' ] = cluster_groups.loc[ :, 'day_breaks' ].astype( np.int16 )# convert bool to integer
	return cluster_groups
def insert_direction_distance( x ):
	''' add in the Direction and Distance columns '''
	from geopy.distance import vincenty
	import numpy as np
	import pandas as pd

	''' calculate the direction attribute for each voyage and put it in a Direction column '''
	lonlats = zip( rolling_window( np.array( x.Longitude), 2 ), rolling_window( np.array( x.Latitude ), 2 ) )
	bearings = [ calculate_initial_compass_bearing( (lats[0], lons[0]), (lats[1], lons[1]) ) for lons, lats in lonlats ]
	bearings.insert( 0, bearings[0] ) # add in a duplicate value at the beginning the series since it is a rolling window output
	# bearings.append( bearings[ len(bearings)-1 ] ) # add in duplicate value at the end of the series 
	x.loc[ :, 'Direction' ] = bearings

	# simple directionality
	simple_directions_dict = {'NE':(-1.0,90.0),'SE':(90.0,180.0),'SW':(180.0,270.0),'NW':(270.0,361.0)}
	x.loc[ :, 'simple_direction' ] = '' # empty col to fill
	# iterate through the key value pairs and fill it in with the new simple value
	for k,v in simple_directions_dict.iteritems():
		l,h = v
		x.loc[ (x['Direction'] >= l) & (x[ 'Direction' ] < h), 'simple_direction' ] = k
	# distance
	dist_list = [ vincenty( (lats[0], lons[0]), (lats[1], lons[1]) ).nautical for lons, lats in lonlats ]
	dist_list.insert( 0, 0 ) # add back in that zero lost at the beginning
	x.loc[ :, 'Distance' ] = dist_list
	return x
def is_outlier( points, thresh=3.5 ):
	"""
	Returns a boolean array with True if points are outliers and False 
	otherwise.

	Parameters:
	-----------
		points : An numobservations by numdimensions array of observations
		thresh : The modified z-score to use as a threshold. Observations with
			a modified z-score (based on the median absolute deviation) greater
			than this value will be classified as outliers.

	Returns:
	--------
		mask : A numobservations-length boolean array.

	References:
	----------
		Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
		Handle Outliers", The ASQC Basic References in Quality Control:
		Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 

	borrowed from:
	--------------
	http://stackoverflow.com/questions/22354094/
			pythonic-way-of-detecting-outliers-in-one-dimensional-observation-data
	"""
	if len(points.shape) == 1:
		points = points[:,None]
	median = np.median(points, axis=0)
	diff = np.sum((points - median)**2, axis=-1)
	diff = np.sqrt(diff)
	med_abs_deviation = np.median(diff)
	modified_z_score = 0.6745 * diff / med_abs_deviation
	return modified_z_score > thresh
def line_it( x ):
	'''
	function to be used in a groupby/apply to help generate the needed output line
	GeoDataFrame.
	'''
	# detect and remove outliers based on latitudes:
	lat_col = 'akalb_lat'
	x = x.loc[ ~is_outlier( x[ lat_col ], thresh=3.5 ), : ]

	# get data for first and last rows
	begin_row = x.head( 1 )
	end_row = x.tail( 1 )
	
	# setup some begin-end values requested by funders
	bearing_begin, bearing_end = ( begin_row[ 'Direction' ].tolist()[0], end_row[ 'Direction' ].tolist()[0] )
	direction_begin, direction_end = ( begin_row[ 'simple_direction' ].tolist()[0], end_row[ 'simple_direction' ].tolist()[0] )
	time_begin, time_end = ( begin_row[ 'Time' ].tolist()[0], end_row[ 'Time' ].tolist()[0] )
	lon_begin, lon_end = ( begin_row[ 'Longitude' ].tolist()[0], end_row[ 'Longitude' ].tolist()[0] )
	lat_begin, lat_end = ( begin_row[ 'Latitude' ].tolist()[0], end_row[ 'Latitude' ].tolist()[0] )

	out_row = begin_row.drop( ['Longitude', 'Latitude', 'Time', 'Direction', 'simple_direction'], axis=1 )
	out_row.index = [0]
	new_cols_df = pd.DataFrame({ 'lon_begin':lon_begin, 'lon_end':lon_end, 'lat_begin':lat_begin, 'lat_end':lat_end, \
								'time_begin':time_begin, 'time_end':time_end, 'bear_begin':bearing_begin, 'bear_end':bearing_end, \
								'dir_begin':direction_begin, 'dir_end':direction_end }, index = out_row.index)

	out_row = out_row.join( new_cols_df )
	out_row[ 'geometry' ] = [ LineString( zip(x.akalb_lon.tolist(),x.akalb_lat.tolist()) ) ]
	return out_row

if __name__ == '__main__':
	import pandas as pd
	import numpy as np
	from geopy.distance import vincenty
	import datetime, math, os, glob, argparse
	import geopandas as gpd
	from pyproj import Proj
	from shapely.geometry import Point, LineString
	from collections import OrderedDict
	from pathos import multiprocessing as mp
	
	parser = argparse.ArgumentParser( description='program to add Voyage and Direction fields to the AIS Data' )
	parser.add_argument( "-p", "--output_path", action='store', dest='output_path', type=str, help='path to output directory' )
	parser.add_argument( "-fn", "--fn", action='store', dest='fn', type=str, help='path to input filename to run' )

	# parse all the arguments
	args = parser.parse_args()
	fn = args.fn
	output_path = args.output_path

	# # FOR DEVELOPMENT REMOVE LATER!
	# l = glob.glob( '/workspace/Shared/Tech_Projects/Marine_shipping/project_data/Output_Data/Thu_Sep_4_2014_121625/csv/grouped/*.csv' )
	# fn = l[8]
	# output_path = '/workspace/Shared/Tech_Projects/Marine_shipping/project_data/Phase_III/Output_Data_fixlines'

	ncpus = 31
	print 'working on: %s' % os.path.basename( fn )

	# make some output filenaming base for outputs
	output_fn_base = os.path.basename( fn ).split( '.' )[0] + '_voyage_direction_phase3'

	# read in the csv to a PANDAS DataFrame
	df = pd.read_csv( fn, sep=',' )

	# Fix columns with mixed types: [15,16,19,20] or ['ROT','SOG','COG','Heading']
	# fix ROT
	if df.ROT.dtype == np.object:
		if (df.ROT == 'None').any() == True:
			df.loc[ df.ROT == 'None', 'ROT' ] = '9999'
			df.loc[ :, 'ROT' ] = df.ROT.astype( np.float32 )
			df.loc[ df.ROT == 9999, 'ROT' ] = np.nan

	# fix SOG
	if df.SOG.dtype == np.object:
		if (df.SOG == 'None').any() == True:
			df.loc[ df.SOG == 'None', 'SOG' ] = '9999'
			df.loc[ :, 'SOG' ] = df.SOG.astype( np.float32 )
			df.loc[ df.SOG == 9999, 'SOG' ] = np.nan

	# fix COG
	if df.COG.dtype == np.object:
		if (df.COG == 'None').any() == True:
			df.loc[ df.COG == 'None', 'COG' ] = '9999'
			df.loc[ :, 'COG' ] = df.COG.astype( np.float32 )
			df.loc[ df.COG == 9999, 'COG' ] = np.nan

	# fix Heading
	if df.Heading.dtype == np.object:
		if (df.Heading == 'None').any() == True:
			df.loc[ df.Heading == 'None', 'Heading' ] = '9999'
			df.loc[ :, 'Heading' ] = df.Heading.astype( np.float32 )
			df.loc[ df.Heading == 9999, 'Heading' ] = np.nan

	# drop SOG with np.nan (NULL) -- speed over ground
	df = df.loc[ -df.SOG.isnull(), : ]

	# run this new version of the function: -- 5.5 mins
	# returns a new column called clusters which have the groupings...
	MMSI_grouped = df.groupby( 'MMSI' ).apply( group_voyages )

	try:
		# lets dig into the data a bit: we are going to keep only transects with > 100 pingbacks since that seems like a fairly short trip @ ~30 sec intervals
		# this could transform into something that looks at the intervals between each timestep and decides whether to drop it. instead of ping counts
		# since we are dropping Voyages with <100 anyhow drop those files now
		if df.shape[0] > 100: 
			unique_counts_df = pd.DataFrame( np.array( np.unique( MMSI_grouped.clusters, return_counts=True ) ).T, \
												columns=[ 'unique', 'count' ] )

			keep_list = unique_counts_df.loc[ unique_counts_df['count'] > 100, 'unique' ]
			MMSI_grouped_keep = MMSI_grouped[ MMSI_grouped.clusters.isin( keep_list ) ].copy()

			# # add in the Voyage column -- the unique id of MMSI and unique transect number
			MMSI_grouped_keep.loc[ :, 'Voyage' ] = MMSI_grouped_keep.loc[ :, 'clusters' ]

			# add in Direction, Distance, and simple_direction fields using the above function
			voyages_complete = MMSI_grouped_keep.groupby( 'Voyage' ).apply( insert_direction_distance )

			# make an output directory to store the csvs and shapefiles if needed
			if not os.path.exists( os.path.join( output_path, 'csvs' ) ):
				os.makedirs( os.path.join( output_path, 'csvs' ) )
			
			if not os.path.exists( os.path.join( output_path, 'shapefiles' ) ):
				os.makedirs( os.path.join( output_path, 'shapefiles' ) )

			# write it out to a csv
			output_filename = os.path.join( output_path, 'csvs', output_fn_base+'.csv' )
			voyages_complete.to_csv( output_filename, sep=',' )

			# a hardwired set of column names and dtypes for output shapefile
			COLNAMES_DTYPES_DICT = OrderedDict([('MMSI', np.int32),
												('Message_ID', np.int32),
												('Repeat_indicator', np.int32),
												('Time', np.object),
												('Millisecond', np.int32),
												('Region', np.float32),
												('Country', np.int32),
												('Base_station', np.int32),
												('Vessel_Name', np.float32),
												('Call_sign', np.float32),
												('IMO_ee', np.float32),
												('Ship_Type', np.float32),
												('Destination', np.float32),
												('ROT', np.float32),
												('SOG', np.float32),
												('Longitude', np.float32),
												('Latitude', np.float32),
												('COG', np.float32),
												('Heading', np.float32),
												('IMO_ihs', np.int32),
												('ShipName', np.object),
												('PortofRegistryCode', np.int32),
												('ShiptypeLevel2', np.object),
												('Voyage', np.object),
												('Direction', np.float32),
												('simple_direction', np.object)] )

			# this has an issue with converting DTYPES since they are of mixed types...  lets try to fix it above...
			voyages_complete = pd.DataFrame( { col:voyages_complete[ col ].astype( dtype ) for col, dtype in COLNAMES_DTYPES_DICT.iteritems() } )

			# make Point()s and add to a field named 'geometry' and Make GeoDataFrame
			# voyages_complete[ 'geometry' ] = voyages_complete.apply( lambda x: Point( x.Longitude, x.Latitude ), axis=1 )

			# parallelize point generation -- shapely
			lonlat = zip( voyages_complete.Longitude.tolist(), voyages_complete.Latitude.tolist() )

			# make a pool and use it
			pool = mp.Pool( ncpus )
			hold = pool.map( Point, lonlat )
			pool.close()

			voyages_complete[ 'geometry' ] = hold

			# GeoDataFrame it -- GeoPANDAS
			gdf = gpd.GeoDataFrame( voyages_complete, crs={'init':'epsg:4326'}, geometry='geometry' )
			
			# reproject this data into 3338 -- AKALBERS
			gdf_3338 = gdf.to_crs( epsg=3338 )

			# serial -- akalbers lon/lat column creation
			# ak_lonlat = gdf_3338.geometry.apply( lambda x: {'akalb_lon':x.x, 'akalb_lat':x.y} )
			# ak_lonlat = pd.DataFrame( ak_lonlat.tolist() )

			# parallelize the akalber lon/lat column creation
			pool = mp.Pool( ncpus )
			ak_lonlat = pd.DataFrame( pool.map( lambda x: { 'akalb_lon':x.x, 'akalb_lat':x.y }, gdf_3338.geometry.tolist() ) )
			pool.close()

			gdf_3338 = gdf_3338.reset_index( drop=True ).join( ak_lonlat )

			# group the data into Voyages
			voyages_grouped = gdf_3338.groupby( 'Voyage' )

			gdf_mod = voyages_grouped.apply( line_it ) # remove errant points function here
			gdf_mod = gdf_mod.reset_index( drop=True )
			gdf_mod = gpd.GeoDataFrame( gdf_mod, crs={'init':'epsg:3338'}, geometry='geometry' )

			# make geo and output as a shapefile
			output_filename = output_filename.replace( '.csv', '.shp' ).replace( 'csvs', 'shapefiles' )
			gdf_mod.to_file( output_filename )
		else:
			print 'Unable to Generate Lines for : %s ' % os.path.basename( fn )
	except Exception as e:
		pass

# # # # # # # # # # # # # # # # # # 
# # # how to run this application:
# if __name__ == '__main__':
# 	import glob, os

# 	# set the path to where the script file is stored: [hardwired]
# 	os.chdir( '/workspace/Shared/Tech_Projects/Marine_shipping/project_data/CODE/ShippingLanes' )

# 	# list the files to run and set output path
# 	l = glob.glob( '/workspace/Shared/Tech_Projects/Marine_shipping/project_data/Output_Data/Thu_Sep_4_2014_121625/csv/grouped/*.csv' )
# 	output_path = '/workspace/Shared/Tech_Projects/Marine_shipping/project_data/Phase_III/Output_Data_fixlines'
# 	command_start = 'ais_shipping_voyage_splitter_phase3.py -p ' + output_path + ' -fn '

# 	for i in l:
# 		try:
# 			os.system( 'ipython -c "%run ' + command_start + i + '"')
# 		except:
# 			print 'ERROR RUN %s: ' % os.path.basename( i )
# 			pass


# # # # # # # # REMOVE FOLLOWING DEVELOPMENT: # # # # # # #
# # GOOD FILES:
# Bulk_Carriers_grouped
# Dry_Cargo_Passenger_grouped
# Fishing_grouped
# Miscellaneous_grouped
# Non_Merchant_Ships_grouped -- QUESTIONABLE...

# # BAD FILES:
# Non_Merchant_Ships_grouped: TypeError: invalid type comparison -- SOG [LOOKS OK!]
# Non_Ship_Structures_grouped: TypeError: invalid type comparison -- SOG [LOOKS OK!]
# Non_Seagoing_Merchant_Ships_grouped: IndexError: index 0 is out of bounds for axis 0 with size -- MMSI_grouped_keep.loc[ MMSI_grouped_keep.index, 'Voyage' ] = MMSI_grouped_keep.loc[ :, 'clusters' ] [LOOKS OK!]

