#!/usr/bin/python
# -*- coding: utf-8 -*-

# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
# Marine Shipping Project -- Phase III
# ABSI-LCC / WCS SOW Data Processing Script -- ExactEarth Data Dump
# created by: Michael Lindgren (malindgren@alaska.edu)
#
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 

def rolling_window( a, window ):
	''' 
	simple rolling window over a numpy array.  
	PANDAS provides similar, but would involve data restructuring
	'''
	shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
	strides = a.strides + (a.strides[-1],)
	return np.lib.stride_tricks.as_strided( a, shape=shape, strides=strides )
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
def cardinal_from_bearing( bearing, simple=True ):
	'''
	take a 0-360.0 bearing measurement and convert to 
	a cardinal direction for ease-of-use

	arguments:
		bearing = float value describing the bearing to convert
		simple = True / False, default = True which means that it
			will return only 4 quadrant based directions
			[ NE, SE, SW, NW ]
			* if simple = False:
			[ 'N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW' ]
			is returned

	returns:
		1 or 2 digit string representation of the cardinality number 
		based on 
	'''
	import numpy as np

	# note that there are 2 North groups since it traverses the 0th degree
	if simple:
		direction_dict = {'NE':(-1.0,90.0),'SE':(90.0,180.0),'SW':(180.0,270.0),'NW':(270.0,361.0)}
	elif simple == False:
		direction_dict = {'E': (67.5, 112.5), 'SW': (202.5, 247.5), 'NE': (22.5, 67.5),
						'N1': (337.5, 360.0), 'N2' : (0, 22.5), 'S': (157.5, 202.5),
						'W': (247.5, 292.5), 'SE': (112.5, 157.5), 'NW': (292.5, 337.5) }
	else:
		 raise ValueError( 'argument ::simple:: must be Boolean True/False only' )

	cardinal = [ k for k,v in direction_dict.iteritems() if np.logical_and(v[0] <= bearing, v[1] > bearing) ][ 0 ]
	if cardinal in [ 'N1', 'N2' ]: # only for simple = False case
		cardinal = 'N'
	return cardinal
def insert_direction_distance( x, lon_col='Longitude', lat_col='Latitude' ):
	''' 
	add in the Direction and Distance columns.  We are stuck doing this in the original 
	WGS84 LatLong space the data are distributed in and the output directions are in 0-360
	degrees and the distance is in Nautical Miles (nm).

	ARGUMENTS:

	x = is a PANDAS DataFrame with Latitude and Longitude columns in WGS 1984 Ellipsoid (Decimal Degrees)
	lon_col = string name of the Longitude column to use in the DataFrame - default is 'Longitude'
	lat_col = string name of the Latitude column to use in the DataFrame - default is 'Latitude'

	RETURNS:

	a modified version of the input data frame (modified in-place, usually by way of a PANDAS GroupBy), 
	with new columns added for Direction-(0-360), cardinal4-(4 cardinal direction strings), 
	cardinal8-(8 cardinal direction strings), and Distance-(nautical miles) between a rolling window
	pair of points.


	'''
	from geopy.distance import vincenty
	import numpy as np
	import pandas as pd

	# calculate the direction attribute for each voyage and put it in a Direction column
	lonlats = zip( rolling_window( np.array( x[ lon_col ]), 2 ), rolling_window( np.array( x[ lat_col ] ), 2 ) )
	bearings = [ calculate_initial_compass_bearing( (lats[0], lons[0]), (lats[1], lons[1]) ) for lons, lats in lonlats ]
	bearings.insert( 0, bearings[0] ) # add in a duplicate value at the beginning the series since it is a rolling window output
	x.loc[ :, 'Direction' ] = bearings

	# return 2 new cardinality sets in 4 cardinal direction or 8 cardinal directions
	x.loc[ :, 'cardinal4' ] = [ cardinal_from_bearing( bearing, simple=True ) for bearing in bearings ]
	x.loc[ :, 'cardinal8' ] = [ cardinal_from_bearing( bearing, simple=False ) for bearing in bearings ]

	# distance
	dist_list = [ vincenty( (lats[0], lons[0]), (lats[1], lons[1]) ).nautical for lons, lats in lonlats ]
	dist_list.insert( 0, 0 ) # add back in that zero lost at the beginning
	x.loc[ :, 'Distance' ] = dist_list
	return x
def is_outlier( points, thresh=2.7 ):
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
def remove_outliers_v2( df, max_speed=40.0 ):
	'''
	remove the outliers based on distance traveled in the interval
	between points in the line, and a max speed of the vessel.  Any 
	points with distances outside the allowable estimated range, are
	dropped.

	arguments:
		df = [geopandas.GeoDataFrame] of a single voyages' data including 
				the AIS standard Time field, and a field of the distances
				between adjacent (chronological) points.
		max_speed = [float] maximum speed limit of the vessel in knots.

	returns:
		GeoDataFrame with any error points dropped.

	'''
	#setup global vars on the fly
	seconds_in_hour = 60.0 * 60.0
	nm_per_sec = (1/seconds_in_hour) * max_speed #knots
	max_hours = 15
	max_point_interval = seconds_in_hour * max_hours

	df.loc[ :, 'datetime_tmp' ] = [ ais_time_to_datetime( i ) for i in df.Time ]
	dt_diff = df.datetime_tmp.diff()
	dt_diff_seconds = pd.Series([ i.total_seconds() for i in dt_diff ])

	distance = df.Distance.reset_index( drop=True )
	distance_estimate = ( dt_diff_seconds * nm_per_sec )
	estimate_diff = -((distance_estimate - distance) < 0.0) 
	
	# inverse test 
	dt_diff_seconds_inv = dt_diff_seconds[::-1]
	distance_inv = distance[::-1]
	distance_estimate_inv = ( dt_diff_seconds_inv * nm_per_sec )
	estimate_diff_inv = -((distance_estimate_inv - distance_inv) < 0.0)

	estimate_diff.index = df.index
	df = df.loc[ estimate_diff, : ]
	return df.drop( 'datetime_tmp', 1 )

def line_it( x ):
	'''
	function to be used in a groupby/apply to help generate the needed output line
	GeoDataFrame.
	'''
	# # detect and remove outliers based on latitudes:
	# lat_col = 'akalb_lat'
	# lon_col = 'akalb_lon'
	# x = x.loc[ ~is_outlier( x[ lat_col ], thresh=3.5 ), : ] # remove odd lats
	# x = x.loc[ ~is_outlier( x[ lon_col ], thresh=3.5 ), : ] # remove odd lons

	# get data for first and last rows
	begin_row = x.head( 1 )
	end_row = x.tail( 1 )
	
	# setup some begin-end values requested by funders -- HARDWIRED FIELDS HERE!!! CAREFUL!!!
	bearing_begin, bearing_end = ( begin_row[ 'Direction' ].tolist()[0], end_row[ 'Direction' ].tolist()[0] )
	direction4_begin, direction4_end = ( begin_row[ 'cardinal4' ].tolist()[0], end_row[ 'cardinal4' ].tolist()[0] )
	direction8_begin, direction8_end = ( begin_row[ 'cardinal8' ].tolist()[0], end_row[ 'cardinal8' ].tolist()[0] )
	
	# # calculate between begin/end points bearing -- does this even make sense?
	# # this is easily built in if we just give it a column to live in below
	# lonlat_begin = begin_row[ ['Longitude', 'Latitude'] ].tolist()
	# lonlat_end = end_row[ ['Longitude', 'Latitude'] ].tolist()
	# beginend_bearings = calculate_initial_compass_bearing( (lonlat_begin[0], lonlat_begin[1]), (lonlat_end[0], lonlat_end[1]) )

	time_begin, time_end = ( begin_row[ 'Time' ].tolist()[0], end_row[ 'Time' ].tolist()[0] )
	lon_begin, lon_end = ( begin_row[ 'Longitude' ].tolist()[0], end_row[ 'Longitude' ].tolist()[0] )
	lat_begin, lat_end = ( begin_row[ 'Latitude' ].tolist()[0], end_row[ 'Latitude' ].tolist()[0] )

	out_row = begin_row.drop( ['Longitude', 'Latitude', 'Time', 'Direction', 'cardinal4', 'cardinal8'], axis=1 )
	out_row.index = [0]
	new_cols_df = pd.DataFrame({ 'lon_begin':lon_begin, 'lon_end':lon_end, 'lat_begin':lat_begin, 'lat_end':lat_end, \
								'time_begin':time_begin, 'time_end':time_end, 'bear_begin':bearing_begin, 'bear_end':bearing_end, \
								'dir4_begin':direction4_begin, 'dir4_end':direction4_end, 'dir8_begin':direction8_begin, 'dir8_end':direction8_end }, index = out_row.index)

	out_row = out_row.join( new_cols_df )
	out_row[ 'geometry' ] = [ LineString( zip(x.akalb_lon.tolist(),x.akalb_lat.tolist()) ) ]
	return out_row
def clean_grouped_voyages( df ):
	''' input df grouped by voyages and return cleaned df '''
	# make shapely points and add to a geometry field
	old_index = df.index
	df = df.reset_index( drop=True )
	df.loc[:, 'geometry'] = df.apply( lambda x: Point( x[['Longitude','Latitude']] ), axis=1 )
	# add in Direction, Distance, and simple_direction fields using the above function
	df = insert_direction_distance( df )
	# remove outliers 
	df = remove_outliers_v2( df, max_speed=40.0 )
	# make a geopandas GeoDataFrame
	gdf = gpd.GeoDataFrame( df, crs={'init':'epsg:4326'}, geometry='geometry' )
	# reproject to AKALBERS
	gdf_3338 = gdf.to_crs( epsg=3338 )
	# add 3338 point fields
	ak_lonlat = pd.DataFrame( [{ 'akalb_lon':x.x, 'akalb_lat':x.y } for x in gdf_3338.geometry.tolist()] )
	gdf_3338 = gdf_3338.reset_index( drop=True ).join( ak_lonlat )
	return gdf_3338
def break_goodbad( df, land ):
	''' 
	break the voyages that intersect land from voyages that dont 
	return 2 dataframes in order good, bad 

	'''
	if LineString( df.geometry.tolist() ).intersects( land ):
		return True
	else:
		return False

if __name__ == '__main__':
	import pandas as pd
	import numpy as np
	from geopy.distance import vincenty
	import datetime, math, os, glob, argparse
	from functools import partial
	import geopandas as gpd
	from pyproj import Proj
	from shapely.geometry import Point, LineString
	from collections import OrderedDict
	from pathos.mp_map import mp_map
	from pathos.pp_map import pp_map
	from pathos import multiprocessing as mp
	from dask import dataframe as dd
	
	parser = argparse.ArgumentParser( description='program to add Voyage and Direction fields to the AIS Data' )
	parser.add_argument( "-p", "--output_path", action='store', dest='output_path', type=str, help='path to output directory' )
	parser.add_argument( "-fn", "--fn", action='store', dest='fn', type=str, help='path to input filename to run' )
	parser.add_argument( "-lfn", "--land_fn", action='store', dest='land_fn', type=str, help='path to land filename to be used for finding non-compliant voyages' )
	
	# parse all the arguments
	args = parser.parse_args()
	fn = args.fn
	output_path = args.output_path
	land_fn = args.land_fn

	ncpus = 32
	print 'working on: %s' % os.path.basename( fn )

	# make some output filenaming base for outputs
	output_fn_base = os.path.basename( fn ).split( '.' )[0] + '_voyage_direction_phase3'

	# read in the csv to a PANDAS DataFrame
	df = pd.read_csv( fn, sep=',' )
	land = gpd.read_file( land_fn ).geometry[0] # multipolygon

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

	# returns a new column called clusters which groups to voyages
	MMSI_grouped = df.groupby( 'MMSI' ).apply( group_voyages )

	# lets dig into the data a bit: we are going to keep only transects with > 100 pingbacks since that seems like a fairly short trip @ ~30 sec intervals
	# this could transform into something that looks at the intervals between each timestep and decides whether to drop it. instead of ping counts
	# since we are dropping Voyages with <100 anyhow drop those files now
	if df.shape[0] > 100:
		unique_counts_df = pd.DataFrame( np.array( np.unique( MMSI_grouped.clusters, return_counts=True ) ).T, columns=[ 'unique', 'count' ] )

		keep_list = unique_counts_df.loc[ unique_counts_df['count'] > 100, 'unique' ]
		MMSI_grouped_keep = MMSI_grouped[ MMSI_grouped.clusters.isin( keep_list ) ].copy()

		def fix_voyage_id( MMSI_group_df ):
			# grouped = MMSI_group_df.groupby( 'clusters' )
			unique_vals = MMSI_group_df.clusters.unique()
			MMSI = unique_vals[0].split( '_' )[0]
			unique_vals_count = len( unique_vals )
			for unique, new_val in zip( unique_vals, range(unique_vals_count) ):
				new_val = new_val+1
				MMSI_group_df.loc[ MMSI_group_df.clusters == unique, 'Voyage' ] = MMSI_group_df[ MMSI_group_df.clusters == unique ].clusters.apply( lambda x: x.split( '_' )[0] + '_' + str(new_val) )
			return MMSI_group_df

		# # add in the Voyage column -- the unique id of MMSI and unique transect number
		new_groups = MMSI_grouped_keep.clusters.apply( lambda x: x.split( '_' )[0] ).tolist()
		MMSI_grouped_keep = MMSI_grouped_keep.groupby( new_groups ).apply( fix_voyage_id )

		# MMSI_grouped_keep.loc[ :, 'Voyage' ] = MMSI_grouped_keep.loc[ :, 'clusters' ]
		# voyage_group_names = grouped.groups.keys()

		# run the voyage cleaner function on the grouped voyage data frames
		# gdf_3338 = MMSI_grouped_keep.groupby( 'Voyage' ).apply( clean_grouped_voyages( df ) )
		# parallelize it 
		MMSI_grouped_voyages = pd.Series([ j.copy() for i,j in MMSI_grouped_keep.groupby( 'Voyage' ) ])
		
		del MMSI_grouped_keep, df # cleanup

		print ('  running voyage cleaner')
		if len( MMSI_grouped_voyages ) >= 2000:
			splitter = np.array_split( range( len( MMSI_grouped_voyages ) ), int( len( MMSI_grouped_voyages ) / 1000 ) )
			out = [ mp_map( clean_grouped_voyages, sequence=MMSI_grouped_voyages[ i ], nproc=ncpus ) for i in splitter ]
			# unlist
			out = [ j for i in out for j in i ]
		else:
			out = mp_map( clean_grouped_voyages, sequence=MMSI_grouped_voyages, nproc=ncpus )

		df = pd.concat( ( i for i in out if i.shape[0] > 0 ) )

		del MMSI_grouped_voyages # cleanup

		# run the intersect testing
		MMSI_grouped_goodbad = pd.Series([ j.copy() for i,j in df.groupby( 'Voyage' ) ])
		break_goodbad_partial = partial( break_goodbad, land=land ) # partial function build
		
		if len( MMSI_grouped_goodbad ) >= 2000:
			splitter = np.array_split( range( len( MMSI_grouped_goodbad ) ), int( len( MMSI_grouped_goodbad ) / 1000 ) )
			intersect_test = [ mp_map( break_goodbad_partial, sequence=MMSI_grouped_goodbad[ i ], nproc=ncpus ) for i in splitter ]
			intersect_test = [ j for i in intersect_test for j in i ] # unlist it
		else:
			intersect_test = mp_map( break_goodbad_partial, sequence=MMSI_grouped_goodbad, nproc=ncpus )

		# # # # # 
		del out # cleanup

		# make an output directory to store the csvs and shapefiles if needed
		if not os.path.exists( os.path.join( output_path, 'csvs' ) ):
			os.makedirs( os.path.join( output_path, 'csvs' ) )
		
		if not os.path.exists( os.path.join( output_path, 'shapefiles','points' ) ):
			os.makedirs( os.path.join( output_path, 'shapefiles', 'points' ) )

		if not os.path.exists( os.path.join( output_path, 'shapefiles','lines' ) ):
			os.makedirs( os.path.join( output_path, 'shapefiles', 'lines' ) )
		
		# set up 'bad' voyage directories
		if not os.path.exists( os.path.join( output_path, 'csvs', 'bad' ) ):
			os.makedirs( os.path.join( output_path, 'csvs', 'bad' ) )
		
		if not os.path.exists( os.path.join( output_path, 'shapefiles','points', 'bad' ) ):
			os.makedirs( os.path.join( output_path, 'shapefiles', 'points', 'bad' ) )

		if not os.path.exists( os.path.join( output_path, 'shapefiles','lines', 'bad' ) ):
			os.makedirs( os.path.join( output_path, 'shapefiles', 'lines', 'bad' ) )

		# a hardwired set of column names and dtypes for output csv and shapefile
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
											('cardinal4', np.object),
											('cardinal8', np.object),
											('akalb_lon', np.object),
											('akalb_lat', np.object)] )

		# this has an issue with converting DTYPES since they are of mixed types...  lets try to fix it above...
		gdf_3338_csv = gpd.GeoDataFrame( pd.DataFrame( { col:df[ col ].astype( dtype ) for col, dtype in COLNAMES_DTYPES_DICT.iteritems() } ), \
											crs={'init':'epsg:3338'}, geometry=df.geometry )

		gdf_3338_csv_grouped = pd.Series( [ j for i, j in gdf_3338_csv.groupby( 'Voyage' ) ] )

		# use the intersects test we calculated above to break out good and bad dataframes
		intersect_test = pd.Series( intersect_test )

		if False in intersect_test: # no intersection with polygons
			gdf_3338_csv = gpd.GeoDataFrame( pd.concat( gdf_3338_csv_grouped[ -intersect_test ].tolist() ), crs={'init':'epsg:3338'}, geometry='geometry' )

		if True in intersect_test.tolist(): # intersection with polygons aka 'bad'
			gdf_3338_csv_bad = gpd.GeoDataFrame( pd.concat( gdf_3338_csv_grouped[ intersect_test ].tolist() ), crs={'init':'epsg:3338'}, geometry='geometry' )
		else:
			gdf_3338_csv_bad = None

		# write it out to a point shapefile
		print( '  writing point shapefiles' )
		output_filename_pts = os.path.join( output_path, 'shapefiles', 'points', output_fn_base+'_points.shp' )
		gdf_3338_csv.to_file( output_filename_pts )
		
		# bad
		if gdf_3338_csv_bad is not None:
			output_filename_pts = os.path.join( output_path, 'shapefiles', 'points', 'bad', output_fn_base+'_points.shp' )
			gdf_3338_csv_bad.to_file( output_filename_pts )

		# write it out to a csv
		print( '  writing point csvs' )
		output_filename = os.path.join( output_path, 'csvs', output_fn_base+'.csv' )
		gdf_3338_csv.drop( 'geometry', 1 ).to_csv( output_filename, sep=',' )
		# group the data into Voyages
		voyages_grouped = gdf_3338_csv.groupby( 'Voyage' )

		# bad
		if gdf_3338_csv_bad is not None:
			output_filename = os.path.join( output_path, 'csvs', 'bad' , output_fn_base+'.csv' )
			gdf_3338_csv_bad.drop( 'geometry', 1 ).to_csv( output_filename )
			voyages_grouped_bad = gdf_3338_csv_bad.groupby( 'Voyage' )
			
		print( '  writing lines shapefiles' )

		# make lines
		gdf_mod = voyages_grouped.apply( line_it ) # add new fields for output shapefile
		# del gdf_3338_csv
		gdf_mod = gdf_mod.reset_index( drop=True )
		gdf_mod = gpd.GeoDataFrame( gdf_mod, crs={'init':'epsg:3338'}, geometry='geometry' )

		output_filename = os.path.join( output_path, 'shapefiles', 'lines', output_fn_base+'.shp' )
		if isinstance( gdf_mod, gpd.GeoDataFrame ) and isinstance( gdf_mod.geometry, gpd.GeoSeries ):
			# make geo and output as a shapefile
			# output_filename = output_filename.replace( '.csv', '.shp' ).replace( 'csvs', 'shapefiles/lines' )
			gdf_mod.to_file( output_filename )
		else:
			# output_filename = output_filename.replace( '.csv', '.shp' ).replace( 'csvs', 'shapefiles/lines' )
			gdf_mod.geometry = gpd.GeoSeries( gdf_mod.geometry )
			gdf_mod = gpd.GeoDataFrame( gdf_mod, crs={'init':'epsg:3338'}, geometry='geometry' )
			gdf_mod.to_file( output_filename )
			print( '  check lines GeoDataFrame' )

		# bad
		if gdf_3338_csv_bad is not None:
			gdf_mod_bad = voyages_grouped_bad.apply( line_it ) # add new fields for output shapefile
			gdf_mod_bad = gdf_mod_bad.reset_index( drop=True )
			gdf_mod_bad = gpd.GeoDataFrame( gdf_mod_bad, crs={'init':'epsg:3338'}, geometry='geometry' )

			output_filename = os.path.join( output_path, 'shapefiles', 'lines', 'bad', output_fn_base+'.shp' )
			# bad line output
			if isinstance( gdf_mod, gpd.GeoDataFrame ) and isinstance( gdf_mod.geometry, gpd.GeoSeries ):
				# make geo and output as a shapefile
				# output_filename = output_filename.replace( '.csv', '.shp' ).replace( 'csvs', 'shapefiles/lines/bad' )
				gdf_mod_bad.to_file( output_filename )
			else:
				# output_filename = output_filename.replace( '.csv', '.shp' ).replace( 'csvs', 'shapefiles/lines/bad' )
				gdf_mod_bad.geometry = gpd.GeoSeries( gdf_mod_bad.geometry )
				gdf_mod_bad = gpd.GeoDataFrame( gdf_mod_bad, crs={'init':'epsg:3338'}, geometry='geometry' )
				gdf_mod_bad.to_file( output_filename )
				print( '  check lines bad GeoDataFrame' )
	else:
		print 'Unable to Generate Lines for : %s ' % os.path.basename( fn )


# # # # # # # # # # # # # # # # # # 
# # # how to run this application:
# if __name__ == '__main__':
# 	import glob, os

# 	# set the path to where the script file is stored: [hardwired]
# 	os.chdir( '/workspace/Shared/Tech_Projects/Marine_shipping/project_data/CODE/ShippingLanes' )

# 	# list the files to run and set output path
# 	l = glob.glob( '/atlas_scratch/malindgren/ShippingLanes_PhaseIII/Thu_Sep_4_2014_121625/csv/grouped/*.csv' )
# 	# l = glob.glob( '/workspace/Shared/Tech_Projects/Marine_shipping/project_data/Output_Data/Thu_Sep_4_2014_121625/csv/grouped/*.csv' )
# 	output_path = '/atlas_scratch/malindgren/ShippingLanes_PhaseIII/Output_Data_NOVEMBER' # /workspace/Shared/Tech_Projects/Marine_shipping/project_data/Phase_III
# 	land_fn = '/workspace/Shared/Tech_Projects/Marine_shipping/project_data/Ancillary_Data/shoreline_shapefile/Bering_Chukchi_Shoreline_3338_aoi.shp' # '/workspace/Shared/Tech_Projects/Marine_shipping/project_data/Phase_III/Output_Data_NOVEMBER/Bering_Chucki_Shoreline_3338_multi.shp'
# 	command_start = 'ais_shipping_voyage_splitter_phase3.py -p ' + output_path + ' -fn '

# 	for i in l:
# 		try:
# 			os.system( 'ipython -c "%run ' + command_start + i + ' -lfn ' + land_fn + '"')
# 		except:
# 			print 'ERROR RUN %s: ' % os.path.basename( i )
# 			pass
