# # 
# script to derive some summary metrics for the ABSI LCC group
# relating to the ability to parse the data and get it to a useable form
# 
# Developed By: Michael Lindgren ( malindgren@alaska.edu ) Spatial Analyst at 
#				Scenarios Network for Alaska & Arctic Planning 

if __name__ == '__main__':
	import pandas as pd
	import glob, os, sys, re, time
	import numpy as np
	import os, json, rasterio, fiona, dill
	import pathos.multiprocessing as mp

	base_path = '/workspace/Shared/Tech_Projects/Marine_shipping/project_data/Output_Data/Thu_Sep_4_2014_121625'
	ihs_path = '/workspace/Shared/Tech_Projects/Marine_shipping/project_data/CODE/ShippingLanes/ETC/IHS_ShipData_COMBINED_MLedit.csv'

	files = glob.glob( os.path.join( base_path, 'csv', 'cleaned','*_clean.csv' ) )
	ihs_df = pd.read_csv( ihs_path, usecols=[ 'MMSI' ], dtype=str, error_bad_lines=False )
	ihs_df.MMSI = ihs_df.MMSI.astype( int )

	def get_match( x, ihs_df=ihs_df ):
		'''
		return some metrics about the data.
		'''
		cur_df = pd.read_csv( x, usecols=[ 'MMSI' ], dtype=str, error_bad_lines=False )
		cur_df.MMSI = cur_df.MMSI.astype( int )
		nrows = cur_df.shape[ 0 ]
		output = [ i for i in cur_df.MMSI.tolist() if i not in ihs_df.MMSI.tolist() ]
		return output, nrows

	# run it in parallel with 10 cores
	p = mp.Pool( 10 )
	out = p.map( get_match, files )

	# unlist the nested lists:
	out_unlist_records = [ j for i,j in out for k in i ]
	out_unlist_nrows = [ j for i,j in out ]

	os.mkdir( os.path.join( base_path, 'csv', 'missing_data_metrics' ) )

	# write the unlisted data to a file so we dont have to run the processing again
	np.array( out_unlist_records ).tofile( os.path.join( base_path,
		'csv','missing_data_metrics','AIS_Satellite_v3_missing_mmsi_ihs_table_all_raw.txt' ),
		sep=',' )  

	# total number of parseable records:
	total_records = sum( out_unlist_nrows )

	# How many bad records were there in the output data
	# - That is how many records could be read successfully from the data recieved from ExactEarth
	#  that were not used.
	records_not_used_count = len( out_unlist_records )

	# number of records not used that are actually valid MMSI 9-digit integers
	records_not_used_count_valid = len( [ i for i in out_unlist_records if len( str( i ) ) == 9 ] )

	# Which are the still missing MMSI's
	missing_unique_mmsi = np.unique( out_unlist_records ).tolist( )

	# write this data to disk
	np.array( missing_unique_mmsi ).tofile( os.path.join( base_path,
		'csv','missing_data_metrics','AIS_Satellite_v3_missing_mmsi_ihs_table_all_uniques.txt' ),
		 sep=',' )

	# how many missing unique MMSI's are there?
	missing_unique_mmsi_count = len( missing_unique_mmsi )

	# which are still missing MMSI's that actually meet the 9-digit criteria?
	missing_unique_mmsi_valid = [ i for i in missing_unique_mmsi if len( str( i ) ) == 9 ]

	# how many of the valid missing MMSI's are there?
	missing_unique_mmsi_valid_count = len( missing_unique_mmsi_valid )

	# write this data to disk
	np.array( missing_unique_mmsi_valid ).tofile( os.path.join( base_path,
		'csv','missing_data_metrics','AIS_Satellite_v3_missing_mmsi_ihs_table_valid_uniques.txt' ),
		 sep=',' )

	# Now it is necessary to figure out which of the MMSI's were actually used in the analysis
	#	unfortunately, to do this we need to run the entire read sequence again due to an oversight
	# # this would be better included in the above multiprocessing pool function.  This is just a time saver
	files = glob.glob( os.path.join(base_path,'csv','dropped','*_dropcols.csv') )

	def get_MMSI_used( x ):
		'''
		return some metrics about the data.
		'''
		cur_df = pd.read_csv( x, usecols=[ 'MMSI' ], dtype=str, error_bad_lines=False )
		cur_df.MMSI = cur_df.MMSI.astype( int )
		return np.unique( cur_df.MMSI )

	# run it in parallel with 14 cores
	p = mp.Pool( 14 )
	out = p.map( get_MMSI_used, files )
	final_MMSI_used = np.unique( [ j for i in out for j in i ] )

	# write that result out to a text file for later use.
	np.array( final_MMSI_used ).tofile( os.path.join( base_path,
		'csv','missing_data_metrics','AIS_Satellite_v3_mmsi_used_in_analysis_uniques.txt' ),
		 sep=',' )

	# now lets print a final output report:
	with open( os.path.join( base_path, 'csv', 'missing_data_metrics', 'AIS_Satellite_v3_output_summary_report_missing_mmsi.txt' ), mode = 'w' ) as output_report:
		output_report.write( '- - - - - - - - - - - - - - - - - - - - - - - - - - -' + '\n' )
		output_report.write( '' + '\n' )
		output_report.write( 'ExactEarth AIS Satellite Data Parsing Metrics <> ' + time.asctime() + '\n' )
		output_report.write( '' + '\n' )
		output_report.write( 'Total Number of Parseable rows from the raw ExactEarth data: ' + str( total_records ) + '\n' )
		output_report.write( '' + '\n' )
		output_report.write( 'Number of Rows parseable, but not used: ' + str( records_not_used_count ) + '\n' )
		output_report.write( '' + '\n' )
		output_report.write( 'Number of Rows parseable, but not used with valid MMSI integers: ' + str( records_not_used_count_valid ) + '\n' )
		output_report.write( '' + '\n' )
		output_report.write( 'Number of Unique MMSIs that were parseable, but not used: ' + str( missing_unique_mmsi_count ) + '\n' )
		output_report.write( '' + '\n' )
		output_report.write( 'Number of Unique MMSIs that were parseable, but not used with valid MMSI integers: ' + str( missing_unique_mmsi_valid_count ) + '\n' )
		output_report.write( '' + '\n' )
		output_report.write( 'Number of unique MMSI that were used in the final analysis: ' + str( len( final_MMSI_used ) ) + '\n' )
		output_report.write( '' + '\n' )
		output_report.write( '' + '\n' )
		output_report.write( 'Please see other report documents for the lists of these actual values if they are needed for further processing later on.' + '\n' )
		output_report.write( 'Report Generated By: Michael Lindgren (malindgren@alaska.edu) www.snap.uaf.edu' + '\n' )
		output_report.write( '- - - - - - - - - - - - - - - - - - - - - - - - - - -' + '\n' )

	print( 'Successfully calculated needed summary metrics.' )

