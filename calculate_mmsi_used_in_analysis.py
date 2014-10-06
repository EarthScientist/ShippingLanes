# # 
# script to derive some summary metrics for the ABSI LCC group
# relating to the ability to parse the data and get it to a useable form
#  --> here we want to see what MMSI values were used in the analysis
# 
# Developed By: Michael Lindgren ( malindgren@alaska.edu ) Spatial Analyst at 
#				Scenarios Network for Alaska & Arctic Planning 
#

if __name__ == '__main__':
	import pandas as pd
	import glob, os, sys, re, time
	import numpy as np
	import os, json, rasterio, fiona, dill
	import pathos.multiprocessing as mp

	base_path = '/workspace/Shared/Tech_Projects/Marine_shipping/project_data/Output_Data/Thu_Sep_4_2014_121625'
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

