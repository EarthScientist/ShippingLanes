import os, sys, re, arcpy, arcpy.mapping

# the output path
outPath = r"X:\projects\ShippingLanes\AIS_Satellite\products\PNG_MSS_conf"

# read in the mxd document
# mxd = arcpy.mapping.MapDocument(r"X:\projects\ShippingLanes\AIS_Satellite\products\outputs_v1\AIS_Satellite_v1_points.mxd")
##mxd = arcpy.mapping.MapDocument('X:\projects\ShippingLanes\AIS_Satellite\products\outputs_v1\AIS_Satellite_v1_points.mxd')
mxd = arcpy.mapping.MapDocument('CURRENT')
	
# get the data frame
df = arcpy.mapping.ListDataFrames(mxd)[0]

# list the layers
layers=arcpy.mapping.ListLayers(mxd,"*",df)
##ext=arcpy.mapping.Layer(r"X:\projects\ShippingLanes\OUTPUTS\JULY_2013\Density\output_extent_polygon.shp")
## ext=arcpy.mapping.Layer(r"X:\projects\ShippingLanes\ancillary\AIS_Satellite_Domain_rectangle_v2.shp")
extents = ["x:/projects/ShippingLanes/AIS_Satellite/products/ancillary/AleutianExtent_Mapping2.shp",\
"x:/projects/ShippingLanes/AIS_Satellite/products/ancillary/AleutianExtent_Mapping3_AdaktoAttu.shp","x:/projects/ShippingLanes/AIS_Satellite/products/ancillary/AleutianExtent_Mapping3_toAdak.shp",\
"x:/projects/ShippingLanes/AIS_Satellite/products/ancillary/BeringIslandsExtent_Mapping.shp"]

# turn on the layers we need for basemap
for layer in layers:
	if layer.name == 'Ocean_Basemap':
		base1 = layer
		base1.visible = True
	if layer.name == 'World Reference Overlay':
		base2 = layer
		base2.visible = True
	if layer.name == 'World Physical Map':
		base3 = layer
		base3.visible = True
	if layer.name == 'ABSI_LCC_Extent':
		base4 = layer
		base4.visible = True


print "setup complete!"


# give the extent to the whole df
for layer in layers:
	print layer.name
	if "pointcount" or "akalb" in layer.name:
		for extent in extents:
			#extent stuff
			ext = arcpy.mapping.Layer(extent)
							
			print(ext.name)

			layer.visible = True
			arcpy.RefreshActiveView()
			df.extent = ext.getExtent()
			
			if ".tif" in layer.name:  
				outname = layer.name[:layer.name.find('.tif') + 1]
				outname = outname.strip(".")
				outname = outname+"_"+ext.name
			else:
				outname = layer.name+"_"+ext.name
			
			outName = os.path.join(outPath,outname+".png")
			arcpy.mapping.ExportToPNG(mxd, outName, df, color_mode="24-BIT_TRUE_COLOR") # df_export_width=3200, df_export_height=2400, resolution=480,
			arcpy.RefreshActiveView()
		
		layer.visible = False
		arcpy.RefreshActiveView()




