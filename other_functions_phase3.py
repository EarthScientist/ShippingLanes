
# old distance calculation used before we found GeoPy
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

