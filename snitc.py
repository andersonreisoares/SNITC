import rasterio
import numpy as np
import math
import fiona
import cv2
from dtaidistance import dtw
from rasterio import features
from shapely.geometry import shape, mapping, MultiPolygon
from scipy import stats
from sys import exit
from osgeo import ogr
from dtaidistance import dtw

def create_polygon(list_of_observations):
    list_of_radius, list_of_angles = get_list_of_points(list_of_observations)

    # create polygon geometry
    ring = ogr.Geometry(ogr.wkbLinearRing)
    # add points in the polygon
    N = len(list_of_radius)
    for i in range(N):
        a = list_of_radius[i] * math.cos(2 * np.pi * i / N)
        o = list_of_radius[i] * math.sin(2 * np.pi * i / N)
        ring.AddPoint(a, o)
    # add first point in the last position to close the ring
    a = list_of_radius[0] * math.cos(2 * np.pi * 0 / N)
    o = list_of_radius[0] * math.sin(2 * np.pi * 0 / N)
    ring.AddPoint(a, o)

    polygon = ogr.Geometry(ogr.wkbPolygon)
    polygon.AddGeometry(ring)

    return polygon

def get_list_of_points(list_of_observations):
    # remove negative points
    positive_list_of_observations = [0 if i < 0 else i for i in list_of_observations]
    # round_list_of_observations = numpy.append(positive_list_of_observations, positive_list_of_observations[0])
    # list_of_angles = numpy.arange(0, 2 * numpy.pi, 2 * numpy.pi / (len(positive_list_of_observations)))
    list_of_angles = np.linspace(0, 2 * np.pi, len(positive_list_of_observations))
    # round_list_of_angles = numpy.append(list_of_angles, list_of_angles[0])
    # return round_list_of_observations, round_list_of_angles
    return positive_list_of_observations, list_of_angles

def polar_rolls(time_series_1, time_series_2):
    dist = math.inf
    pos = math.inf
    polygon_1 = create_polygon(time_series_1)
    
    if polygon_1.IsValid() == False:
        polygon_1 = polygon_1.Buffer(0)
    
    if min(time_series_1) > max(time_series_2) or min(time_series_2) > max(time_series_1):
        polygon_2 = create_polygon(time_series_2)
        polygons_symmetric_difference = polygon_1.SymDifference(polygon_2)
        dist = polygons_symmetric_difference.Area()
        
    else:
        for i in range(len(time_series_1)):
            shifted_time_series_2 = np.roll(time_series_2, i)
            temp = np.linalg.norm(time_series_1-shifted_time_series_2)

            if temp<dist:
                dist = temp
                pos = i
               
        time_series_2 = np.roll(time_series_2,pos)
        polygon_2 = create_polygon(time_series_2)
        if polygon_2.IsValid():
            polygons_symmetric_difference = polygon_1.SymDifference(polygon_2)
            dist = polygons_symmetric_difference.Area()
        else:
            polygon_2 = polygon_2.Buffer(0)
            polygons_symmetric_difference = polygon_1.SymDifference(polygon_2)
            dist = polygons_symmetric_difference.Area()
            
    return dist

def distance_polar(C, subim, S, m, rmin, cmin):
    
    #Initialize submatrix
    dc = np.zeros([subim.shape[1],subim.shape[2]])
    ds = np.zeros([subim.shape[1],subim.shape[2]])
            
    #get cluster centres
    a2 = C[:subim.shape[0]]                                #Average time series
    ic = (int(np.floor(C[subim.shape[0]])) - rmin)         #X-coordinate
    jc = (int(np.floor(C[subim.shape[0]+1])) - cmin)       #Y-coordinate
    
    # Critical Loop - need parallel implementation
    for u in range(subim.shape[1]):
        for v in range(subim.shape[2]):
            a1 = subim[:,u,v]                                           # Get pixel time series 
            dc[u,v] = polar_rolls(a1.astype(float),a2.astype(float))
            ds[u,v] = (((u-ic)**2 + (v-jc)**2)**0.5)                         #Calculate Spatial Distance
    D =  (dc**2 + (m**2) * (ds/S)**2)**0.5                                 #Calculate SPatial-temporal distance
            
    return D

def distance_slic_fast(C, subim, S, m, rmin, cmin):
    
    #Initialize submatrix
    ds = np.zeros([subim.shape[1],subim.shape[2]])
    
    #get cluster centres
    a2 = C[:subim.shape[0]]                                #Average time series
    ic = (int(np.floor(C[subim.shape[0]])) - rmin)         #X-coordinate
    jc = (int(np.floor(C[subim.shape[0]+1])) - cmin)       #Y-coordinate
    
    subset = subim[:,:,:]
    asds = subim.shape[1]*subim.shape[2]
    
    linear = subim.transpose(1,2,0).reshape(asds,subim.shape[0])
    merge  = np.vstack((linear,a2))

    c = dtw.distance_matrix_fast(merge, block=((0, merge.shape[0]), (merge.shape[0]-1,merge.shape[0])), compact=True, parallel=True)
    dc = c.reshape(subim.shape[1],subim.shape[2])

    # Critical Loop - need parallel implementation
    for u in range(subim.shape[1]):
        for v in range(subim.shape[2]):
            ds[u,v] = (((u-ic)**2 + (v-jc)**2)**0.5)                         #Calculate Spatial Distance
    
    D = (dc**2 + m**2 * (ds / S)**2)**0.5                                 #Calculate SPatial-temporal distance
             
    return D

def distance_slic(C, subim, S, m, rmin, cmin):
    
    #Initialize submatrix
    dc = np.zeros([subim.shape[1],subim.shape[2]])
    ds = np.zeros([subim.shape[1],subim.shape[2]])
            
    #get cluster centres
    a2 = C[:subim.shape[0]]                                #Average time series
    ic = (int(np.floor(C[subim.shape[0]])) - rmin)         #X-coordinate
    jc = (int(np.floor(C[subim.shape[0]+1])) - cmin)       #Y-coordinate
    
    # Critical Loop - need parallel implementation
    for u in range(subim.shape[1]):
        for v in range(subim.shape[2]):
            a1 = subim[:,u,v]                                              # Get pixel time series 
            dc[u,v] = dtw.distance_fast(a1.astype(float),a2.astype(float)) #Compute DTW distance
            ds[u,v] = (((u-ic)**2 + (v-jc)**2)**0.5)                       #Calculate Spatial Distance
    D =  (dc**2 + (m**2) * (ds/S)**2)**0.5                                 #Calculate SPatial-temporal distance
            
    return D

def update_cluster(C,img,l,rows,columns,bands,k):

    #Allocate array info for centres
    C = np.zeros([k,bands+3]).astype(float) 

    #Update cluster centres with mean values
    for r in range(rows):
        for c in range(columns):
            tmp = np.append(img[:,r,c],np.array([r,c,1]))
            kk = l[r,c].astype(int)
            C[kk,:] = C[kk,:] + tmp
  
    #Compute mean
    for kk in range(k):
        C[kk,:] = C[kk,:]/C[kk,bands+2]
    
    return C


def makeregiondistinct(H):
    #create disjoint regions on numpy array 
    
    labels = np.unique(H.astype(int))         #get regions labels
    maxlabel = np.max(labels)                 #get max label
    out = np.zeros([H.shape[0],H.shape[1]])   #allocate new numpy array
    
    for i in range(np.max(labels)):           #for each label define the disjoint regions 
        l_loop = np.where(H == i, H, 0)
        num_cc,cc = cv2.connectedComponents(l_loop.astype(dtype = np.uint8), 4);  #Get new regions using 4-connected restriction

        for n in range(1,num_cc):            #relabel the new regions
            maxlabel = maxlabel+1;           #atribute a new label
            out = np.where(cc == n, maxlabel, out)
     
    labels = None
    maxlabel=None
    
    return out

def renumberregions(L):
    #Renumber regions from 1 to N
    
    list = np.unique(L.astype(np.uint32))               #Get the list of labeled regions
    relabeled = np.zeros([L.shape[0],L.shape[1]])       #Allocate on memory image after relabelling
    label = 1
        
    for i in list:
        relabeled = np.where(L == i, label, relabeled)  #For each region on numpy array relabel
        label = label+1
           
    return relabeled

def postprocessing(l,S):
    
    #Remove spourious regions generated during segmentation
    dump = makeregiondistinct(l+1)                  #Generate disjoint regions
    relabeled = renumberregions(dump)               #relabel those regions
    dump = None
    
    #Remove regions smaller than S
    #Since S define the average expected size of superpixels, regions smaller than it do not make strong sense
    #Use Connectivity as 4 to avoid undesired connections
    im_clean = rasterio.features.sieve(relabeled.astype(dtype=np.int32),int(S),connectivity = 4)
    
    #DO OVER, disjoint and renumbering
    dump = makeregiondistinct(im_clean)
    final = renumberregions(dump)
    
    dump = None
    relabeled = None
    return final

def write_shp(final_segmentation,name,meta):

    #Get-Set transform and CRS
    transform = meta["transform"]
    crs = meta["crs"]
    
    #Define shapefile schema
    shp_schema = {
        'geometry': 'MultiPolygon',
        'properties': {'pixelvalue': 'int'}
    }
    
    # select the records from shapes where the value is 1,
    # or where the mask was True
    unique_values = np.unique(final_segmentation)
    
    #Use fiona to write shapefile
    with fiona.open((name+'.shp'), 'w', 'ESRI Shapefile', shp_schema, crs.data) as shp:
        for pixel_value in unique_values: #attribbute the pixels with same value to one polygon
            polygons = [shape(geom) for geom, value in rasterio.features.shapes(final_segmentation.astype(dtype = np.int32), transform=transform)
                        if value == pixel_value]
            multipolygon = MultiPolygon(polygons)
            shp.write({
                'geometry': mapping(multipolygon),
                'properties': {'pixelvalue': int(pixel_value)}
            })
            
    shp = None

def SNITC(filename,k,m,shape_name):

    print('Simple Non-Linear Iterative Temporal Clustering V 1.0')
    
    ##READ FILE
    dataset = rasterio.open(filename)
    img = dataset.read() 
    meta = dataset.profile #get image metadata
    transform = meta["transform"]
    crs = meta["crs"]
    
    #img = img[1:24,:,:]
    ##############################
    #Corte temporal do cubo do Hugo
    #img = img[106:160,:,:]
    ##############################
    m = m/10
    #Normalize data
    for band in range(img.shape[0]):
        img[np.isnan(img)] = 0
        img[band,:,:] = img[band,:,:]+abs(np.min(img[band,:,:]))
    #    img = (img/np.max(img[band,:,:]))
    
    #Get image dimensions
    bands = img.shape[0]
    rows = img.shape[1]
    columns = img.shape[2]
    N = rows * columns
    
    #Setting up SNITC
    S = (rows*columns / (k * (3**0.5)/2))**0.5

    #Get nodes per row allowing a half column margin at one end that alternates
    nodeColumns = round(columns/S - 0.5);
    #Given an integer number of nodes per row recompute S
    S = columns/(nodeColumns + 0.5); 

    # Get number of rows of nodes allowing 0.5 row margin top and bottom
    nodeRows = round(rows/((3)**0.5/2*S));
    vSpacing = rows/nodeRows;

    # Recompute k
    k = nodeRows * nodeColumns;

    # Allocate memory and initialise clusters, labels and distances.
    C = np.zeros([k,bands+3])                 # Cluster centre data  1:times is mean on each band of series
                                              # times+1 and times+2 is row, col of centre, times+3 is No of pixels
    l = -np.ones([rows,columns])              # Matrix labels.
    d = np.full([rows,columns], np.inf)       # Pixel distance matrix from cluster centres.

    # Initialise grid
    kk = 0;
    r = vSpacing/2;
    for ri in range(nodeRows):
        x = ri      #?
        if x % 2:
            c = S/2
        else:
            c = S

        for ci in range(nodeColumns):
            cc = int(np.floor(c)); rr = int(np.floor(r))
            ts = img[:,rr,cc]
            st = np.append(ts,[rr,cc,0])
            C[kk, :] = st
            c = c+S
            kk = kk+1

        r = r+vSpacing
    
    #FIX S
    S = round(S)
    
    #Start clustering
    for n in range(5):
        print(n)
        for kk in range(k):
            
            # Get subimage around cluster
            rmin = int(np.floor(max(C[kk,bands]-S, 0)));         rmax = int(np.floor(min(C[kk,bands]+S, rows))+1);   
            cmin = int(np.floor(max(C[kk,bands+1]-S, 0)));       cmax = int(np.floor(min(C[kk,bands+1]+S, columns))+1); 
            
            #Create subimage 2D np.array
            subim = img[:,rmin:rmax,cmin:cmax];  
            
            #Calculate Spatio-temporal distance
            #D = distance_slic_fast(C[kk, :], subim, S, m, rmin, cmin)
            D = distance_polar(C[kk, :], subim, S, m, rmin, cmin)
            
            subd = d[rmin:rmax,cmin:cmax]
            subl = l[rmin:rmax,cmin:cmax]
            
            #Check if Distance from new cluster is smaller than previous
            subl = np.where( D < subd, kk, subl)  
            subd = np.where( D < subd, D, subd)       
            
            #Replace the pixels that had smaller difference
            d[rmin:rmax,cmin:cmax] = subd
            l[rmin:rmax,cmin:cmax] = subl
            
        C = update_cluster(C,img,l,rows,columns,bands,k)          #Update Clusters
        
    print('Fixing segmentation')
    final_segmentation = postprocessing(l,S)                 #Remove noise from segmentation
    
    #print('Writing shapefile')
    #write_shp(final_segmentation,shape_name,meta)
        
    dataset = None
    
    return final_segmentation                                 #Return labeled np.array for visualization on python