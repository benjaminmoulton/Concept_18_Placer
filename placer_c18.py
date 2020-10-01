import numpy as np
import json
import dxf
import os
import C18_geometry as C18
import zbump_nonmorphing as zmorph
from scipy import interpolate as interp
import machupX as mx
import airfoil_db as adb
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# function that reads in a json file as a dictionary
def read_json(filename):
    # import json file
    json_string=open(filename).read()
    vals = json.loads(json_string)
    return vals

# read in MachUpX distributions file
def read_dist(filename):
    # initialize data dictionary
    info = []

    # open file
    with open(filename,"r") as fn:
        col_names = fn.readline()

        # read in data
        for line in fn:
            info.append([float(point) for point in line.split()[2:]])
        
        # make a numpy 2d array
        info = np.matrix(info)

        # close file
        fn.close()
    
    return info

def read_dist_new(filename):
    # read in file
    dist_dict = read_json(filename)

    # fin the length of each list
    length = len(dist_dict["chord"])

    # initialize array 
    dist = np.zeros((length,5))

    # add parts
    dist[:,0] = np.array(dist_dict["cpx"])
    dist[:,1] = np.array(dist_dict["cpy"])
    dist[:,2] = np.array(dist_dict["cpz"])
    dist[:,3] = np.array(dist_dict["chord"])
    dist[:,4] = np.array(dist_dict["dihedral"])

    return dist

# read in text file
def read_text(filename):
    # initialize data dictionary
    info = []

    # open file
    with open(filename,"r") as fn:
        # read column names
        col_names = fn.readline()

        # read in data
        for line in fn:
            info.append([float(point) for point in line.split()])
        
        # make a numpy 2d array
        info = np.array(info)

        # split off x and y
        x = np.array([info[:,0]])
        y = np.array([info[:,1]])
        z = np.array([info[:,2]])

        # close file
        fn.close()
    
    return x,y,z

# create dist dictionary
def get_dist(vals):
    # read in distributions file
    dist_filename = vals["distributions file"]

    if dist_filename[-5:] == ".dist":
        dist = read_dist_new(dist_filename)
    elif dist_filename[-4:] == ".txt":
        dist = read_dist(dist_filename)

        # remove all but control point, chord, and dihedral
        dist = dist[:,1:7]
        dist = np.delete(dist,4,axis=1)

    # where the y value is negative (ie, left wing)
    # make dihedral negative
    for i in range(dist.shape[0]):
        if dist[i,1] < 0.0:
            dist[i,4] *= -1.
    
    # put units for cpx,cpy,cpz,chord in terms of inches and scale them
    dist[:,:4] *= 12. * vals["scale"]

    # scale specific values in the vals dictionary
    for i in range(len(vals["ducted fan cg [in]"])):
        vals["ducted fan cg [in]"][i] *= vals["scale"]
    for i in range(len(vals["ducted fan diameter, width, and length [in]"])):
        vals["ducted fan diameter, width, and length [in]"][i] *= vals["scale"]
    vals["z bump"]["ducted fan cover thickness [in]"] *= vals["scale"]
    vals["z bump"]["ducted fan cg shift [in]"] *= vals["scale"]
    for i in range(len(vals["z bump"]["span range Behind [in]"])):
        vals["z bump"]["span range Behind [in]"][i] *= vals["scale"]
    for i in range(len(vals["z bump"]["span range Over [in]"])):
        vals["z bump"]["span range Over [in]"][i] *= vals["scale"]
    for i in range(len(vals["z bump"]["span range Front [in]"])):
        vals["z bump"]["span range Front [in]"][i] *= vals["scale"]
    
    # print(vals["z bump"])

    vals["wing tip y start [in]"] *= vals["scale"]
    
    # create dxf file of lines that can be used to define planes
    # determine where the "mid point" lies -- index of last negative y value
    index = np.max(np.argwhere(dist[:,1] < 0.0)[:,0])
    
    # input a distribution for 0 if not there
    if not dist[index+1,1] == 0.0:
        # create cubic interpolations of each unit in dist as function of y (span)
        cubic = interp.interp1d
        x = []; y = []; z = []; c = []; d = []
        for i in range(dist.shape[0]):
            x.append(dist[i,0])
            y.append(dist[i,1])
            z.append(dist[i,2])
            c.append(dist[i,3])
            d.append(dist[i,4])
        x_y = cubic(y,x,kind="cubic")
        z_y = cubic(y,z,kind="cubic")
        c_y = cubic(y,c,kind="cubic")
        d_y = cubic(y,d,kind="cubic")
        dist = np.insert(dist,index+1,np.array([x_y(0.0),0.0,z_y(0.0),\
            c_y(0.0),0.0]),axis=0)

    # set start from index
    start = index + 1
    
    # create dimension specifier
    if vals["dxf"]["2D"]:
        dim = "_2D"
    else:
        dim = "_3D"
    
    # create a folder in the desired directory for output files
    folder = vals["dxf"]["file path"]+vals["dxf"]["folder name"]+dim+"/"
    # if not os.path.isdir(folder[:-1]):
    #     os.mkdir(folder[:-1])

    # save folder path
    vals["C18"]["dxf file path"] = folder

    return dist,start,folder

# translate guides
def translate_guides(guides,dist,i):
    # translate C18 guides
    for q in range(guides.shape[0]):
        # mirror across x axis
        guides[q][:,0] *= -1

        # flip iy and iz
        ia = guides[q][:,1] * -1.; guides[q][:,1] = guides[q][:,2] * 1.
        guides[q][:,2] = ia * 1.

        # rotate by the dihedral
        ry = guides[q][:,1]*np.cos(dist[i,4]) + \
            guides[q][:,2]*np.sin(dist[i,4])
        rz = guides[q][:,2]*np.cos(dist[i,4]) - \
            guides[q][:,1]*np.sin(dist[i,4])
        guides[q][:,2] = rz; guides[q][:,1] = ry

        # shift to the c/4 point
        guides[q][:,0] += dist[i,0]
        guides[q][:,1] += dist[i,1]
        guides[q][:,2] += dist[i,2]

    return guides

# determine x shift value
def get_x_shift(i,dist):
    # determine slope of plane
    m = np.cos(dist[i,4]) / - np.sin(dist[i,4])

    # read in y and z values
    yval = dist[i,1]
    zval = dist[i,2]

    # determine intercept and diagonal values
    b = zval - m * yval
    d = np.abs(b) * np.cos(dist[i,4])

    # calculate normal point values ( of origin)
    vval = d * np.sin(np.abs(dist[i,4]))
    wval = b + d * np.cos(np.abs(dist[i,4]))

    # calculate x shift
    xshift = ((vval-yval)**2. + (wval-zval)**2.)**0.5

    return xshift

# transform the 2D x y arrays for the 2D dxf file
def transform_2D(x,y,dist,i,index,out_arrays=False):
    ### determine 2D values of the shape
    # mirror airfoil across y axis
    x2d = -1. * x

    # rotate by 90 degrees around origin
    rx2d = y*-1.
    ry2d = x2d*1.
    x2d = rx2d; y2d = ry2d

    # determine how much to shift down due to dihedral
    # shift back x and y
    if not dist[i,4] == 0.0:
        # solve for shift value
        xshift  = get_x_shift(i,dist)
        xsshift = get_x_shift(index,dist)

        # add x shift and y shift to proper arrays
        if out_arrays:
            x2d += xsshift
            y2d += dist[index,0]
        else:
            x2d  += xshift
            y2d  += dist[i,0]

    return x2d,y2d

# transform the 3D x y z arrays for the 3D dxf file and plotting
def transform_3D(x,y,z,dist,i,index,out_arrays=False):
    ### determine 3D values of the shape
    # mirror across x axis
    x3d = -1. * x
    
    # flip iy and iz
    a3d = y * -1.; y3d = z * 1.; z3d = a3d * 1.

    # rotate by the dihedral
    ry3d = y3d*np.cos(dist[i,4]) + z3d*np.sin(dist[i,4])
    rz3d = z3d*np.cos(dist[i,4]) - y3d*np.sin(dist[i,4])
    z3d = rz3d; y3d = ry3d

    # shift to the c/4 point
    if out_arrays:
        x3d += dist[index,0]
        y3d += dist[index,1]
        z3d += dist[index,2]
    else:
        x3d += dist[i,0]
        y3d += dist[i,1]
        z3d += dist[i,2]
    
    return x3d,y3d,z3d

# make cubic interpolations for later use
def make_interpolations(dist,vals):
    # create cubic interpolations of each unit in dist as function of y (span)
    cubic = interp.interp1d
    x = []; y = []; z = []; c = []; d = []
    for i in range(dist.shape[0]):
        x.append(dist[i,0])
        y.append(dist[i,1])
        z.append(dist[i,2])
        c.append(dist[i,3])
        d.append(dist[i,4])
    vals["x(y)"] = cubic(y,x,kind="cubic")
    vals["z(y)"] = cubic(y,z,kind="cubic")
    vals["c(y)"] = cubic(y,c,kind="cubic")
    vals["d(y)"] = cubic(y,d,kind="cubic")

    # initialize lengths var list, ismorphing bool list, and section anme list
    b = [0.0]
    ismorphing = [True]
    name = ["bay"]
    spec = ["bay"]

    # determine lengths
    # determine where the fan portion starts
    b.append(vals["ducted fan cg [in]"][1]-\
        vals["ducted fan diameter, width, and length [in]"][1]/2.)
    ismorphing.append(False); name.append("fan"); spec.append("fan")
    # determine where the fan portion ends
    b.append(vals["ducted fan cg [in]"][1]+\
        vals["ducted fan diameter, width, and length [in]"][1]/2.)
    ismorphing.append(True); name.append("servo00"); spec.append("sv0")

    # determine the servo lengths
    n = vals["actuators per semispan"]
    for i in range(1,n):
        b.append((vals["wing tip y start [in]"] - b[2])/n*i + b[2])
        ismorphing.append(True)
        name.append("servo{}".format(str(i).zfill(2)))
        spec.append("sv{}".format(str(i).zfill(1)))
    
    # add the last two points, the start and end of the wing tips
    b.append(vals["wing tip y start [in]"])
    ismorphing.append(False); name.append("wingtip"); spec.append("wtp")
    b.append(dist[-1,1])
    # save to vals
    b = np.array(b)
    
    # for j in range(b.shape[0]-1):
    #     print("{:<8} {:>5.2f} {:>5.2f} {:>5.2f}".format(name[j],b[j],b[j+1],b[j+1]-b[j]))
    print("chord at sv[-1] tip is {}".format(vals["c(y)"](b[-2])))
    # initialize important things
    vals["b"] = b
    vals["is morphing"] = ismorphing
    vals["name"] = name
    vals["spec"] = spec

    # create array of kinks start and end values
    # determine kink start and end percentages
    start = vals["C18"]["flex start [x/c]"]
    end =   vals["C18"]["TE wall start [x/c]"]

    # determine start and end x values 
    x0 = vals["x(y)"](b) - (start-0.25) * vals["c(y)"](b)
    x1 = vals["x(y)"](b) - (end-0.25) * vals["c(y)"](b)

    # initialize kink0 and kink1 lists
    vals["kink0"] = []; vals["kink1"] = []


    # run through each group and determine the min x0 value and max x1
    for i in range(x0.shape[0]-1):
        vals["kink0"].append(min(x0[i],x0[i+1]))
        vals["kink1"].append(max(x1[i],x1[i+1]))
    
    # turn into numpy arrays
    vals["kink0"] = np.array(vals["kink0"])
    vals["kink1"] = np.array(vals["kink1"])
    skin = vals["C18"]["shell thickness [mm]"] / 25.4
    n_kinks = vals["C18"]["num kinks"]

    # create bool array of whether a straightened kink could be made
    greater = vals["kink0"] > vals["kink1"]
    skinfit = (vals["kink0"] - vals["kink1"]) > skin * n_kinks
    vals["can straighten"] = np.logical_and(greater,skinfit)

    return

# create 3d, 2d, and guidecurves
def make_shapes(dist,start,vals):
    # # report index after which the airframe can be airfoils
    # i_wingtip = np.argwhere(dist[:,1] >= vals["wing tip y start [in]"])[0][0]
    # i_wingtip -= start
    # print("Airfoils allowed at dist file shape index {}".format(i_wingtip))
    print()

    # save former values
    vals["c0"] = vals["C18"]["chord [in]"]
    vals["t0"] = vals["C18"]["tongue start [in]"]
    vals["m0"] = vals["C18"]["mouth start [in]"]

    # set "max" vals
    c_ymax = vals["c(y)"](vals["wing tip y start [in]"])
    vals["C18"]["chord [in]"] = c_ymax
    vals["C18"]["tongue start [in]"] = c_ymax * vals["t0"]/vals["c0"]
    vals["C18"]["mouth start [in]"]  = c_ymax * vals["m0"]/vals["c0"]

    try:
        C18.main(vals)
    except:
        raise ValueError("C18 geometry cannot be created at wing tip y start"\
            + "with c={0}, t={1}, and m={2}".format(vals["C18"]["chord [in]"],\
                vals["C18"]["tongue start [in]"],\
                    vals["C18"]["mouth start [in]"]))


    vals["start"] = start
    # report
    print("Creating {} shape...".format(vals["distributions file"].split(".")\
        [0]))
    print()

    # initialize counter index
    j = 0
    first_airfoil = True

    # create airfoil shape
    add = {}
    add["geometry"] = {}
    add["geometry"]["type"] = "outline_points"
    add["geometry"]["outline_points"] = \
        vals["C18"]["airfoil dat file location"]

    # initialize airfoil database
    foil = adb.Airfoil("foil",add)
    coords = foil.get_outline_points()
    x = coords[:,0]; y = coords[:,1]
    # ensure the airfoil is closed
    x0 = x[0]; x1 = x[-1]
    if not x0 == x1:
        if x0 > x1:
            x = np.append(x,x[0])
            y = np.append(y,y[0])
        else:
            x = np.insert(x,0,x[-1])
            y = np.insert(y,0,y[-1])
    # determine camberline values
    x_cam = 0.25; y_cam = 0#foil.get_camber(x_cam)

    # Dallin airfoil shape
    vals["x zb-af"] = x - x_cam
    vals["y zb-af"] = y - y_cam

    # save to airfoil shape
    x = np.array([x[:100],x[99:]]) - x_cam
    y = np.array([y[:100],y[99:]]) - y_cam
    # save to vals
    vals["x af"] = x
    vals["y af"] = y

    # initialize pointer list
    C18_guides = []; af_guides = []; inn_guides = []
    ax = []; ay = []; az = []; afxs = []; afys = []; afzs = []

    vals["ton tip"] = []

    # initialize part counter
    part = 0

    # create a dxf file at each plane for the morphing shape
    for i in range(start,dist.shape[0]): 
        # initilize info dictionary to calculate C18
        # set values
        vals["C18"]["chord [in]"] = dist[i,3]
        vals["C18"]["tongue start [in]"] = dist[i,3] * vals["t0"]/vals["c0"]
        vals["C18"]["mouth start [in]"]  = dist[i,3] * vals["m0"]/vals["c0"]

        # determine start and end vals
        if dist[i,1] > vals["b"][part+1]:
            part += 1
        
        # change kink start and end values
        if vals["straighten kinks"] and vals["can straighten"][part] and part == 0:
            skink = (dist[i,0] - vals["kink0"][part]) / dist[i,3] + 0.25
            ekink = (dist[i,0] - vals["kink1"][part]) / dist[i,3] + 0.25
            vals["C18"]["kinks start [x/c]"] = skink
            vals["C18"]["kinks end [x/c]"] = ekink
            vals["C18"]["straighten TE"] = True
        else:
            if "kinks start [x/c]" in vals["C18"]:
                vals["C18"].pop("kinks start [x/c]",None)
            if "kinks end [x/c]" in vals["C18"]:
                vals["C18"].pop("kinks end [x/c]",None)
        # print(vals["C18"]["chord [in]"],vals["C18"]["tongue start [in]"],\
        #      vals["C18"]["mouth start [in]"])
        # C18.main(vals)
        try:
            ix,iy,iz,isx,isy,isz,guides,kinksgc = C18.main(vals)
        except:
            # initialize x y z arrays
            ix = x * dist[i,3] # 
            iy = y * dist[i,3] # 
            iz = iy * 0.

            # if this is the first failed airfoil, make an airfoil of the last 
            if first_airfoil:
                isx = x * dist[i-1,3] # 
                isy = y * dist[i-1,3] # 
                isz = iy * 0.
                first_airfoil = False
            else: # otherwise, make them zeros.
                isx = 0.; isy = 0.; isz = 0.

            # run through each line and spit out how many points is in the file
            num_points = 0
            for n in range(ix.shape[0]):
                num_points += ix[n].shape[0]
            # report file study
            print("Airfoil c = {:>7.4f} [in] with {:>4d} points {:>2d}/{}".\
                format(dist[i,3],num_points,j,dist.shape[0]-start-1))
            extra = " airfoil"; out = " airfoil"
        else:
            # run through each line and spit out how many points is in the file
            num_points = 0
            for n in range(ix.shape[0]):
                num_points += ix[n].shape[0]
            # report file study
            print("C18 geo c = {:>7.4f} [in] with {:>4d} points {:>2d}/{}".\
                format(dist[i,3],num_points,j,dist.shape[0]-start-1))
            extra = ""; out = " out"

            # translate guides
            guides = translate_guides(guides,dist,i)
            kinksgc["inn gc"] = np.array([kinksgc["inn gc"]]) # for future use
            kinksgc["inn gc"]  = translate_guides( kinksgc["inn gc"],dist,i)[0]
            
            # add to list
            C18_guides.append(guides)
            inn_guides.append(kinksgc["inn gc"])
            vals["ton tip"].append(kinksgc["ton tip"])

        # arfoil coords
        afx = x * dist[i,3] # 
        afy = y * dist[i,3] # 
        afz = y * 0.

        # initialize guide curve
        af_guide = np.zeros((2,3))
        imax = 0
        imin = 0
        af_guide[0,0] = afx[0][imax]; af_guide[1,0] = afx[1][imin]
        af_guide[0,1] = afy[0][imax]; af_guide[1,1] = afy[1][imin]

        # translate airfoil guides
        af_guide = np.array([af_guide])
        translate_guides(af_guide,dist,i)
        af_guide = af_guide[0]
        
        # add to list
        af_guides.append(af_guide)
        
        # save index for future use
        index = i
        if not first_airfoil:
            index = i-1
                
        ### determine 3D values of the shape
        ix3d,iy3d,iz3d    = transform_3D( ix, iy, iz,dist,i,index)
        afx3d,afy3d,afz3d = transform_3D(afx,afy,afz,dist,i,index)

        # append original shape for plotting
        ax.append(ix3d); ay.append(iy3d); az.append(iz3d)
        afxs.append(afx3d); afys.append(afy3d); afzs.append(afz3d)

        # increase count
        j += 1
    
    # turn guides array into numpy array
    C18_guides = np.array(C18_guides)
    inn_guides = np.array(inn_guides)
    af_guides  = np.array(af_guides)

    # turn ton tip into interpolations
    vals["ton tip"] = np.array(vals["ton tip"])
    end = start + vals["ton tip"].shape[0]
    dar = []; fx = []; fy = []; bx = []; by = []; j = 0
    for i in range(start,end):
        dar.append(dist[i,1])
        fx.append(vals["ton tip"][j,0,0])
        fy.append(vals["ton tip"][j,0,1])
        bx.append(vals["ton tip"][j,1,0])
        by.append(vals["ton tip"][j,1,1])
        j += 1
    vals["ton tip x(y)"] = interp.interp1d(dar,fx,kind="cubic")
    vals["ton tip y(y)"] = interp.interp1d(dar,fy,kind="cubic")
    vals["ton mid x(y)"] = interp.interp1d(dar,bx,kind="cubic")
    vals["ton mid y(y)"] = interp.interp1d(dar,by,kind="cubic")

    # store for future use
    apts = [ax,ay,az]
    afpts = [afxs,afys,afzs]

    print()

    return apts,afpts,C18_guides,af_guides,inn_guides

# create C18 guidecurves dxf file
def C18_guides_dxf(C18_guides,vals,folder,specifier):
    # NOTE: 
    # i -- plane #
    # j -- hole #
    # k -- guide curve
    # l -- x, y, or z value
    
    # run through each "hole" and make guide curves
    m = 0
    for j in range(C18_guides[0].shape[0]):
        # initialize extruder guide curves
        x_dxf = np.zeros((C18_guides[0][j].shape[0],),dtype=np.ndarray)
        y_dxf = np.zeros((C18_guides[0][j].shape[0],),dtype=np.ndarray)
        z_dxf = np.zeros((C18_guides[0][j].shape[0],),dtype=np.ndarray)

        # run through each guide curve
        for k in range(C18_guides[0][j].shape[0]): # 1): # 
            # initialize plane lists
            x_list = []; y_list = []; z_list = []

            # run through each plane
            for i in range(C18_guides.shape[0]):
                x_list.append(C18_guides[i][j][k][0])
                y_list.append(C18_guides[i][j][k][1])
                z_list.append(C18_guides[i][j][k][2])

            # add to dxf as numpy arrays
            x_dxf[k] = np.array(x_list)
            y_dxf[k] = np.array(y_list)
            z_dxf[k] = np.array(z_list)
        

        # save as a dxf file
        if m == 0:
            step = "00_"
        else:
            step = "OP_"
        file_location = folder + step + specifier + "_mn"+ str(m).zfill(2)\
             + "_GC"
        dxf.dxf(file_location,x_dxf,y_dxf,z_dxf,geometry="spline")
        m += 1
    
    return

# create kink guidecurves dxf file
def kink_guides_dxf(kink_guides,vals,folder,specifier):
    # NOTE: 
    # i -- plane #
    # j -- hole #
    # k -- guide curve
    # l -- x, y, or z value
    
    # determine how many kinks guide curves to make
    num = kink_guides[0].shape[0] * kink_guides[0][0].shape[0]

    # initialize extruder guide curves
    x_dxf = np.zeros((num,),dtype=np.ndarray)
    y_dxf = np.zeros((num,),dtype=np.ndarray)
    z_dxf = np.zeros((num,),dtype=np.ndarray)
    
    # run through each kink and make guide curves
    m = 0
    for j in range(kink_guides[0].shape[0]):
        # run through each guide curve
        for k in range(kink_guides[0][j].shape[0]): # 1): # 
            # initialize plane lists
            x_list = []; y_list = []; z_list = []

            # run through each plane
            for i in range(kink_guides.shape[0]):
                x_list.append(kink_guides[i][j][k][0])
                y_list.append(kink_guides[i][j][k][1])
                z_list.append(kink_guides[i][j][k][2])

            # add to dxf as numpy arrays
            x_dxf[m] = np.array(x_list)
            y_dxf[m] = np.array(y_list)
            z_dxf[m] = np.array(z_list)
            m += 1

    # save as a dxf file
    file_location = folder + "04_" + specifier + "_kink_GC"
    dxf.dxf(file_location,x_dxf,y_dxf,z_dxf,geometry="spline")
    
    return

# create airfoil wing tip guidecurves dxf file
def af_guides_dxf(af_guides,vals,folder,specifier,typ="main"):
    # NOTE: 
    # i -- plane #
    # k -- guide curve
    # l -- x, y, or z value
    
    # initialize extruder guide curves
    x_dxf = np.zeros((af_guides[0].shape[0],),dtype=np.ndarray)
    y_dxf = np.zeros((af_guides[0].shape[0],),dtype=np.ndarray)
    z_dxf = np.zeros((af_guides[0].shape[0],),dtype=np.ndarray)

    # run through each guide curve
    for k in range(af_guides[0].shape[0]): # 1): # 
        # initialize plane lists
        x_list = []; y_list = []; z_list = []

        # run through each plane
        for i in range(af_guides.shape[0]):
            x_list.append(af_guides[i][k][0])
            y_list.append(af_guides[i][k][1])
            z_list.append(af_guides[i][k][2])

        # add to dxf as numpy arrays
        x_dxf[k] = np.array(x_list)
        y_dxf[k] = np.array(y_list)
        z_dxf[k] = np.array(z_list)

    # save as a dxf file
    if typ == "KIhl":
        step = "01_"
    elif typ == "LEhl":
        step = "02_"
    else:
        step = "00_"
    file_location = folder + step + specifier + "_" + typ + "_GC"
    dxf.dxf(file_location,x_dxf,y_dxf,z_dxf,geometry="spline")

    return

# create cylinders to help guide pieces together
def make_cylinders(vals,chord=12,is_hole = True,is_bay = False):
    # initialize x y z arrays
    num = 2 + 2 * is_bay
    x = np.zeros((num,),dtype=np.ndarray)
    y = np.zeros((num,),dtype=np.ndarray)
    z = np.zeros((num,),dtype=np.ndarray)

    # initialize is_peg bool
    is_peg = not is_hole

    # initialize circle radius # 0.3" on 12" chord
    rad = 0.3 / 12 + (0.0/25.4*is_hole - 0.5/25.4*is_peg) / chord

    # initialize theta array
    theta = np.linspace(0,2*np.pi,num=vals["C18"]["num kink points"])

    # initialize radius array
    r = np.full(theta.shape,rad)

    # run through each and create a circle
    x_center = 0; y_center = 0
    for i in range(2):
        # initialize circle center
        x_center += vals["C18"]["flex start [x/c]"] / 3.

        # save arrays
        x[i] = r * np.cos(theta) + x_center
        y[i] = r * np.sin(theta) + y_center
        z[i] = r * 0.

        # save arrays if bay
        if is_bay:
            x[i+2] = (r+1.0/25.4/chord) * np.cos(theta) + x_center
            y[i+2] = (r+1.0/25.4/chord) * np.sin(theta) + y_center
            z[i+2] = (r+1.0/25.4/chord) * 0.

    # shift back by quarter chord
    x -= 0.25

    # resize by chord
    x *= chord; y *= chord; z *= chord

    return x,y,z

# determine start and end shapes for each part of Horizon
def make_parts(dist,vals,airframe_guidecurves):
    # draw out airframe guidecurves
    C18_afm,af_afm,inn_afm = airframe_guidecurves

    # initialize important things
    b = vals["b"]
    ismorphing = vals["is morphing"]
    name = vals["name"]
    spec = vals["spec"]

    # create new dist array which can have all the b values interpolated
    bist = np.zeros((b.shape[0],5))
    bist[:,0] = vals["x(y)"](b)
    bist[:,1] = b
    bist[:,2] = vals["z(y)"](b)
    bist[:,3] = vals["c(y)"](b)
    bist[:,4] = vals["d(y)"](b)

    # run through each b value and determine the index just before in the dist
    # file
    dist_inds = []
    for val in b:
        inds = np.argwhere(val <= dist[:,1])[0][0]
        dist_inds.append(inds-vals["start"])
    dist_inds = np.array(dist_inds)

    # initialize to plot lists
    xplt = []; yplt = []; zplt = []

    # create dxf files for each section
    for i in range(len(name)):
        # report
        print("Creating {:<7} files".format(name[i]))
        
        # ROOT

        if ismorphing[i]:
            # set values
            vals["C18"]["chord [in]"] = bist[i,3]
            vals["C18"]["tongue start [in]"] = bist[i,3] * \
                vals["t0"]/vals["c0"]
            vals["C18"]["mouth start [in]"]  = bist[i,3] * \
                vals["m0"]/vals["c0"]

            # change kink start and end values
            if vals["straighten kinks"] and vals["can straighten"][i] and \
                name[i] == "bay":
                skink = (bist[i,0] - vals["kink0"][i]) / bist[i,3] + 0.25
                ekink = (bist[i,0] - vals["kink1"][i]) / bist[i,3] + 0.25
                vals["C18"]["kinks start [x/c]"] = skink
                vals["C18"]["kinks end [x/c]"] = ekink
                vals["C18"]["straighten TE"] = True
            else:
                skink = vals["C18"]["flex start [x/c]"]
                ekink = vals["C18"]["TE wall start [x/c]"]
                vals["C18"]["kinks start [x/c]"] = skink
                vals["C18"]["kinks end [x/c]"] = ekink

            # calculate C18 shape values
            ix,iy,iz,isx,isy,isz,guides_rt,kinksgc_rt = C18.main(vals)

            # translate guide curve shapes
            guides_rt = translate_guides(guides_rt,bist,i)
            kinksgc_rt["inn gc"] = np.array([kinksgc_rt["inn gc"]])
            kinksgc_rt["inn gc"] = translate_guides(kinksgc_rt["inn gc"],\
                bist,i)[0]

            # initialize inx and iny and inz
            inx = kinksgc_rt["x inner"]; iny = kinksgc_rt["y inner"];inz = 0.*iny
        else:
            # initialize x y z arrays
            ix = vals["x af"] * bist[i,3] # 
            iy = vals["y af"] * bist[i,3] # 
            iz = iy * 0.
            
            # initialize guide curve
            af_guide_rt = np.zeros((2,3))
            imax = 0
            imin = 0
            af_guide_rt[0,0] = ix[0][imax]; af_guide_rt[1,0] = ix[1][imin]
            af_guide_rt[0,1] = iy[0][imax]; af_guide_rt[1,1] = iy[1][imin]

            # translate airfoil guide_rts
            af_guide_rt = np.array([af_guide_rt])
            translate_guides(af_guide_rt,bist,i)
            af_guide_rt = af_guide_rt[0]
        
        # initialize cyx and cyy and cyz
        cyx,cyy,cyz = make_cylinders(vals,bist[i,3],is_hole=True,is_bay=i==0) 
        if name[i] == "bay":
            pyx,pyy,pyz = make_cylinders(vals,bist[i,3],is_hole=False) 

        
        # manipulate arrays based on whether 2D or 3D desired
        ix3d,iy3d,iz3d = transform_3D(ix,iy,iz,bist,i,i)
        xplt.append(ix3d); yplt.append(iy3d); zplt.append(iz3d)
        if vals["dxf"]["2D"]:
            ix,iy = transform_2D(ix,iy,bist,i,i)
            cyx,cyy = transform_2D(cyx,cyy,bist,i,i)
            if ismorphing[i]:
                isx,isy = transform_2D(isx,isy,bist,i,i)
                inx,iny = transform_2D(inx,iny,bist,i,i)
                if name[i] == "bay":
                    pyx,pyy = transform_2D(pyx,pyy,bist,i,i)
        else:
            ix = ix3d; iy = iy3d; iz = iz3d
            cyx,cyy,cyz = transform_3D(cyx,cyy,cyz,dist,i,vals)
            if ismorphing[i]:
                isx,isy,isz = transform_3D(isx,isy,isz,bist,i,i)
                inx,iny,inz = transform_3D(inx,iny,inz,bist,i,i)
                if name[i] == "bay":
                    pyx,pyy,pyz = transform_3D(pyx,pyy,pyz,bist,i,i)
        
        # create folder for this part
        folder = vals["C18"]["dxf file path"]+str(i).zfill(2)+"_"+name[i]+"/"
        
        # specify dxf file type
        if ismorphing[i]:
            typ = "C18 "
        else:
            typ = "Airfoil "
        airfoil_type = vals["C18"]["airfoil dat file location"].split("/")[-1]\
            .split(".")[0]
        typ = airfoil_type + " " + typ

        # arrange file info for dxf file
        file_info = [
            typ + "geometry with chord length {:>7.4f} [in]".format(bist[i,2]),
            "control point at ({:>7.4f},{:>7.4f},{:>7.4f}) [in]".format(\
                bist[i,0],bist[i,1],bist[i,2])
        ]
        
        # create dxf file
        if ismorphing[i]:
            step_gm = "OP_"
            step_pg = "02_"
        else:
            step_gm = "00_"
            step_pg = "01_"
        # main geometry file
        dxf.dxf(folder+step_gm+spec[i]+"_main_"+str(i).zfill(2),ix,iy,iz,\
            file_info=file_info)
        # pegs file
        dxf.dxf(folder+step_pg+spec[i]+"_pegs_"+str(i).zfill(2),cyx,cyy,cyz,\
            file_info=file_info)
        if ismorphing[i]:
            # outer geometry file
            dxf.dxf(folder+"00_"+spec[i]+"_oute_"+str(i).zfill(2),isx,isy,isz,\
                file_info=file_info)
            # kinkS hole
            dxf.dxf(folder+"01_"+spec[i]+"_KIhl_"+str(i).zfill(2),inx,iny,inz,\
                file_info=file_info)
            if name[i] == "bay":
                # pegs file
                dxf.dxf(folder+step_pg+"cen_pegs_"+str(i).zfill(2),pyx,pyy,\
                    pyz,file_info=file_info)

            


        # TIP

        if ismorphing[i]:
            # set values
            vals["C18"]["chord [in]"] = bist[i+1,3]
            vals["C18"]["tongue start [in]"] = bist[i+1,3] * \
                vals["t0"]/vals["c0"]
            vals["C18"]["mouth start [in]"]  = bist[i+1,3] * \
                vals["m0"]/vals["c0"]

            # change kink start and end values
            if vals["straighten kinks"] and vals["can straighten"][i] and\
                name[i] == "bay":
                skink = (bist[i+1,0] - vals["kink0"][i]) / bist[i+1,3] + 0.25
                ekink = (bist[i+1,0] - vals["kink1"][i]) / bist[i+1,3] + 0.25
                vals["C18"]["kinks start [x/c]"] = skink
                vals["C18"]["kinks end [x/c]"] = ekink
                vals["C18"]["straighten TE"] = True
            else:
                skink = vals["C18"]["flex start [x/c]"]
                ekink = vals["C18"]["TE wall start [x/c]"]
                vals["C18"]["kinks start [x/c]"] = skink
                vals["C18"]["kinks end [x/c]"] = ekink

            # calculate C18 shape values
            ix,iy,iz,isx,isy,isz,guides_tp,kinksgc_tp = C18.main(vals)

            # translate guide curve shapes
            guides_tp = translate_guides(guides_tp,bist,i+1)
            kinksgc_tp["inn gc"] = np.array([kinksgc_tp["inn gc"]])
            kinksgc_tp["inn gc"]  = translate_guides(kinksgc_tp["inn gc"]\
                ,bist,i+1)[0]

            # initialize inx and iny and inz
            inx = kinksgc_tp["x inner"]; iny = kinksgc_tp["y inner"]
            inz = 0.*iny
        else:
            # initialize x y z arrays
            ix = vals["x af"] * bist[i+1,3] # 
            iy = vals["y af"] * bist[i+1,3] # 
            iz = iy * 0.
            
            # initialize guide curve
            af_guide_tp = np.zeros((2,3))
            imax = 0
            imin = 0
            af_guide_tp[0,0] = ix[0][imax]; af_guide_tp[1,0] = ix[1][imin]
            af_guide_tp[0,1] = iy[0][imax]; af_guide_tp[1,1] = iy[1][imin]

            # translate airfoil guide_tps
            af_guide_tp = np.array([af_guide_tp])
            translate_guides(af_guide_tp,bist,i+1)
            af_guide_tp = af_guide_tp[0]
        
        
        # initialize cyx and cyy and cyz
        cyx,cyy,cyz = make_cylinders(vals,bist[i+1,3],is_hole=False) 
        
        # manipulate arrays based on whether 2D or 3D desired
        ix3d,iy3d,iz3d = transform_3D(ix,iy,iz,bist,i+1,i+1)
        if vals["dxf"]["2D"]:
            ix,iy = transform_2D(ix,iy,bist,i+1,i+1)
            cyx,cyy = transform_2D(cyx,cyy,bist,i+1,i+1)
            if ismorphing[i]:
                isx,isy = transform_2D(isx,isy,bist,i+1,i+1)
                inx,iny = transform_2D(inx,iny,bist,i+1,i+1)
        else:
            ix = ix3d; iy = iy3d; iz = iz3d
            cyx,cyy,cyz = transform_3D(cyx,cyy,cyz,bist,i+1,i+1)
            if ismorphing[i]:
                isx,isy,isz = transform_3D(isx,isy,isz,bist,i+1,i+1)
                inx,iny,inz = transform_3D(inx,iny,inz,bist,i+1,i+1)

        # arrange file info for dxf file
        file_info = [
            typ + "geometry with chord length {:>7.4f} [in]".format(bist[i+1,3]),
            "control point at ({:>7.4f},{:>7.4f},{:>7.4f}) [in]".format(\
                bist[i+1,0],bist[i+1,1],bist[i+1,2])
        ]
        
        # create dxf file
        dxf.dxf(folder+step_gm+spec[i]+"_main_"+str(i+1).zfill(2),ix,iy,iz,\
            file_info=file_info)
        dxf.dxf(folder+step_pg+spec[i]+"_pegs_"+str(i+1).zfill(2),cyx,cyy,cyz,\
            file_info=file_info)
        if ismorphing[i]:
            dxf.dxf(folder+"00_"+spec[i]+"_oute_"+str(i+1).zfill(2),isx,\
                isy,isz,file_info=file_info)
            dxf.dxf(folder+"01_"+spec[i]+"_KIhl_"+str(i+1).zfill(2),inx,iny,\
                inz,file_info=file_info)

        # GUIDE CURVES
        # determine start and end indices
        start_i = dist_inds[i]
        if i == 0:
            start_i += 1
        end_i   = dist_inds[i+1]

        if ismorphing[i]:
            # initialize guide curves lists
            C18_gc = [guides_rt]
            inn_gc = [kinksgc_rt["inn gc"]]

            # run through each and add the airframe into them
            for j in range(start_i,end_i):
                C18_gc.append(C18_afm[j])
                inn_gc.append(inn_afm[j])

            # append the final part
            C18_gc.append(guides_tp)
            inn_gc.append(kinksgc_tp["inn gc"])

            # turn into numpy arrays
            C18_gc = np.array(C18_gc)
            inn_gc = np.array(inn_gc)

            # send to dxf maker
            C18_guides_dxf(C18_gc,vals,folder,spec[i])
            af_guides_dxf(inn_gc,vals,folder,spec[i],typ="KIhl")
        else:
            # initialize guide curves lists
            af_gc = [af_guide_rt]

            # run through each and add the airframe into them
            for j in range(start_i,end_i):
                af_gc.append(af_afm[j])

            # append the final part
            af_gc.append(af_guide_tp)

            # turn into numpy arrays
            af_gc = np.array(af_gc)

            # send to dxf maker
            af_guides_dxf(af_gc,vals,folder,spec[i])

    # add last 3d shapes
    xplt.append(ix3d); yplt.append(iy3d); zplt.append(iz3d)

    # turn into numpy arrays
    xplt = np.array(xplt); yplt = np.array(yplt); zplt = np.array(zplt)
    pltxyz = [xplt,yplt,zplt]

    print()
    return pltxyz,bist

# create nomenclature specifying file
def make_nomenclature(folder):
    filename = folder + "README.txt"
    FILE = """FILE NOMENCLATURE

    system  : **_***_****_**
    purpose : (1)_(2)_(3)_(4)
    example : 00_bay_oute_00

    (1) Step number
    This number indicates the step this file corresponds to
    NOTE :: All number is performed with zero based indexing
    IE : 00 is first, 01 next, then 02, and so on.
         OP indicates optional

    (2) Part name
    The part this file will be used to create
    IE : Base for Base file, bay for bay section, fan section,
         servo 0 - ? section, and wingtip section

    (3) Descriptor
    file descriptor, mainly for the author's benefit, described as follows
    accs - 2D file for access holes to be used to access hubs after printing
    KIhl - 2D file for kink hole shape
    hubs - 2D file for actuation hubs
    main - 2D file for main morphing/airfoil shape
    mn## - guidecurves file for main shape
    pegs - 2D file for pegs to be extruded and used to connect sections
    sect - 2D file for lines used to create section planes (see (4), ##)
    tTip - 3D file for tongue tip 3D lines, used in modeling
    oute - 2D file for outer shape of main morphing shape


    (4) Plane to be inserted on
    ##     - plane number created by user from lines in 00_Base_sect_RPlane.dxf
    A~     - plane number created by user from lines in 00_Base_actu_RPlane.dxf
             Examples: Aa, Ab, Ac, etc.
    Above  - plane created above airframe body parallel to front plane
    GC     - guidecurves -- INSERT ON ANY PLANE, BUT MUST INSERT AS 3D DXF FILE
    Nose   - plane created 
    RPlane - Right Plane
    TPlane - Top Plane
    """
    
    # create directory if not existant yet
    directory = filename.split("/")[1:]
    path = "/"
    for i in range(len(directory)-1):
        path += directory[i] + "/"
        if not os.path.isdir(path[:-1]):
            os.mkdir(path[:-1])
    
    # write to file
    with open(filename,"w") as f:
        f.write(FILE)
        f.close()

    return

# code to be inserted into 'placer_c14.py'
def dallin_prelim(vals):
    # create z array of y values
    z = np.linspace(vals["b"][1],vals["b"][2],70)

    # create z 2D array in the format of dist
    zist = np.zeros((z.shape[0],5))
    zist[:,0] = vals["x(y)"](z)
    zist[:,1] = z
    zist[:,2] = vals["z(y)"](z)
    zist[:,3] = vals["c(y)"](z)
    zist[:,4] = vals["d(y)"](z)
    
    # create guide curve points for an airfoil at each zist location
    xgc = []
    ygc = []
    zgc = []
    for i in range(zist.shape[0]):
        x_chord = vals["x zb-af"] * zist[i,3]
        y_chord = vals["y zb-af"] * zist[i,3]
        x3d,y3d,z3d = transform_3D(x_chord,y_chord,y_chord*0.,zist,i,i,out_arrays=False)
        xgc.append(x3d)
        ygc.append(y3d)
        zgc.append(z3d)

    # make xgc, ygc, and zgc numpy arrays
    xgc = np.array(xgc)
    ygc = np.array(ygc)
    zgc = np.array(zgc)

    # test transpose
    xgc = np.transpose(xgc)
    ygc = np.transpose(ygc)
    zgc = np.transpose(zgc)

    return xgc,ygc,zgc

# split up 2D shape by indices
def make_2D_split(hpicks,vals,bist,folder,xgc,ygc,zgc):
    # initialize index for bist use
    root_index = 1
    tip_index = 2

    # initialize new airfoil arrays
    afb_x = np.zeros((len(hpicks)-1,),dtype=np.ndarray)
    afb_y = np.zeros((len(hpicks)-1,),dtype=np.ndarray)
    afb_z = np.zeros((len(hpicks)-1,),dtype=np.ndarray)

    # run through each split
    for i in range(afb_x.shape[0]):
        afb_x[i] = vals["x zb-af"][hpicks[i]:hpicks[i+1]+1]
        afb_y[i] = vals["y zb-af"][hpicks[i]:hpicks[i+1]+1]
        afb_z[i] = afb_y[i] * 0.
    
    # determine 2d airfoil shape at root
    rt_x,rt_y = transform_2D(afb_x*bist[root_index,3],afb_y*bist[root_index,3]\
        ,bist,root_index,root_index)
    
    # determine 2d airfoil shape at tip
    tp_x,tp_y = transform_2D(afb_x*bist[tip_index,3],afb_y*bist[tip_index,3],\
        bist,tip_index,tip_index)

    # file info list
    file_info = [
        "Airfoil geometry with chord length {:>7.4f} [in]".format(\
            bist[root_index,2]),
        "control point at ({:>7.4f},{:>7.4f},{:>7.4f}) [in]".format(\
            bist[root_index,0],bist[root_index,1],bist[root_index,2])
    ]

    # save the dxf to the folder
    dxf.dxf(folder+"00_fan_main_01",rt_x,rt_y,rt_y*0.,file_info=file_info)

    # file info list
    file_info = [
        "Airfoil geometry with chord length {:>7.4f} [in]".format(\
            bist[tip_index,2]),
        "control point at ({:>7.4f},{:>7.4f},{:>7.4f}) [in]".format(\
            bist[tip_index,0],bist[tip_index,1],bist[tip_index,2])
    ]

    # save the dxf to the folder
    dxf.dxf(folder+"00_fan_main_02",tp_x,tp_y,tp_y*0.,file_info=file_info)

    # save wing guide curves file
    dxf.dxf(folder+"00_fan_main_GC",xgc,ygc,zgc)
    
    return

# make airfoil and 2d circle and square shapes (shift)
def make_zbump_shapes(x,y,z,angle,indices,folder,name):

    # initialize new airfoil arrays
    new_x = np.zeros((len(indices)-1,),dtype=np.ndarray)
    new_y = np.zeros((len(indices)-1,),dtype=np.ndarray)
    new_z = np.zeros((len(indices)-1,),dtype=np.ndarray)

    # run through each split
    for i in range(new_x.shape[0]):
        new_x[i] = x[indices[i]:indices[i+1]+1]
        new_y[i] = y[indices[i]:indices[i+1]+1]
        new_z[i] = z[indices[i]:indices[i+1]+1]
    
    # for now, save dxf file of 3d shape
    dxf.dxf(folder+"02_fan_" + name + "_GC",new_x,new_y,new_z)

    # determine shape midpoint (control point)
    cpx = np.average(x)
    cpy = np.average(y)
    cpz = np.average(z)

    # shift the shape so midpoint is (0,0,0)
    x -= cpx; y -= cpy; z -= cpz

    # # rotate by the dihedral
    # ry3d = y3d*np.cos(dist[i,4]) + z3d*np.sin(dist[i,4])
    # rz3d = z3d*np.cos(dist[i,4]) - y3d*np.sin(dist[i,4])
    # z3d = rz3d; y3d = ry3d

    # turn into 2d shape
    




    # determine the index of the min y point
    y_min_index = np.argwhere(y==np.min(y))[0]
    y_max_index = np.argwhere(y==np.max(y))[0]

    # save points
    x_plane = np.array([x[y_min_index],x[y_max_index]])
    y_plane = np.array([y[y_min_index],y[y_max_index]])
    z_plane = y_plane * 0.

    return x_plane,y_plane,z_plane

def zbumper(vals,folder,bist):
    # report
    print("creating z bump")

    # get preliminary info
    xgc,ygc,zgc = dallin_prelim(vals)

    # create zbump
    xgc,ygc,zgc,hpicks,stuff = zmorph.main(xgc,ygc,zgc,vals)

    # # undo transpose
    # xgc = np.transpose(xgc)
    # ygc = np.transpose(ygc)
    # zgc = np.transpose(zgc)

    # split out stuff tuple
    bx,by,bz,alpha_b,fx,fy,fz,alpha_f,gc1,gc2,gc3,gc4,behind_i,front_i = stuff

    # split out guide curves
    gc1x,gc1y,gc1z = gc1
    gc2x,gc2y,gc2z = gc2
    gc3x,gc3y,gc3z = gc3
    gc4x,gc4y,gc4z = gc4

    # order hpicks, behind_i and front_i after making them numpy arrays
    hpicks = np.sort(np.array(hpicks))
    behind_i = np.sort(np.array(behind_i))
    front_i = np.sort(np.array(front_i))

    # ADD begin and end indices for use in splitting out 
    # if the first index of hpicks is not 0, add it
    if hpicks[0] != 0:
        hpicks = np.insert(hpicks,0,0)

    # if the final index of hpicks is not the end, add final index to hpicks
    if hpicks[-1] != vals["x zb-af"].shape[0]:
        hpicks = np.append(hpicks,vals["x zb-af"].shape[0])

    # if the first index of behind_i is not 0, add it
    if behind_i[0] != 0:
        behind_i = np.insert(behind_i,0,0)

    # if the final index of behind_i is not the end,add final index to behind_i
    if behind_i[-1] != bx.shape[0]:
        behind_i = np.append(behind_i,bx.shape[0])

    # if the first index of front_i is not 0, add it
    if front_i[0] != 0:
        front_i = np.insert(front_i,0,0)

    # if the final index of front_i is not the end, add final index to front_i
    if front_i[-1] != fx.shape[0]:
        front_i = np.append(front_i,fx.shape[0])

    # make guide curve numpy arrays
    # initialize arrays
    gch_x = np.zeros((4,),dtype=np.ndarray)
    gch_y = np.zeros((4,),dtype=np.ndarray)
    gch_z = np.zeros((4,),dtype=np.ndarray)

    # add pieces
    gch_x[0] = gc1x; gch_x[1] = gc2x; gch_x[2] = gc3x; gch_x[3] = gc4x
    gch_y[0] = gc1y; gch_y[1] = gc2y; gch_y[2] = gc3y; gch_y[3] = gc4y
    gch_z[0] = gc1z; gch_z[1] = gc2z; gch_z[2] = gc3z; gch_z[3] = gc4z

    # create folder for use
    fan_folder = folder + "01_fan/"

    # split up airfoil into the appropriate number of sections
    make_2D_split(hpicks,vals,bist,fan_folder,xgc,ygc,zgc)

    # split up and turn into 2d the front and back shapes
    xpf,ypf,zpf = make_zbump_shapes(fx,fy,fz,alpha_f,front_i,fan_folder,"fron")
    xpb,ypb,zpb =make_zbump_shapes(bx,by,bz,alpha_b,behind_i,fan_folder,"back")

    # combine plane lines and create dxf file
    xp = np.array([xpf,xpb]);yp = np.array([ypf,ypb]);zp = np.array([zpf,zpb])
    dxf.dxf(fan_folder+"02_fan_plns_FP",xp,yp,zp,geometry="line")

    # save hole guide curves file
    dxf.dxf(fan_folder+"02_fan_hole_GC",gch_x,gch_y,gch_z)

    print("MY STUFF!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    return

# create plane lines
def make_planes(bist,vals,folder,name):
    # report
    print("Creating {} file".format(name))

    # initialize arrays for planar lines
    n = bist.shape[0] * 2
    X_plane = np.zeros((n,),dtype=np.ndarray)
    Y_plane = np.zeros((n,),dtype=np.ndarray)
    Z_plane = np.zeros((n,),dtype=np.ndarray)

    # run through each point and create a line to be used to create the planes
    j = 0
    for i in range(bist.shape[0]):
        # initialize line
        X_plane[j] = np.zeros((2,))
        Y_plane[j] = np.zeros((2,))
        Z_plane[j] = np.zeros((2,))

        # set last point as the planar control point
        Y_plane[j][1] = bist[i,1]
        X_plane[j][1] = bist[i,2]

        # set first point as one unit away from the control point
        # at the dihedral angle
        line = vals["plane line length"]
        Y_plane[j][0] = Y_plane[j][1] + np.sin(bist[i,4]) * line
        X_plane[j][0] = X_plane[j][1] - np.cos(bist[i,4]) * line
        
        # initialize line
        X_plane[j+1] = np.zeros((2,))
        Y_plane[j+1] = np.zeros((2,))
        Z_plane[j+1] = np.zeros((2,))

        if not j == 0:
            # set last point as the planar control point
            mn = -1/np.tan(bist[i,4])
            dy = ( (line/20.)**2./(1+mn**2.) )**0.5
            Y_plane[j+1][0] = Y_plane[j][0] - dy*mn
            X_plane[j+1][0] = X_plane[j][0] + dy

            # set first point as one unit away from the control point
            # at the dihedral angle
            Y_plane[j+1][1] = Y_plane[j][0] + dy*mn
            X_plane[j+1][1] = X_plane[j][0] - dy
        else:
            # set last point as the planar control point
            Y_plane[j+1][0] = Y_plane[j][0] + line/20.
            X_plane[j+1][0] = X_plane[j][0]

            # set first point as one unit away from the control point
            # at the dihedral angle
            Y_plane[j+1][1] = Y_plane[j][0] - line/20.
            X_plane[j+1][1] = X_plane[j][0]

        # increase index
        j += 2
    
    # reverse across the x axis
    X_plane *= -1.
    
    # create a dxf file of the lines
    dxf.dxf(folder+name,X_plane,Y_plane,Z_plane,geometry="line")

    return

# create lines for hubs planes
def make_hubs_planes(dist,folder,name):
    # report
    print("Creating HubsPlanes file")

    # find min z value, max x value
    zmin = np.min(dist[:,2])
    xmax = np.max(dist[:,0] + dist[:,3] * .25)

    # add an inch buffer
    zmin -= 1.
    xmax += 1.

    # create lines
    x = np.array([np.array([xmax,xmax]),np.array([-1,1])])
    z = np.array([np.array([-1,1]),np.array([zmin,zmin])])
    
    # create a dxf file of the lines
    dxf.dxf(folder+name,x,-z,z*0.,geometry="line")

    return

# create actuation hubs
def make_hubs(bist,vals,folder):
    # report
    print("Creating actuation hubs files")
    
    # create a bist-like array of the b values and chord values
    hubs = np.zeros((bist.shape[0]-1,5))

    # determine b values, and subsequent
    hubs[:,1] = (bist[:-1,1] + bist[1:,1])/2.
    # fix first value due to bay
    hubs[0,1] = 0.25
    hubs[:,0] = vals["x(y)"](hubs[:,1])
    hubs[:,2] = vals["z(y)"](hubs[:,1])
    hubs[:,3] = vals["c(y)"](hubs[:,1])
    hubs[:,4] = vals["d(y)"](hubs[:,1])

    # initialize dxf arrays 
    # for actuation hubs
    hub_n = int(5 * hubs.shape[0])
    x_hub = np.zeros((hub_n,),dtype=np.ndarray)
    y_hub = np.zeros((hub_n,),dtype=np.ndarray)
    # access holes
    x_acc = np.zeros((hubs.shape[0],),dtype=np.ndarray)
    y_acc = np.zeros((hubs.shape[0],),dtype=np.ndarray)
    # LE holes
    x_leh = np.zeros((hubs.shape[0],),dtype=np.ndarray)
    y_leh = np.zeros((hubs.shape[0],),dtype=np.ndarray)
    # hub points
    x_hpt = np.zeros((hubs.shape[0]*2,),dtype=np.ndarray)
    y_hpt = np.zeros((hubs.shape[0]*2,),dtype=np.ndarray)
    z_hpt = np.zeros((hubs.shape[0]*2,),dtype=np.ndarray)

    # run through each hub value
    j = 0
    for i in range(hubs.shape[0]):
        # determine the center of the actuation hole
        x_center = (vals["C18"]["flex start [x/c]"] * 2 - \
            vals["t0"] / vals["c0"]) / 2. * hubs[i,3]
        y_center = hubs[i,1]
        
        # remove quarter chord and x loc
        x_center -= vals["C18"]["shift to c/4"]*0.25*hubs[i,3]
        x_center *= -1.
        x_center += hubs[i,0]

        # create two theta arrays, one for a full circle, one for a half
        thcirc = np.linspace(0,2*np.pi,vals["C18"]["num kink points"])
        thsemi = np.linspace(np.pi/2.,3*np.pi/2.,vals["C18"]["num kink points"])

        # create actuation hole radius array
        r = vals["actuation hole diameter [mm]"]/25.4/2.
        rcirc = np.full(thcirc.shape,r)
        rsemi = np.full(thsemi.shape,3.*r)

        # create access hole radius
        racc  = np.full(thcirc.shape,6.666666666667*r) # to make it 14 mm

        # create leading edge hole radius
        rleh  = np.full(thcirc.shape,vals["connector height [mm]"]/25.4)

        # determine the tip of the tongue value
        if hubs[i,1] > vals["wing tip y start [in]"]:
            # tongue tip values
            tonx = hubs[i,0]
            tony = hubs[i,1]
            tonz = hubs[i,2]
            # mid tongue values
            midx = hubs[i,0]
            midy = hubs[i,1]
            midz = hubs[i,2]
        else:
            # tongue tip values
            tonx = -(vals["ton tip x(y)"](hubs[i,1])) + hubs[i,0]
            tony = hubs[i,1]
            tonz = -(vals["ton tip y(y)"](hubs[i,1])) + hubs[i,2]
            # mid tongue values
            midx = -(vals["ton mid x(y)"](hubs[i,1])) + hubs[i,0]
            midy = hubs[i,1]
            midz = -(vals["ton mid y(y)"](hubs[i,1])) + hubs[i,2]

        # create hole for hub
        x_hub[j] = rcirc * np.cos(thcirc) + x_center
        y_hub[j] = rcirc * np.sin(thcirc) + y_center
        j += 1

        # create semicircle
        x_hub[j] = rsemi * np.cos(thsemi) + x_center
        y_hub[j] = rsemi * np.sin(thsemi) + y_center
        j += 1

        # create side lines
        x_hub[j] = np.array([x_hub[j-1][0],tonx])
        y_hub[j] = np.array([y_hub[j-1][0],y_hub[j-1][0]])
        j += 1
        x_hub[j] = np.array([x_hub[j-2][-1],tonx])
        y_hub[j] = np.array([y_hub[j-2][-1],y_hub[j-2][-1]])
        j += 1

        # create end line
        x_hub[j] = np.array([x_hub[j-1][1],x_hub[j-2][1]])
        y_hub[j] = np.array([y_hub[j-1][1],y_hub[j-2][1]])
        j += 1

        # create access hole array
        x_acc[i] = racc * np.cos(thcirc) + x_center
        y_acc[i] = racc * np.sin(thcirc) + y_center

        # create LE hole array
        x_leh[i] = rleh * np.cos(thcirc) - tonz + \
            vals["connector height [mm]"]/25.4 + 1.2/25.4
        y_leh[i] = rleh * np.sin(thcirc) + hubs[i,1]

        # add tongue tip points to array
        x_hpt[2*i] = np.array([tonx,midx])
        y_hpt[2*i] = np.array([tony,midy])
        z_hpt[2*i] = np.array([tonz,midz])

        # determine half way between mouth clearances
        dz = (vals["C18"]["shell thickness [mm]"] + \
            vals["C18"]["mouth clearance [mm]"]) / 2. / 25.4

        # finish tongue tip points array
        x_hpt[2*i+1] = np.array([midx,midx])
        y_hpt[2*i+1] = np.array([midy,midy])
        z_hpt[2*i+1] = np.array([midz+dz,midz-dz])
        

    # create each DXF file
    # actuation hubs
    dxf.dxf(folder+"03_Base_hubs_Above",x_hub,y_hub,y_hub*0.,geometry="spline")
    # access holes
    dxf.dxf(folder+"04_Base_accs_Above",x_acc,y_acc,y_acc*0.,geometry="spline")
    # LE holes
    if vals["make leading edge holes"]:
        dxf.dxf(folder+"06_Base_LEhl_Nose",x_leh,y_leh,y_leh*0.,\
            geometry="spline")
    # tongue tip points
    dxf.dxf(folder+"05_Base_tTip_GC",x_hpt,y_hpt,z_hpt,geometry="line")

    return

# plot half
def half_plotter(a,mins,maxs,C18_guides,af_guides,vals,dist,axe,face,ymult=1.):
    # release values
    ax,ay,az = a; xmin,ymin,zmin = mins; xmax,ymax,zmax = maxs

    # run through each airfoil and plot it
    ind = 1; col = "k"
    for i in range(len(ax)):
        if vals["b"][ind] < dist[i+vals["start"],1]:
            if ind == (vals["b"].shape[0]-2):
                col = "0.3"
            elif ind == 1:
                col = "m"
            else:
                if col == "k":
                    col = "g"
                else:
                    col = "k"
            ind += 1
        # run through each line segment and plot it
        for j in range(ax[i].shape[0]):
            axe.plot(ax[i][j],ymult*ay[i][j],-az[i][j],c=col,zdir= face)

            # solve min and maxes
            axmin = np.min(ax[i][j]); axmax = np.max(ax[i][j])
            aymin = np.min(ymult*ay[i][j]); aymax = np.max(ymult*ay[i][j])
            azmin = np.min(-az[i][j]); azmax = np.max(-az[i][j])

            if axmin < xmin: xmin = axmin
            if axmax > xmax: xmax = axmax
            if aymin < ymin: ymin = aymin
            if aymax > ymax: ymax = aymax
            if azmin < zmin: zmin = azmin
            if azmax > zmax: zmax = azmax
    
    if vals["plot"]["guide curves"]:
        # plot guide curves
        for j in range(C18_guides[0].shape[0]): # 1): # 
            for k in range(C18_guides[0][j].shape[0]):
                # initialize plane lists
                x_list = []; y_list = []; z_list = []

                # run through each plane
                for i in range(C18_guides.shape[0]):
                    x_list.append(C18_guides[i][j][k][0])
                    y_list.append(ymult*C18_guides[i][j][k][1])
                    z_list.append(-C18_guides[i][j][k][2])
                
                # set guidecurves blue if desired
                if vals["plot"]["guides solid color"]:
                    co = "b"
                else:
                    co = ""

                # plot
                axe.plot(x_list,y_list,z_list,c=co,linewidth=0.5,zdir=face)
        
        for k in range(af_guides[0].shape[0]):
            # initialize plane lists
            x_list = []; y_list = []; z_list = []

            # run through each plane
            for i in range(af_guides.shape[0]):
                x_list.append(af_guides[i][k][0])
                y_list.append(ymult*af_guides[i][k][1])
                z_list.append(-af_guides[i][k][2])
            
            # set guidecurves blue if desired
            if vals["plot"]["guides solid color"]:
                co = "r"
            else:
                co = ""

            # plot
            axe.plot(x_list,y_list,z_list,c=co,linewidth=0.5,zdir=face)
    
    mins = [xmin,ymin,zmin]; maxs = [xmax,ymax,zmax]

    return mins, maxs

# plot full
def plotter(apts,C18_guides,af_guides,vals,dist):
    print("Plotting...")

    # withdraw points
    ax,ay,az = apts

    # plot a 3d plot of the wing
    fig = plt.figure()
    axe = fig.add_subplot(111, projection='3d')
    face = 'z'
    
    # initialize min and max vars
    xmin = ax[0][0][0]; xmax = ax[0][0][0]
    ymin = ay[0][0][0]; ymax = ay[0][0][0]
    zmin = az[0][0][0]; zmax = az[0][0][0]

    # plot half
    a = [ax,ay,az]; mins = [xmin,ymin,zmin]; maxs = [xmax,ymax,zmax]
    mins,maxs = half_plotter(a,mins,maxs,C18_guides,af_guides,\
        vals,dist,axe,face)
    
    # if the whole design is to be plotted:
    if not vals["plot"]["halfspan"]:
        mins,maxs = half_plotter(a,mins,maxs,C18_guides,af_guides,\
            vals,dist,axe,face,ymult=-1)
    
    # release values
    xmin,ymin,zmin = mins; xmax,ymax,zmax = maxs
    
    # solve for center
    xcent = (xmax + xmin) / 2.
    ycent = (ymax + ymin) / 2.
    zcent = (zmax + zmin) / 2.

    # solve for differences
    xdiff = np.abs(xmax - xmin)
    ydiff = np.abs(ymax - ymin)
    zdiff = np.abs(zmax - zmin)

    # solve for max difference
    max_diff = max([xdiff,ydiff,zdiff])

    # define limits
    x_lims = [xcent + 0.5*max_diff,xcent - 0.5*max_diff]
    y_lims = [ycent + 0.5*max_diff,ycent - 0.5*max_diff]
    z_lims = [zcent + 0.5*max_diff,zcent - 0.5*max_diff]

    # set limits
    axe.set_xlim3d(x_lims[1], x_lims[0])
    axe.set_ylim3d(y_lims[0], y_lims[1])
    axe.set_zlim3d(z_lims[1], z_lims[0])

    # # plot where ducted fans go# plot a 3d plot of the wing
    # axe.plot([0,0],[ 4.5, 8.5],[0,0],c="m",zdir=face)
    # axe.plot([0,0],[-4.5,-8.5],[0,0],c="m",zdir=face)

    axe.view_init(20,-30)
    plt.show()

    return

def main(input_file):
    # read in json file
    all_vals = read_json(input_file)

    # find runner and run
    for run in all_vals["run"]:
        # set vals dict
        vals = all_vals[run]

        # get dist dictionary
        dist,start,folder = get_dist(vals)
        # 0 - cpx, 1 - cpy, 2 - cpz, 3 - chord, 4 - dihedral
        # body fixed coordinates

        # create nomenclature file
        make_nomenclature(folder)

        # make interpolation functions
        make_interpolations(dist,vals)

        # create 3d, 2d, and guidecurves
        apts,afpts,C18_gc,af_gc,inn_gc = make_shapes(dist,start,\
            vals)

        # airframe guide curves
        airframe_guidecurves = [C18_gc,af_gc,inn_gc]

        # create plane dxf files
        pltxyz,bist = make_parts(dist,vals,airframe_guidecurves) # ,hist

        # Dallin Code
        zbumper(vals,folder,bist)
                
        # create lines for planes
        folder += "Base/"
        make_planes(bist,vals,folder,"00_Base_sect_RPlane")
        make_hubs_planes(dist,folder,"02_Base_hubs_TPlane")

        # create actuation hubs and holes to reach hubs
        make_hubs(bist,vals,folder)
        
        # plot the data
        if vals["plot"][""]:
            plotter(apts,C18_gc,af_gc,vals,dist)
            
    return

input_file = "placer_c18_input.json"
main(input_file)
