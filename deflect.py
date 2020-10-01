import numpy as np
from scipy import optimize as opt
from matplotlib import pyplot as plt
# determine the differential of a point
def derivate(af,x):
    dx = 0.001
    f = af["c"]
    ok_backward = np.min(af["cx"]) < x-dx and np.min(af["cx"]) < x-dx*2.
    ok_forward  = np.max(af["cx"]) > x+dx and np.max(af["cx"]) > x+dx*2.

    if ok_backward and ok_forward:
        return (-f(x+2.*dx)+8*f(x+dx)-8*f(x-dx)+f(x-2.*dx))/(12.*dx)
    elif not ok_backward and ok_forward:
        return (-f(x+2.*dx)+4.*f(x+dx)-3.*f(x))/(2.*dx)
    elif ok_backward and not ok_forward:
        return (3.*f(x)-4.*f(x-dx)+f(x-2.*dx))/(2.*dx)
    else:
        raise ValueError("Can not do backward or forward differencing")
        return 0

# determine closest camber line point to a given point
def min_dist(af,xp,yp):
    # initialize arrays for results
    xc  = np.zeros(xp.shape)
    yc  = np.zeros(xp.shape)
    t   = np.zeros(xp.shape)
    dyc = np.zeros(xp.shape)
    is_above = np.zeros(xp.shape,dtype=bool)

    # run through each point and get the minimized dist point for each
    for i in range(xp.shape[0]):
        # function to find point
        def func(x):
            while af["cx"][-1] < x:
                x -= 1e-10
            return ((af["c"](x) - yp[i])**2. + (x - xp[i])**2.)**0.5
        
        # determine where point is 
        bnds = ((af["cx"][0],af["cx"][-1]),)
        # print(xp[i],bnds)
        xmin = opt.minimize(func,xp[i],bounds=bnds).x[0]

        # input/calculate values
        xc[i]  = xmin
        yc[i]  = af["c"](xmin)
        t[i]   = ((xc[i] - xp[i])**2. + (yc[i] - yp[i])**2.)**0.5 * 2.
        dyc[i] = derivate(af,xc[i])
        is_above[i] = not (yc[i] >= yp[i])

    return xc,yc,t,dyc,is_above

# get radius
def get_radius(x0,yc0,hp):
    # calculate radius
    r = ( (yc0-hp[1])**2. + (x0-hp[0])**2. )**0.5

    return r

# get psi
def get_psi(x0,yc0,hp):
    # calculate psi
    psi = np.arctan((yc0-hp[1])/(x0-hp[0]))

    return psi

# get deflected camberline
def get_deflected_camberline(x0,yc0,df,psi,r,hp):
    # initialize arrays to save to
    xc = np.zeros(x0.shape)
    yc = np.zeros(x0.shape)

    # run through each point and create camberline
    for i in range(x0.shape[0]):
        if x0[i] < hp[0]:
            xc[i] =  x0[i]
            yc[i] = yc0[i]
        else:
            xc[i] = hp[0] + r[i] * np.cos(df-psi[i])
            yc[i] = hp[1] - r[i] * np.sin(df-psi[i])

    return xc,yc

# get deflected camberline derivative
def get_deflected_camberline_deriv(x0,dyc0,df,hp):
    # initialize arrays to save to
    dyc = np.zeros(x0.shape)

    # run through each point and create camberline derivative
    for i in range(x0.shape[0]):
        if x0[i] < hp[0]:
            dyc[i] = dyc0[i]
        else:
            dyc[i] = (dyc0[i]-np.tan(df))/(1+dyc0[i]*np.tan(df))

    return dyc

# get upper and lower surfaces
def get_surfaces(xp,yp,t,dyc,is_above):
    # initialize x y arrays
    x = np.zeros(xp.shape)
    y = np.zeros(xp.shape)

    # calculate upper and lower surface x y values
    for i in range(xp.shape[0]):
        x[i] = xp[i] -(t[i]/2./(1+(dyc[i]**2.))**0.5 * dyc[i] if is_above[i] \
            else -t[i]/2./(1+(dyc[i]**2.))**0.5 * dyc[i])
        y[i] = yp[i] +(t[i]/2./(1+(dyc[i]**2.))**0.5 if is_above[i] \
            else -t[i]/2./(1+(dyc[i]**2.))**0.5)

    return x,y

# calculate R 
def get_R(dp):
    R = (4*np.tan(dp)**2. +1)**.5 + np.arcsinh(2*np.tan(dp))/2./np.tan(dp)
    return R

# secant method
def Secant(fun,x0,x1, Err = 1e-5, maxiter = 1000):
    
    # initialize a large number so the err is less than
    E = 10000000
    
    # initialize the icounter
    i = 0
    
    # till threshold error is reached, continue
    while(E > Err and i < maxiter):
        
        x2 = x1 - fun(x1)*(x1-x0)/(fun(x1)-fun(x0))
        
        E = abs( (x2 - x1) / x1 )
        
        # start on the next value
        x0 = x1
        x1 = x2
        
        # add counter value
        i += 1
    
    # return the root
    return x2

# find Zp
def get_Zp_Np(Z0,l,R,dp,ZTE):
    # initialize zp array
    Zparr = np.zeros(Z0.shape)

    for i in range(Z0.shape[0]):
        # define function for secant method
        def func(Zp):
            a = Zp/2.*(Zp**2./l**2.*R**2.*np.tan(dp)**2. + 1)**0.5
            b = l/2./R/np.tan(dp)*np.arcsinh(Zp/l*R*np.tan(dp))
            return a+b - Z0[i]
        
        # determine value
        Zparr[i] = Secant(func,ZTE*Z0[i]/l,0.1)
    
    # calculate Np
    Np = - Zparr**2./ZTE * np.tan(dp)

    return Zparr,Np

# calculate xp yp values
def get_xpyp(Zp,Np,hp,phi):
    # calculate values
    xp = hp[0] + Zp*np.cos(phi) - Np*np.sin(phi)
    yp = hp[1] + Zp*np.sin(phi) + Np*np.cos(phi)

    return xp,yp

# get deflected camberline
def get_parabolic_camberline(x0,yc0,xp,yp,Dyc,Zp,ZTE,dp,hp):
    # initialize arrays to save to
    xc = np.zeros(x0.shape)
    yc = np.zeros(x0.shape)

    # run through each point and create camberline
    for i in range(x0.shape[0]):
        if x0[i] < hp[0]:
            xc[i] =  x0[i]
            yc[i] = yc0[i]
        else:
            xc[i] = xp[i] + Dyc[i]*np.sin(np.arctan(2*Zp[i]/ZTE*np.tan(dp)))
            yc[i] = yp[i] + Dyc[i]*np.cos(np.arctan(2*Zp[i]/ZTE*np.tan(dp)))

    return xc,yc

# get deflected camberline derivative
def get_parabolic_camberline_deriv(x0,dyc0,Zp,ZTE,dp,hp):
    # initialize arrays to save to
    dyc = np.zeros(x0.shape)

    # run through each point and create camberline derivative
    for i in range(x0.shape[0]):
        if x0[i] < hp[0]:
            dyc[i] = dyc0[i]
        else:
            dyc[i] = (dyc0[i]-2.*Zp[i]*np.tan(dp/ZTE))/(1+2.*Zp[i]*\
                (np.tan(dp/ZTE))*dyc0[i])

    return dyc

def get_deflected(af_dict,x,y,vals,c=1.0):
    # write in values
    xf_c = vals["f"]
    df   = vals["da"]

    # get x0, yc0, dyc0 and t
    x0,yc0,t,dyc0,is_above = min_dist(af_dict,x,y)

    # get hinge point
    if vals["th"]:
        yhp = af_dict["t"](xf_c) - vals["t"]/2.
    else:
        yhp = af_dict["c"](xf_c)

    # set up tuple of hinge point
    hp = (xf_c,yhp)

    # determine radius function
    r = get_radius(x0,yc0,hp)

    # determine psi
    psi = get_psi(x0,yc0,hp)

    # turn degree df to radians
    df = np.deg2rad(df)

    # determine deflected camberline
    xc,yc = get_deflected_camberline(x0,yc0,df,psi,r,hp)

    # determine deflected camberline derivative
    dyc = get_deflected_camberline_deriv(x0,dyc0,df,hp)

    # calculate l
    l = (hp[1]**2. + (c-hp[0])**2.)**0.5

    # calculate flap neutral line angle
    phi = - np.arctan2(hp[1],c-hp[0])

    # calculate R value
    dp = df
    R = get_R(dp)

    # calculate ZTE
    ZTE = 2.*l/R

    # calculate NTE
    NTE = -2.*l/R*np.tan(dp)

    # calculate Z0
    Z0 = (x0-hp[0])/(c-hp[0])*l

    # calculate Zp
    Zp,Np = get_Zp_Np(Z0,l,R,dp,ZTE)

    # calculate xp and yp
    xp,yp = get_xpyp(Zp,Np,hp,phi)

    # calculate ynl
    ynl = hp[1]*(1-(x0-hp[0])/(c-hp[0]))

    # calculate Dyc
    Dyc = yc0 - ynl

    # calculate parabolic camberline
    xcp,ycp = get_parabolic_camberline(x0,yc0,xp,yp,Dyc,Zp,ZTE,dp,hp)

    # calculate parabolic camberline derivative
    dycp = get_parabolic_camberline_deriv(x0,dyc0,Zp,ZTE,dp,hp)

    # determine upper and lower surfaces
    un = {}
    un["x"],un["y"] = get_surfaces(x0 ,yc0,t,dyc0,is_above)
    # determine surfaces deflected
    de = {}
    de["x"],de["y"] = get_surfaces(xc ,yc ,t,dyc ,is_above)
    # determine surfaces deflected parabolically
    pa = {}
    pa["x"],pa["y"] = get_surfaces(xcp,ycp,t,dycp,is_above)
    
    return un,de,pa

def main(af,x,y,vals):
    # initialize un,de,pa dictionary arrays
    un = {}
    un["x"] = np.zeros(x.shape,dtype=np.ndarray)
    un["y"] = np.zeros(x.shape,dtype=np.ndarray)
    de = {}
    de["x"] = np.zeros(x.shape,dtype=np.ndarray)
    de["y"] = np.zeros(x.shape,dtype=np.ndarray)
    pa = {}
    pa["x"] = np.zeros(x.shape,dtype=np.ndarray)
    pa["y"] = np.zeros(x.shape,dtype=np.ndarray)

    # run through each line in x and shift it in the deflected shape
    for i in range(x.shape[0]):
        if type(x[i]) == np.ndarray:
            uni,dei,pai = get_deflected(af,x[i],y[i],vals)
            un["x"][i] = uni["x"]; un["y"][i] = uni["y"]
            de["x"][i] = dei["x"]; de["y"][i] = dei["y"]
            pa["x"][i] = pai["x"]; pa["y"][i] = pai["y"]
    # if type(x[0]) != np.ndarray:
    #     uni,dei,pai = get_deflected(af,x,y,vals)
    #     un["x"] = x; un["y"][i] = y
    #     de["x"] = x; de["y"][i] = y
    #     pa["x"] = x; pa["y"][i] = y

    if vals["dty"] == "traditional":
        xvals = de["x"]; yvals = de["y"]
    elif vals["dty"] == "parabolic":
        xvals = pa["x"]; yvals = pa["y"]
    else: # vals["dty"] == "none":
        xvals = un["x"]; yvals = un["y"]

    return xvals,yvals


# import airfoil_db as adb

# foilfile = "/home/ben/foilshapes/E335.txt"

# # create airfoil shape
# add = {}
# add["geometry"] = {}
# add["geometry"]["type"] = "outline_points"
# add["geometry"]["outline_points"] = foilfile

# # initialize airfoil database
# foil = adb.Airfoil("foil",add)
# coords = foil.get_outline_points()
# x = coords[:,0]; y = coords[:,1]
# # ensure the airfoil is closed
# x0 = x[0]; x1 = x[-1]
# if not x0 == x1:
#     if x0 > x1:
#         x = np.append(x,x[0])
#         y = np.append(y,y[0])
#     else:
#         x = np.insert(x,0,x[-1])
#         y = np.insert(y,0,y[-1])

# x0 = []; y0 = []
# main()