# Program to be included in Ben's source code for zbumping.


# Re-Plots the guide_curves and a rough outline of the Engines

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import math

import zbumper as zbum
import handpicking as hp
import chamber as ch


def circ_points(r,x,y,z,every_deg):
    x_const = []
    yes = []
    zes = []
    for i in range(0,360,every_deg):
        rad = i/180*math.pi
        x_const.append(x)
        yes.append((r* math.cos(rad))+y)
        zes.append(((r* math.sin(rad))+z))
    x_const.append(x)
    yes.append(yes[0])
    zes.append(zes[0])
    return (x_const,yes,zes)


def ellipse_points(r,x,y,z,res,shrink):
    x_const = []
    yels = []
    zpos = []
    zneg = []
    step = (2*r) / res
    for i in range(res+1):
        yels.append((y-r)+(step*i))
        x_const.append(x)
        lipz = ( ( (z+r) - (0.16*r) ) - (math.sqrt((1/shrink)*(r**2)))  )
        zintp = math.sqrt( (1/shrink) *((r**2) - ((yels[i]-y)**2)) )  + lipz
        zintn = ((-1) * math.sqrt( (1/shrink) *((r**2) - ((yels[i]-y)**2)) ) ) + lipz
        zpos.append(zintp)
        zneg.append(zintn)
    return (x_const,yels,zpos,zneg,lipz)


def printer(x,y,z,ax):
    ax.plot(x,y,z, label = 'Engine',color = 'blue')
    

def engine(cg,r,length,cg_shift,shrink,ax):
    #cg = [-2.28,6.5,-1.86]
    #r = 1.5                         
    #length = 2.4                    
    #cg_shift = 2.3         
    circles_front = 10
    circles_back = 5
    every__deg = 15
    #shrink = 100                    # Ellipse z-direction shrink factor

    for i in range(0,circles_front+1):
        step = cg_shift/circles_front *i
        (x_const,yes,zes) = circ_points(r,cg[0]+step,cg[1],cg[2],every__deg)
        printer(x_const,yes,zes,ax)
    for i in range(0,circles_back+1):
        step = (length-cg_shift)/circles_back *i
        (x_const,yes,zes) = circ_points(r,cg[0]-step,cg[1],cg[2],every__deg)
        printer(x_const,yes,zes,ax)
    for xelem in [-8.4,-9.6,-10.8,-12]:
        (x_const,yels,zpos,zneg,lipz) = ellipse_points(r,xelem,cg[1],cg[2],100,shrink)
        printer(x_const,yels,zpos,ax)
        printer(x_const,yels,zneg,ax)
    #print('\nyes is: ',yes)
    #print('\nzes is: ',zes)
    return (ax,yes,zes,zpos,lipz)


def og_gc_printer(x,y,z,ax):
    for i in range(len(x)):
        if i in range(4,67):
            ax.plot(x[i],y[i],z[i], label = 'GC',color = 'red')


def new_gc_printer(x,y,z,ax):
    for i in range(len(x)):
        # if i in range(0,99):
        #if i < 67:
        #if i < 114:
        #if i > 44:
        #if i in [0,10,21,32,35,39,43,44,49,54,59,66,67,70,74,77,84,94,104,113,118,128,138,148,155,164,170,177,186,195,205]:
        #if i in [118,128,138,148,155,164,170,177,186,195,205]:
            #ax.plot(x[i],y[i],z[i], label = 'NEW',color = 'red')
        #else:
        ax.plot(x[i],y[i],z[i], label = 'NEW',color = 'green')


def index_finder(gcpts,cgy):
    index = 0
    for i in range(len(gcpts)-1):
        #if ( (gcpts[i+1] > cgy) and (gcpts[i] < cgy) ):
        if ( (abs(gcpts[i+1]-cgy)) < (abs(gcpts[i]-cgy)) ):
            index = (i+1)
    return index


def zgc_bumpNadd(xgc,ygc,zgc,circy,circz,elipz,cg,length,shift,r,shrink,lipz,t,srB,srO,srF,gcnumbeh,gcnumfro,ax):
    front = []
    over = []
    behind = []
    alone = []
    #print(len(ygc))
    #gcnumfro = 20
    #gcnumbeh = 30
    # for loop through each individual guide_curve
    for i in range(len(ygc)):
        # Determine whether gc is a FRONT, OVER, OR BEHIND
        for j in range(len(ygc[0])):
            if ((min(circy)<= ygc[i][j] <= max(circy)) and (i < (len(ygc)/3) ) and ((zgc[i][j]) >= (max(elipz)) ) ):
                elip_i = i              # index of the first gc to have BOTH endpoints 'below' elipz or zmax
    for i in range(len(ygc)):
        for j in range(len(ygc[0])):
            if ((min(circy)<= ygc[i][j] <= max(circy)) and (cg[0]+shift >= xgc[i][j]) and (zgc[i][j] < cg[2]) ):
                front_i = i             # the last index in over
    #print('the elip is: ',elip_i)
    #print('\nthe front_i is: ',front_i)
    for i in range(len(ygc)):
        if (i < elip_i):
            alone.append(i)
        elif (elip_i <= i < (front_i-gcnumbeh)):
            behind.append(i)
        elif ((front_i-gcnumbeh) <= i <= front_i+2):
            over.append(i)
        elif (front_i+2 < i <= (front_i+2+gcnumfro)):
            front.append(i)
        elif ((front_i+2+gcnumfro) < i):
            alone.append(i)
    #print('\ny min and max: ',min(circy),max(circy))
    #print('yeup: ', cg[0]-length+shift, cg[0]+shift)
    #print('front: ',front)
    #print('over: ',over)
    #print('behind: ',behind)
    #print('alone: ',alone)
    # for loop through each gc again to make corresponding adjustments
    #t = .35                      # max material thickness beyond nominal radius of ducted fan
    countfro = 0
    countbeh = 0

#-------------------Front Longitudinal Cubic---------------------#
    endi = index_finder(ygc[(min(front)+gcnumfro)],cg[1])
    x0f = abs(xgc[(min(front)+gcnumfro)][endi] - (cg[0]+shift))
    z0f = abs(zgc[(min(front)+gcnumfro)][endi] - (cg[2]-r-t))
    xf = x0f/z0f
    zf = z0f/z0f
    #print(x0f)
    #print(z0f)
    #print(xf)
    #print(zf)

    cub_front = zbum.cubic_solver(xf,zf,0,0)
#-------------------Over Longitudinal Cubic----------------------#
    begini = index_finder(ygc[(max(over)-gcnumbeh)],cg[1])
    x0b = abs(  (cg[0]+shift) - xgc[(max(over)-gcnumbeh)][begini]  )
    z0b = abs(  (cg[2]+r+t) - zgc[(max(over)-gcnumbeh)][begini]  )
    xb = x0b/z0b
    zb = z0b/z0b
    #print(x0b)
    #print(z0b)
    #print(xb)
    #print(zb)

    cub_behind = zbum.cubic_solver(xb,zb,-.15,0.15)
#-----------------------z bumping section------------------------#   
    for k in range(len(ygc)):
        if k in behind:             # down bump for BEHIND gc's
            #### Preliminary calcs for intervals for spanwise smoothing
            oldif_s = 10
            oldif_e = 10
            for l in range(len(ygc[k])):
                nedif_s = abs(ygc[k][l] - (min(circy)-srB[0]))
                nedif_e = abs(ygc[k][l] - (max(circy)+srB[1]))
                if nedif_s < oldif_s:
                    start_b = l
                    oldif_s = nedif_s
                if nedif_e < oldif_e:
                    end_b = l
                    oldif_e = nedif_e
            #### Preliminary calcs for longitudinal smoothing
            (cub1,cub2,iy2,iy3) = zbum.interval_mkr_elip(r,shrink,lipz,cg[1],ygc[k][start_b],zgc[k][start_b],ygc[k][start_b+1],zgc[k][start_b+1],ygc[k][end_b-1],zgc[k][end_b-1],ygc[k][end_b],zgc[k][end_b],min(elipz),max(elipz))
            #### File through a second time to perform the appropriate adjustments
            for l in range(len(ygc[k])):
                zgc[k][l] = zbum.zbump_elip(ygc[k][l],zgc[k][l],cg[1],r,shrink,lipz,cub1,cub2,ygc[k][start_b],iy2,iy3,ygc[k][end_b])
        
        if k in over:             # UP bump for OVER gc's
            #### Preliminary calcs for intervals for spanwise smoothing
            oldif_s = 10
            oldif_e = 10
            for l in range(len(ygc[k])):
                nedif_s = abs(ygc[k][l] - (min(circy)-srO[0]))
                nedif_e = abs(ygc[k][l] - (max(circy)+srO[1]))
                if nedif_s < oldif_s:
                    start_o = l
                    oldif_s = nedif_s
                if nedif_e < oldif_e:
                    end_o = l
                    oldif_e = nedif_e          
            #### Preliminary calcs for longitudinal smoothing
            index = index_finder(ygc[k],cg[1])
            exam = ( (cg[0]+shift) - xgc[k][index] ) /z0b
            frac = zbum.frac_finder(exam,cub_behind)
            (cub1,cub2,iy2,iy3,newr) = zbum.interval_mkr(r,t,circz,cg[1],cg[2],ygc[k][start_o],zgc[k][start_o],ygc[k][start_o+1],zgc[k][start_o+1],ygc[k][end_o-1],zgc[k][end_o-1],ygc[k][end_o],zgc[k][end_o],True)
            #### File through a second time to perform the appropriate adjustments
            for l in range(len(ygc[k])):
                if (countbeh <= gcnumbeh) and (k >= (over[-(gcnumbeh+1)])):
                    zog = frac * zgc[k][l]
                    zbu = zbum.zbump(ygc[k][l],zgc[k][l],cub1,newr,cg[1],cg[2],cub2,ygc[k][start_o],iy2,iy3,ygc[k][end_o],False)
                    znew = (1-frac) * zbu
                    zgc[k][l] = (zog + znew)
            if k >= (over[-gcnumbeh]):
                countbeh += 1

        elif k in front:            # down bump for FRONT gc's
            #### Preliminary calcs for intervals for spanwise smoothing
            oldif_s = 10
            oldif_e = 10
            for l in range(len(ygc[k])):
                nedif_s = abs(ygc[k][l] - (min(circy)-srF[0]))
                nedif_e = abs(ygc[k][l] - (max(circy)+srF[1]))
                if nedif_s < oldif_s:
                    start_f = l
                    oldif_s = nedif_s
                if nedif_e < oldif_e:
                    end_f = l
                    oldif_e = nedif_e          
            #### Preliminary calcs for longitudinal smoothing
            index = index_finder(ygc[k],cg[1])
            exam = ( xgc[k][index] - (cg[0]+shift) ) /z0f
            frac = zbum.frac_finder(exam,cub_front)
            (cub1,cub2,iy2,iy3,newr) = zbum.interval_mkr(r,0,circz,cg[1],cg[2],ygc[k][start_f],zgc[k][start_f],ygc[k][start_f+1],zgc[k][start_f+1],ygc[k][end_f-1],zgc[k][end_f-1],ygc[k][end_f],zgc[k][end_f],False)
            #### File through a second time to perform the appropriate adjustments
            for l in range(len(ygc[k])):
                if countfro <= gcnumfro:
                    zog = frac * zgc[k][l] 
                    zbu = zbum.zbump(ygc[k][l],zgc[k][l],cub1,newr,cg[1],cg[2],cub2,ygc[k][start_f],iy2,iy3,ygc[k][end_f],True)
                    znew = (1-frac)* zbu
                    zgc[k][l] = (zog + znew)
            countfro += 1


    # new_gc_printer(xgc,ygc,zgc,ax)
    return (xgc,ygc,zgc),alone,behind,over,front


def list2numpy(li):
    npy = np.zeros((len(li),len(li[0])))
    for i in range(len(li)):
        for j in range(len(li[0])):
            npy[i][j] = li[i][j]
    return npy


def finalNprint(ax):
    ax.set_xlabel('X-Axis')
    ax.set_ylabel('Y_Axis')
    ax.set_zlabel('Z_Axis')
    ax.axes.set_xlim3d(left=-48, right=48) 
    ax.axes.set_ylim3d(bottom=-36, top=48) 
    ax.axes.set_zlim3d(bottom=-48, top=48)
    
    #ax.plot([.54,.54],[-2,2],[.03,.03], label = 'LINE',color = 'red')

    #ax.legend()
    plt.show()


def main(xgc,ygc,zgc,vals):
    cg = vals["ducted fan cg [in]"]
    r = (vals["ducted fan diameter, width, and length [in]"][0]) /2
    length = vals["ducted fan diameter, width, and length [in]"][2]
    cg_shift = vals["z bump"]["ducted fan cg shift [in]"]
    t = vals["z bump"]["ducted fan cover thickness [in]"]
    shrink = vals["z bump"]["ducted fan exhaust shrink factor"]
    srB = vals["z bump"]["span range Behind [in]"]
    srO = vals["z bump"]["span range Over [in]"]
    srF = vals["z bump"]["span range Front [in]"]
    gcnumbeh = vals["z bump"]["number guide curves aft"]
    gcnumfro = vals["z bump"]["number guide curves forward"]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')

    # ax,yes,zes,zpos,cg,length,cg_shift,r,shrink,lipz
    (ax,yes,zes,zpos,lipz) = engine(cg,r,length,cg_shift,shrink,ax)

    (xnewgc,ynewgc,znewgc),alone,behind,over,front = zgc_bumpNadd(xgc,ygc,zgc,yes,zes,zpos,cg,length,cg_shift,r,shrink,lipz,t,srB,srO,srF,gcnumbeh,gcnumfro,ax)

    xgc = list2numpy(xnewgc)
    ygc = list2numpy(ynewgc)
    zgc = list2numpy(znewgc)

    # run chamber
    bx,by,bz,alpha_b,fx,fy,fz,alpha_f,gc1,gc2,gc3,gc4,behind_i,front_i = ch.main(xgc,ygc,zgc,over,cg,r,cg_shift,zpos,ax,vals)

    # combine
    stuff = bx,by,bz,alpha_b,fx,fy,fz,alpha_f,gc1,gc2,gc3,gc4,behind_i,front_i

    # intialize guide curve number
    gc_number = 35

    # initialize return arrays
    xhp = np.zeros((gc_number,xgc.shape[1]))
    yhp = np.zeros((gc_number,xgc.shape[1]))
    zhp = np.zeros((gc_number,xgc.shape[1]))

    # handpicking
    hpicks = hp.handpick(gc_number,alone,behind,over,front)
    # xgc shape (200,70)

    # for loop to retrieve handpicks
    for i in range(gc_number):
        xhp[i] = xgc[hpicks[i]]
        yhp[i] = ygc[hpicks[i]]
        zhp[i] = zgc[hpicks[i]]
    
    new_gc_printer(xhp,yhp,zhp,ax)

    finalNprint(ax)
    return xhp,yhp,zhp,hpicks,stuff


#main()