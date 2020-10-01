# All z-adjustment assistant fucntions

import math
import numpy as np


####--------------------------------CIRCLE---------------------------------####
def zinyout(r,ycg,zcg,z,side):
    y = math.sqrt(abs((r**2) - ((z-zcg)**2) )) + ycg
    if (y > ycg) and (side == 'L'):
        diff = y-ycg
        y = ycg-diff
    return y


def yinzout(r,ycg,zcg,y,down):
    if down == False:
        zup = math.sqrt(abs((r**2) - ((y-ycg)**2) )) + zcg
        diff = zup - zcg
        z = zcg - diff
    else:
        z = math.sqrt(abs((r**2) - ((y-ycg)**2) )) + zcg
    return z


def mat_fill(y,z,m,mat,b,adder):
    mat[0+adder][0] = (y**3)
    mat[0+adder][1] = (y**2)
    mat[0+adder][2] = y
    mat[0+adder][3] = 1
    b[0+adder][0] = z

    mat[1+adder][0] = 3*(y**2)
    mat[1+adder][1] = 2*y
    mat[1+adder][2] = 1
    mat[1+adder][3] = 0
    b[1+adder][0] = m
    return (mat,b)


def z_adj(z,zcg,circz,over):
    if over == True:
        nclow = (min(circz)) - ( 0.85*(min(circz)-zcg) )    # new circle low/ circle center
        if (z < min(circz)):
            zadj = min(circz)
        elif (min(circz) <= z <= max(circz)):
            perc = ( (z-max(circz)) / (min(circz)-max(circz)) )
            zadj = (perc*((min(circz))-nclow)) + nclow
        elif (max(circz) < z):
            zadj = nclow
    else:
        nchigh = (max(circz)) + ( 0.85*(zcg - max(circz)) )    # new circle high/ circle center
        if (z < min(circz)):
            zadj = nchigh
        elif (min(circz) <= z <= max(circz)):
            perc = ( (z-max(circz)) / (min(circz)-max(circz)) )
            zadj = (perc*(nchigh - (max(circz)) )) + max(circz)
        elif (max(circz) < z):
            zadj = max(circz)
    return zadj


def interval_mkr(r,t,circz,ycg,zcg,iy1,iz1,iyup,izup,iydo,izdo,iy4,iz4,over):
    newr = r+t
    m1 = (izup-iz1)/ (iyup-iy1)
    m4 = (izdo-iz4)/ (iydo-iy4)
    iz2 = z_adj(iz1,zcg,circz,over)
    iy2 = zinyout(newr,ycg,zcg,iz2,'L')

    iz3 = z_adj(iz4,zcg,circz,over)
    iy3 = zinyout(newr,ycg,zcg,iz3,'R')

    m2 = -1/ ((zcg-iz2)/ (ycg-iy2))
    m3 = -1/ ((iz3-zcg)/ (iy3-ycg))

    mat1 = np.zeros((4,4))
    b1 = np.zeros((4,1))
    mat2 = np.zeros((4,4))
    b2 = np.zeros((4,1))

    (mat1,b1) = mat_fill(iy1,iz1,m1,mat1,b1,0)
    (mat1,b1) = mat_fill(iy2,iz2,m2,mat1,b1,2)
    (mat2,b2) = mat_fill(iy3,iz3,m3,mat2,b2,0)
    (mat2,b2) = mat_fill(iy4,iz4,m4,mat2,b2,2)

    inv1 = np.linalg.inv(mat1)
    inv2 = np.linalg.inv(mat2)
    cub1 = np.dot(inv1,b1)
    cub2 = np.dot(inv2,b2)
    return (cub1,cub2,iy2,iy3,newr)


def zbump(y,zog,cub1,newr,ycg,zcg,cub2,iy1,iy2,iy3,iy4,down):
    if (y < iy1) or (y > iy4):
        z = zog
    elif (iy1 <= y) and (y <= iy2):
        z = cub1[0]*(y**3) + cub1[1]*(y**2) + cub1[2]*y + cub1[3]
    elif (iy2 <= y) and (y <= iy3):
        if down == True:
            z = yinzout(newr,ycg,zcg,y,True)
        else:
            z = yinzout(newr,ycg,zcg,y,False)
    elif (iy3 <= y) and (y <= iy4):
        z = cub2[0]*(y**3) + cub2[1]*(y**2) + cub2[2]*y + cub2[3]
    return z

####------------------------------ELLIPSE---------------------------------####
def zinyout_elip(z,iy,ycg,r,shrink,lipz):
    if (iy < ycg):
        y = ((-1) * math.sqrt(abs((r**2)-(shrink*((z-lipz)**2))) ) ) + ycg
    else:
        y = (math.sqrt(abs((r**2)-(shrink*((z-lipz)**2))) )) + ycg        
    return y


def yinzout_elip(y,r,ycg,shrink,lipz):
    z = (math.sqrt( (1/shrink) *((r**2) - ((y-ycg)**2)) ) ) + lipz
    return z

def z_adj_elip(z,zcen,zpos):
    nchigh = (zpos) + ( 0.85*(zcen - zpos) )    # new circle high/ circle center
    top = zcen + (zcen-zpos)
    if (z < top):
        zadj = nchigh
    elif (top <= z <= zpos):
        perc = ( (z-zpos) / (top-zpos) )
        zadj = (perc*(nchigh - zpos)) + zpos
    elif (zpos < z):
        zadj = zpos
    return zadj


def interval_mkr_elip(r,shrink,lipz,ycg,iy1,iz1,iyup,izup,iydo,izdo,iy4,iz4,zcen,zpos):
    m1 = (izup-iz1)/ (iyup-iy1)
    m4 = (izdo-iz4)/ (iydo-iy4)
    iz2 = z_adj_elip(iz1,zcen,zpos)
    iy2 = zinyout_elip(iz2,iy1,ycg,r,shrink,lipz)

    iz3 = z_adj_elip(iz4,zcen,zpos)
    iy3 = zinyout_elip(iz3,iy4,ycg,r,shrink,lipz)

    m2 =( ((1/shrink)*((2*ycg)-(2*iy2))) / ((2*iz2)-(2*lipz)) )
    m3 =( ((1/shrink)*((2*ycg)-(2*iy3))) / ((2*iz3)-(2*lipz)) )
    
    mat1 = np.zeros((4,4))
    b1 = np.zeros((4,1))
    mat2 = np.zeros((4,4))
    b2 = np.zeros((4,1))

    (mat1,b1) = mat_fill(iy1,iz1,m1,mat1,b1,0)
    (mat1,b1) = mat_fill(iy2,iz2,m2,mat1,b1,2)
    (mat2,b2) = mat_fill(iy3,iz3,m3,mat2,b2,0)
    (mat2,b2) = mat_fill(iy4,iz4,m4,mat2,b2,2)

    inv1 = np.linalg.inv(mat1)
    inv2 = np.linalg.inv(mat2)
    cub1 = np.dot(inv1,b1)
    cub2 = np.dot(inv2,b2)
    return (cub1,cub2,iy2,iy3)


def zbump_elip(y,zog,ycg,r,shrink,lipz,cub1,cub2,iy1,iy2,iy3,iy4):
    if (y < iy1) or (y > iy4):
        z = zog
    elif (iy1 <= y) and (y <= iy2):
        z = cub1[0]*(y**3) + cub1[1]*(y**2) + cub1[2]*y + cub1[3]
    elif (iy2 <= y) and (y <= iy3):
        z = yinzout_elip(y,r,ycg,shrink,lipz)
    elif (iy3 <= y) and (y <= iy4):
        z = cub2[0]*(y**3) + cub2[1]*(y**2) + cub2[2]*y + cub2[3]
    return z


####-----------------------------LONGITUDINAL-----------------------------####
# The matrix filler subroutine for the cubic solver
def mat_fill(x,z,m,mat,b,adder):
    mat[0+adder][0] = (x**3)
    mat[0+adder][1] = (x**2)
    mat[0+adder][2] = x
    mat[0+adder][3] = 1
    b[0+adder][0] = z

    mat[1+adder][0] = 3*(x**2)
    mat[1+adder][1] = 2*x
    mat[1+adder][2] = 1
    mat[1+adder][3] = 0
    b[1+adder][0] = m
    return (mat,b)


# Recieves the x,y coordinate of the 2nd points (with the 1st being the origin)
#   and the 2 endpoints slopes. Outputs the cubic coefficients
def cubic_solver(x,y,m1,m2):

    mat = np.zeros((4,4))
    b = np.zeros((4,1))

    (mat,b) = mat_fill(0,0,m1,mat,b,0)
    (mat,b) = mat_fill(x,y,m2,mat,b,2)

    inv = np.linalg.inv(mat)
    cub = np.dot(inv,b)
    return cub


# Solves the cubic at a given x-position
# This y-coordinate is the fraction used in the GC transition
def frac_finder(exam,cub):
    frac0 = cub[0]*(exam**3) + cub[1]*(exam**2) + cub[2]*exam + cub[3]
    frac = float(frac0[0])
    return frac 


