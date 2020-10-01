import json
import numpy as np

def interpolate(x1,x2,x,y1,y2):
    return (y2-y1)/(x2-x1)*(x-x1)+y1

def isIterable(val):
    try:
        iter(val)
        return True
    except:
        return False

f = open('HorizonAircraft.json', 'r')
horizonDict = json.load(f)
f.close()
f = open('HorizonScene.json', 'r')
sceneDict = json.load(f)
f.close()




## update the chord function in the main wing of the aircraft
#####################################################################
def chord(s):
    if s < 0. or s > 1.: raise ValueError('Unexpected value for span fraction')
    
    s_bay = 0.07560964056743188
    c_bay = 2.75                ## ft
    if s <= s_bay: return c_bay
    
    s_taper = 0.1613005665438547
    
    c_tip = 0.9166666666666666      ## 11. / 12.
    c_taper = 1.1458333333333333     ## c_tip / 0.8
    
    if s >= s_taper: return interpolate(s_taper, 1., s, c_taper, c_tip)
    
    A, B, C, D = 5.06167082e+03, -1.80033656e+03,  1.85435652e+02, -3.16641665e+00
    
    return A * s**3. + B * s**2. + C * s + D

def Chord(S):
    if isIterable(S):
        return np.array([chord(s) for s in S])
    else:
        return chord(S)

horizonDict['wings']['wing']['chord'] = Chord

## update scene with horizon dict

sceneDict['scene']['aircraft']['Horizon']['file'] = horizonDict

## update control functions in the control state of the scene
######################################################################
s_bay = 0.07560964056743188
s_taper = 0.1613005665438547
s_0 = 0.32904045
s_1 = 0.49678034
s_2 = 0.66452023
s_3 = 0.83226011
def sym(s, symCenter=10., symDefl=[8,6,4,2,-1]):
    if s < 0. or s > 1.: raise ValueError('Unexpected value for span fraction')
    sym0, sym1, sym2, sym3, sym4 = symDefl
    if s < s_bay: return symCenter
    if s < s_taper: return 0.
    if s < s_0: return sym0
    if s < s_1: return sym1
    if s < s_2: return sym2
    if s < s_3: return sym3
    return sym4
def asym(s, asymDefl=[0,0,0,0,0]):
    if s < 0. or s > 1.: raise ValueError('Unexpected value for span fraction')
    asym0, asym1, asym2, asym3, asym4 = asymDefl
    if s < s_taper: return 0.
    if s < s_0: return asym0
    if s < s_1: return asym1
    if s < s_2: return asym2
    if s < s_3: return asym3
    return asym4
def Sym(S):
    if isIterable(S):
        return np.array([sym(s) for s in S])
    else:
        return sym(S)
def Asym(S):
    if isIterable(S):
        return np.array([asym(s) for s in S])
    else:
        return asym(S)
sceneDict['scene']['aircraft']['Horizon']['control_state'][ 'sym'] =  Sym
sceneDict['scene']['aircraft']['Horizon']['control_state']['asym'] = Asym


## test
######################################################################
'''
import numpy as np

S = np.linspace(0,1,30001)
C = Chord(S)

import matplotlib.pyplot as plt

plt.plot(S,C, '-o', fillstyle='none')
plt.show()
'''


## view Horizon
import machupX as mx
import matplotlib.pyplot as plt

scene = mx.Scene(sceneDict)

# scene.add_aircraft('Horizon', horizonDict, state=sceneDict['scene']['aircraft']['Horizon']['state'],
                                   # control_state=sceneDict['scene']['aircraft']['Horizon']['control_state'])

# scene.display_wireframe(show_vortices=False, show_legend=True)

scene.solve_forces(verbose=True)

dist = scene.distributions(make_plots=['chord', 'dihedral', 'sweep', 'delta_flap'], show_plots=True)

dist = dist['Horizon']

# run through each wing
for wing in dist:
    delete = []

    # run through each key, and delete if unneccesary
    for key in dist[wing]:
        if key == "dihedral" or key == "cpx" or key == "cpy" or key == "cpz" \
            or key == "chord":
            thisvariable = 0
        else:
            delete.append(key)
    
    # delete unneeded keys
    for i in range(len(delete)):
        del dist[wing][delete[i]]

# save for later use
first = {}
ind = -1
first["chord"] = dist["wing_left"]["chord"][ind]
first["cpx"] = dist["wing_left"]["cpx"][ind]
first["cpy"] = dist["wing_left"]["cpy"][ind]
first["cpz"] = dist["wing_left"]["cpz"][ind]
first["dihedral"] = dist["wing_left"]["dihedral"][ind]

del dist["wing_left"]
del dist["winglet_left"]

# create new distributions file
nist = {}

# create numpy array for each important key
for key in dist["wing_right"]:
    nist[key] = np.insert(np.append(np.array(dist["wing_right"][key]),\
        np.array(dist["winglet_right"][key])),0,first[key]).tolist()

import json
with open("distributions_Sep22_2020.dist","w") as fp:
    json.dump(nist,fp, sort_keys=True, indent=4)

# for i in dist['wing_right']:
#     print(i)

# cpyWR = dist['wing_right']['cpy']
# cpyWL = dist['wing_left' ]['cpy']
# cpyWLR = dist['winglet_right']['cpy']
# cpyWLL = dist['winglet_left' ]['cpy']
# cpy = cpyWR + cpyWL + cpyWLR + cpyWLL

# chordWR = dist['wing_right']['chord']
# chordWL = dist['wing_left' ]['chord']
# chordWLR = dist['winglet_right']['chord']
# chordWLL = dist['winglet_left' ]['chord']
# chordPlot = chordWR + chordWL + chordWLR + chordWLL

# sweepWR = dist['wing_right']['sweep']
# sweepWL = dist['wing_left' ]['sweep']
# sweepWLR = dist['winglet_right']['sweep']
# sweepWLL = dist['winglet_left' ]['sweep']
# sweep = sweepWR + sweepWL + sweepWLR + sweepWLL

# delta_flapWR = dist['wing_right']['delta_flap']
# delta_flapWL = dist['wing_left' ]['delta_flap']
# delta_flapWLR = dist['winglet_right']['delta_flap']
# delta_flapWLL = dist['winglet_left' ]['delta_flap']
# delta_flap = delta_flapWR + delta_flapWL + delta_flapWLR + delta_flapWLL

# plt.plot(cpy, delta_flap, 'o')
# plt.show()

# scene.export_stl(filename='Horizon.stl')
