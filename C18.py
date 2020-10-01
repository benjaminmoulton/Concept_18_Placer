import C18_geometry as c18

# run file
jsonfile = 'input_c18.json'
# x2,y2,_ = 
info = c18.read_json(jsonfile)
for i in range(len(info["run"])):
    c18.main(info,info["run"][i])
    if len(info["run"]) != 1:
        print("{:>6.2f}% ({:>s}/{:>s})".format((i+1)/len(info["run"])*100.,\
            str(i+1).zfill(2),str(len(info["run"])).zfill(2)))


# # P18I2
# vals = c18.read_json(jsonfile)

# # get out original values
# c0 = vals["C18"]["chord [in]"]
# t0 = vals["C18"]["tongue start [in]"]
# m0 = vals["C18"]["mouth start [in]"]
# vals["C18"]["write dxf"] = True
# vals["C18"]["show plot"] = False
# vals["C18"]["dxf file path"] += "/P18I2/"
# vals["C18"]["dxf file name"] = "P18I2M0"

# # set chord lengths to test
# for i in range(4,11):
#     vals["C18"]["chord [in]"] = float(i)
#     vals["C18"]["tongue start [in]"] = vals["C18"]["chord [in]"] * t0 / c0
#     vals["C18"]["mouth start [in]"] = vals["C18"]["chord [in]"] * m0 / c0
#     vals["C18"]["dxf file name"] = vals["C18"]["dxf file name"][:-1] + str(i-4)

#     # run
#     c18.main(vals)
    


