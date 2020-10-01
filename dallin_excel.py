

# code for excel files
def send_excel(xgc,ygc,zgc):
    # convert the array into a dataframe
    #df = pd.DataFrame(guide_curve_points)
    # save to xlsx file
    dfx = pd.DataFrame(xgc)
    dfy = pd.DataFrame(ygc)
    dfz = pd.DataFrame(zgc)
    print('if you got here')
    writer = pd.ExcelWriter('guide_curves_4p0.xlsx', engine = 'xlsxwriter')
    dfx.to_excel(writer, sheet_name='X-coord.',index = False)
    dfy.to_excel(writer, sheet_name='Y-coord.',index = False)
    dfz.to_excel(writer, sheet_name='Z-coord.',index = False)
    writer.save()
