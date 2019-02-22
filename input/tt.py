import xlrd
import numpy as np
import matplotlib.pyplot as plt
file_location = "/home/local-admin/aaa.xlsx"
workbook = xlrd.open_workbook(file_location)
first_sheet = workbook.sheet_by_index(0)
print(first_sheet.ncols)
print(first_sheet.nrows)
x = [first_sheet.cell_value(i, 4) for i in range(first_sheet.nrows)]
y = [first_sheet.cell_value(i, 8) for i in range(first_sheet.nrows)]
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set(xlabel='layer: from center to edge',ylabel='$\sigma/\mu$')
ax.scatter(x,y,marker='.')
ax.plot(x,y)
axes = plt.gca()
axes.set_ylim([1.0,1.5])
plt.grid(True)
plt.show()
