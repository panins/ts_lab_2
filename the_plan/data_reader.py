import numpy as np
from datetime import datetime
from io import StringIO
from os import listdir, path
# from os imoirt

def csv_reader(file_path):
    with open(file_path,'r',encoding='utf-8') as f:
        data = f.read().replace('买盘','1').replace('卖盘','-1').replace('中性盘','0').replace('--','0')
        # names = ('datetime','price','change','volume','amount','type')
        convertfunc_datetime = lambda x: datetime.strptime(x.decode('utf-8'), '%Y-%m-%d %H:%M:%S')
        dtype = [
            ('datetime', datetime),
            ('price', np.float32),
            ('change', np.float32),
            ('volume', np.int32),
            ('amount', np.int32),
            ('type', np.int32)
        ]
        a = np.genfromtxt(StringIO(data),
                          skip_header=0,
                          usecols=(1,2,3,4,5,6),
                          delimiter=',',
                          names = True,
                          dtype = dtype,
                          converters={'datetime': convertfunc_datetime}
                       )
    return(a)

def source_finder(dir_path, suffix='.csv', fullpath=True):
    files=listdir(dir_path)
    csv_files =[]
    for file in files:
        if file.endswith(suffix):
            csv_files.append(file)
    if fullpath:
        for i in range(len(csv_files)):
            csv_files[i] = path.join(dir_path,csv_files[i])
    return(csv_files)

a=source_finder('/Users/panyuan/Desktop')
for i in a:
    print(csv_reader(i))

# a = csv_reader('/Users/panyuan/Desktop/000004.csv',dtype)
# print(a)
# a = csv_reader('/Users/panyuan/Desktop/000004.csv')
# print(a)