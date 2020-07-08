from CPT import *

model = CPT()
# data, target = model.load_files("./data/train.csv", "./data/test.csv")

data = [['PH2', 'HD2', 'HDR', 'HDA', 'HDE'],
        ['PH2', 'HD2', 'PH7', 'HD7', 'HDR'],
        ['PH7', 'HD7', 'HDR'],
        ['PH7', 'HD7', 'HDR']]

target = [['PH2', 'HDZ']]

model.train(data)
predictions = model.predict(data, target, 5, 3)
