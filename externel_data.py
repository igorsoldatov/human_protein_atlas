import requests
import pandas as pd

colors = ['red','green','blue','yellow']
DIR = "input/HPAv18/jpg/"
v18_url = 'http://v18.proteinatlas.org/images/'

imgList = pd.read_csv("input/HPAv18/HPAv18RBGY_wodpl.csv")

len(imgList)

for i in imgList['Id'][:5]: # [:5] means downloard only first 5 samples, if it works, please remove it
    img = i.split('_')
    for color in colors:
        img_path = img[0] + '/' + "_".join(img[1:]) + "_" + color + ".jpg"
        img_name = i + "_" + color + ".jpg"
        img_url = v18_url + img_path
        r = requests.get(img_url, allow_redirects=True)
        open(DIR + img_name, 'wb').write(r.content)