from calendar import c
import os
import re
import csv
import pdb
import pandas as pd


# 假设response.txt
# line1 : input
# line2 : response

response_path = './base_8k/'
test_path= './test/'
file_count = len(os.listdir(response_path))

with open('prediction.csv','w',newline='',encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['ID','Category','Privacy','Pos_e','Pos_b'])


    for index in range(file_count):
        # index=780
        resp=[]
        
        with open(test_path+str(index)+'.txt','r',encoding='utf-8') as f1:
            x = f1.readline()
        with open(response_path+str(index)+'.txt','r',encoding='utf-8') as f2:
            y = f2.readline()

            # 删除response 中的 "
            y = re.sub(r'"','',y)

        # pdb.set_trace()
        privacy = []
        category =[]
        l,r = 0,0
        for i in range(len(y)):
            if y[i] =='<':
                l,r = i,i
            if y[i] =='>':
                r = i
                privacy = y[l+1:r]

            if y[i] =='[':
                l,r = i,i
            if y[i] ==']' :
                r = i
                category = y[l+1:r]
                resp.append([index,category,privacy])

        num=0  
        length = len(resp)

        i=0
        while(i<length):
            # pdb.set_trace()
            if resp[i][2] !=[] and  resp[i][2] !='':
                result = x[num:].find(resp[i][2])
            else:
                del resp[i]
                length -=1
                continue
                
            
            if result>=0:
                
                result += num
                resp[i].extend([result+len(resp[i][2])-1,result])
                num = result +len(resp[i][2])
            else:
                del resp[i]
                i -=1
                length -=1
            i += 1
        
        writer.writerows(resp)

# 读取CSV文件
data = pd.read_csv('prediction.csv')
data['Privacy'], data['Pos_b'] = data['Pos_b'], data['Privacy']
data.to_csv('predict.csv', index=False)
# print(resp)

        

            
