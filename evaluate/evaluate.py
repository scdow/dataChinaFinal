#coding:utf8
import sys
import os

label = open('predict_lgb_12_14.txt', 'r').readlines() ## 读取标签
predict = open('predict_lgb_12_15.txt','r').readlines() ## 读取预测结果
assert(len(label) == len(predict))

## 统计准确率
oneerror = 0.0
threeerror = 0.0
for i in range(0,len(label)):
   thelabel = label[i].strip().split(":")[1]
   thepredict = predict[i].strip().split(":")[1]
   onelabel,threelabel = thelabel.split(' ')
   onepredict,threepredict = thepredict.split(' ')

   tmponeerror = abs(float(int(onelabel)-int(onepredict))) / (max(float(onelabel),float(onepredict))+1)
   tmpthreeerror = abs(float(int(threelabel)-int(threepredict))) / (max(float(threelabel),float(threepredict))+1)

   oneerror += tmponeerror
   threeerror += tmpthreeerror

   print(label[i].strip().split(":")[0]+' predict oneerror is:'+str(tmponeerror))
   print(label[i].strip().split(":")[0]+' predict threeerror is:'+str(tmpthreeerror))

oneerror /= len(label)
threeerror /= len(label)

## 输出分数
print("未来1个月预测误差="+str(oneerror))
print("未来3个月预测误差="+str(threeerror))

