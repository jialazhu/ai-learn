import csv
import datetime
import os.path
import re

import Car
from Car import car
from Car import dongfeng_car

car = car("2022年","7月份","火车")
print(car)
car.drive(500)
print(car.get_total())
car.is_max_speed(500)

dongfeng_car = dongfeng_car("2022年","7月份","火车")
print(dongfeng_car)
print(car.get_total())


def datas(value,*avalue,**bvalue):
    if value:
        print(value)
    if avalue:
        print(avalue)
    if bvalue:
        for key,value in bvalue.items():
            print(key,value)

datas(1)
datas(1,2,"3",3.44, name ="1",phone = 1234)

def test(a,b):
    return a + b

print(test(b =1, a = 2))

# lambda 函数
list1 = [1,2,3,4,5,6]

list2 = list(filter(lambda x:x % 2 == 0,list1))
list3 = list(map(lambda x:x ** 2 ,list1))
print(list2)
print(list3)

dict(filter(lambda x:x[1] > 5,[(1,2),(2,3),(3,4),(4,5),(5,6)]))
print(set(filter(lambda x:x[1] >= 5,[(1,2),(2,3),(3,4),(4,5),(5,6)])))


txt = open("开放平台.txt","r",encoding="utf-8")
print(txt.readlines())
txt.close()


with open("开放平台.txt","r",encoding="utf-8") as f :
    print(f.readlines())

content = [
    ['姓名','年龄'],
    ['章三','23'],
    ['章四','24'],
    ['章王','25']
]

with open("测试csv.csv",mode= 'w' ,encoding="utf-8") as f:
    csv.writer(f).writerows(content)

try:
    with open("测试csv.csv",mode= 'r' ,encoding="utf-8") as f:
        read = csv.reader(f)
        print(next(read))
        for i, e in enumerate(read, 1):
            print(f" {i} : {e},({e[0]},{e[1]})")
finally:
    if os.path.exists("测试csv.csv"):
        os.remove("测试csv.csv")

re_data = '的课你莫耳机你的饭3,3的卡分,1,45你'
if re.search(r'\b(\d{1,3}\.\d{1,3})\b',re_data):
    print(re.search(r'\b(\d{1,3}\.\d{1,3})\b',re_data).group())
else:
    print("没有匹配")

date = datetime.datetime.now()
print(date)
date1 = date + datetime.timedelta(days=1)
# 判断date1 的类是什么
print(type(date1))
diff = date1 - date
print(type(diff))
print(diff , diff.total_seconds())
print(date.strftime("%Y-%m-%d %H:%M:%S"))

#视频三 练习作业
def inputGrade(prompt):
    while True:
        try:
            grade_str = input(prompt)
            grade =float(grade_str)
            if grade >= 0 and grade <= 100:
                return grade
            else:
                print("输入的数字格式有误,请输入0-100")
        except ValueError:
            print("输入的数字格式有误,请输入0-100")

def calculate(scores_list):
    return sum(scores_list) / len(scores_list)

def determine(avg):
    if avg >= 60:
        return "c级"
    elif avg >= 80:
        return "b级"
    elif avg >= 90:
        return "a级"
    else:
        return"不及格"


name = input("请输入你的名字：")
num = 3
score_list = []
for i in range(num):
    yuwen = inputGrade(f"请输入第{i+1}门成绩：")
    score_list.append(yuwen)

avg = calculate(score_list)
level = determine(avg)
print(f"{name} 的各科成绩是{score_list},平均分:{avg:.2f},评级:{level}")


