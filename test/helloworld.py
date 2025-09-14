def a(a) :
     print(f"a :{a}")


c = b = 4

# print(f"{ not c< 0 and  not b< 0}")

lista = [1,"2",2.3]
listb = [ x ** 3 for x in range(10) if x % 2 == 0 ]
print(listb)

for i in range(10):
    if i % 2 == 0:
        print(i ** 2)

# for e in lista:
#     print(e)

trup = (2,"3",True)
# for i, e in enumerate(trup):
#     print(i, e)
print(trup)

contents = { 'id' :"1","name":"张三",'phone' :"1234"}
contents2 = { 'id' :"1","name":"李四",'phone' :"1234"}

elment = {1:contents,"2":2, c:True}
print(elment.keys())
print(elment.values())
elment[1] = contents2
print(elment.values())
print(elment.items())

set = {1,2,3,4,5,6,7,8,9,1}
set1 = {2,3}
print(len(set))

#第二天
score = 80
if score >= 90:
    print("优秀")
elif score >= 80:
    print("良好")
elif score >= 70:
    print("及格")
elif score >= 60:
    print("及格")
else:
    print("不及格")

isTre =False

Tres = 0.11 if isTre else 0.32
print(f"{Tres * 100:.3f}")

for i, e in enumerate(contents2, start=1):
    print(i, contents2.get(e))

names = ["张三","李四","王五"]
ages = (18,19,20)
schools = ("1","2","3")
for name, age , school in zip(names, ages,schools):
    print(name, age,school)

task = 3
t = 0
trys = 3

while task > 0 and t < trys :
    print(f"任务还剩{task}, 尝试:{t+1}")
    t += 1
    task -= 1
else:
    if(task > 0):
        print("任务失败")
    else:
        print("任务完成")

