class car:
    total = 0
    max_speed = 1000
    def __init__(self, year ,make, model):
        self.make = make
        self.model = model
        self.year = year
        self.odometer_reading = 0
        car.total += 1

    def __str__(self):
        return f"当前汽车:{self.year} {self.make} {self.model}"


    def drive(self, odometer):
        if odometer > 0:
            self.odometer_reading += odometer
            print(f"{self.year} {self.make} {self.model} 行驶了 {odometer} 公里,总公里数为 {self.odometer_reading}")
        else:
            print("请输入正确的里程数")
    @classmethod
    def get_total(cls):
        return f"一共有多少辆车: {cls.total}"

    @staticmethod
    def is_max_speed(speed):
         print(f"当前{speed}未超速" if speed <= car.max_speed else f"当前{speed}超速")

class dongfeng_car( car):
    def __init__(self, make, model, year):
        super().__init__(make, model, year)
        self.type = '东风'

    def __str__(self):
        return f"当前东风牌汽车:{self.year} {self.make} {self.model} {self.type}"