class Car:
    wheel = 4 # クラス変数（クラス内で共通の変数）
    def __init__(self, iro, sokudo): # __init__はコンストラクタ（一番最初に呼び出される）
        self.color = iro
        self.speed = sokudo
    
    def value(self, kati): #クラス内の第一引数はselfにする慣習
        print(f"この車の価値は{kati}です") #selfをつける⇒クラス内どこでも使える（変数のスコープが広い）

my_car = Car("red", 100) # インスタンス化 （初期化メソッドで書いた引数のself以降を指定）

my_car.value("300万円") # メソッドの呼び出し

print(my_car.color) # インスタンス変数の呼び出し
print(my_car.speed)

print(Car.wheel) # クラス変数の呼び出し
print(my_car.wheel) # インスタンスからもクラス変数を呼び出せる

class Bus(Car): # 継承（Carクラスを継承してBusクラスを作成）
    def __init__(self, iro, sokudo, ninsuu):
        super().__init__(iro, sokudo) # 親クラスのコンストラクタを呼び出す (selfは不要)
        self.capacity = ninsuu # Busクラスで新たに追加したインスタンス変数

    def value(self, kati): # メソッドのオーバーライド
        print(f"このバスの価値は{kati}です")

bus = Bus("blue", 80, 50) # Busクラスのインスタンス化
bus.value("500万円") # Busクラスのメソッドを呼び出し
my_car.value("300万円") # Carクラスのメソッドを呼び出し(メソッドのオーバーライドでは親クラスのメソッドは変わらない)要はメソッドの名前そろえただけ