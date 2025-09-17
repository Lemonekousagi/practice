add = lambda x, y: print(x + y) # 無名関数 lambda 引数 : 戻り値
add(1, 2)

a = ("aiueo", 1, 2, 3)
print(a[-1])

def greeting(*args): # 可変長引数
    for s in args:
        print(s)
greeting("Hello", "Python", 2)

dict = {'apple': 'りんご', 'orange': 'みかん', 'banana': 'バナナ'}
print(dict.items())  # キーと値のペアを出力
for kagi, value in dict.items(): # 辞書のキーと値を同時に取得
    print(f'{kagi}は{value}です')

def func(*args, **kwargs): # 可変長引数とキーワード引数
    for s in args:
        print(s)
    for k, v in kwargs.items():
        print(f'{k}は{v}です')
func("Hello", "Python", 3, b=2)