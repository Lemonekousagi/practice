str1 = 'Hello, {}!'
str2 = 'Python'
print(str1.format(str2))  # formatメソッドを使って文字列を結合して表示

list = [1, 2, 3]
for i, num in enumerate(list):
    print(i, num)  # リストの要素とインデックスを表示

list = [1, 2, 3]
for num in list:
    print(num)  # リストの要素を表示

dict = {}
dict['apple'] = 'りんご'
print(dict)  # 辞書を表示

dict = {'apple': 'りんご', 'orange': 'みかん', 'banana': 'バナナ'}
print(dict.keys())  # キーを出力

str = 'a b c'
list = str.split()  # スペースで区切ってリストに変換
print(list)  # リストを表示

list = ['a', 'b', 'c']
str = ' '.join(list)  # スペースで要素を連結
print(str)  # 文字列を表示