a = "かきくけこ"
print(f'あいうえお \n{a}')

print('あいうえお \n{}'.format(a))
print('aiueo{1}{0}'.format('123', '456'))
print('aiueo{a}{b}'.format(a='123', b='456'))

key = {'apple': 'りんご', 'orange': 'みかん', 'banana': 'バナナ'}
print('apple is {apple}, orange is {orange}, banana is {banana}'.format(**key))

