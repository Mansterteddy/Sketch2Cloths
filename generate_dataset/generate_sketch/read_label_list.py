import scipy.io

label_list_filenmae = './label_list.mat'

label_data = scipy.io.loadmat(label_list_filenmae)
print("label_data: ", label_data)

'''
Ans:
'null' 0
'accessories' 1
'bag' 2
'belt' 3
'blazer' 4
'blouse' 5
'bodysuit' 6
'boots' 7
'bra' 8
'bracelet' 9
'cape' 10
'cardigan' 11
'clogs'] 12
'coat' 13
'dress' 14
'earrings' 15
'flats' 16
'glasses' 17
'gloves' 18
'hair' 19
'hat' 20
'heels' 21
'hoodie' 22
'intimate' 23
'jacket' 24
'jeans' 25
'jumper' 26
'leggings' 27
'loafers' 28
'necklace' 29
'panties' 30
'pants' 31
'pumps' 32
'purse' 33
'ring' 34
'romper' 35
'sandals' 36
'scarf' 37
'shirt' 38
'shoes' 39
'shorts' 40
'skin' 41
'skirt' 42
'sneakers' 43
'socks' 44
'stockings' 45
'suit' 46
'sunglasses' 47
'sweater' 48
'sweatshirt' 49
'swimwear' 50
't-shirt' 51
'tie' 52
'tights' 53
'top' 54
'vest' 55
'wallet' 56
'watch' 57
'wedges' 58

'''