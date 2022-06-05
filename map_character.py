import numpy as np

def map(word):
        if word == 0:
            return chr(1488)
        if word == 1:
            return chr(1506)
        if word == 2:
            return chr(1489)
        if word == 3:
            return chr(1491)
        if word == 4:
            return chr(1490)
        if word == 5:
            return chr(1492)
        if word == 6:
            return chr(1495)
        if word == 7:
            return chr(1499)
        if word == 8:
            return chr(1498)
        if word == 9:
            return chr(1500)
        if word == 10:
            return chr(1501)
        if word == 11:
            return chr(1502)
        if word == 12:
            return chr(1503)
        if word == 13:
            return chr(1504)
        if word == 14:
            return chr(1508)
        if word == 15:
            return chr(1507)
        if word == 16:
            return chr(1511)
        if word == 17:
            return chr(1512)
        if word == 18:
            return chr(1505)
        if word == 19:
            return chr(1513)
        if word == 20:
            return chr(1514)
        if word == 21:
            return chr(1496)
        if word == 22:
            return chr(1509)
        if word == 23:
            return chr(1510)
        if word == 24:
            return chr(1493)
        if word == 25:
            return chr(1497)
        if word == 26:
            return chr(1494)


def map_characters(uniques):
    # print(uniques)
    sentence = np.array(uniques)
    ascii = []
    for letter in sentence:
        ascii.append(map(letter))

    return ascii
