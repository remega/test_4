import random

count = -1
val = []
for i in range(0,58):
    count += 1
    val.append(count)
val_A  = random.sample(val, 29)
for i in val_A:
    val.remove(i)
val_B = val

print(val_A,val_B)
