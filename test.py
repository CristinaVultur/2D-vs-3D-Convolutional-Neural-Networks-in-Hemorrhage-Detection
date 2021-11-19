fruits = ["apple", "banana", "cherry", "kiwi", "mango"]

newlist = [x for x in fruits if "a" in x]

print(newlist)

newlist = [x for x in range(10) if x % 2 == 0 ]
print(newlist)

newlist = [x.upper() for x in fruits]
print(newlist)

newlist = [x if x!="banana" else "arange" for x in fruits]
print(newlist)
