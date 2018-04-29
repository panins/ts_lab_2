a=b'--'
print((lambda x: '0.0' if str(x,encoding = "utf-8")=='--' else x)(a))
print(str(a))

b = lambda x:x
print(b(1))