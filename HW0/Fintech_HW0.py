# read input integer n, output nth Fibonacci number
# if input is not a valid integer, output 0
try:
    n = int(input())
except ValueError:
    print(0)
    exit()
a, b = 0, 1
for _ in range(n):
    a, b = b, a + b
print(a)