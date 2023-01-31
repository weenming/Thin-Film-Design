
def timer(func):
    import time
    def func_wrapper():
        t = 0
        n = 1000
        for _ in range(n):
            start = time.time()
            func()
            end = time.time()
            t += end - start
        t /= n
        print("avg time per call:", t)
    return func_wrapper


if __name__ == "__main__":
    @timer
    def f():
        for _ in range(10000):
            a = 9345780.
            b = a
    
    @timer
    def f_ls():
        for _ in range(10000):
            a = []
            a.append(9345780.)
            b = a[0]
    
    f()
    f_ls()

