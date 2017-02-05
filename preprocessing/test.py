class test:
    def __init__(self):
        self.x = 'x'
        self.y = 'y'
    def call_other_self_fn(self):
        self.print_stuff()
    def print_stuff(self):
        print self.x
        print self.y

if __name__ == '__main__':
    x =  test()
    x.call_other_self_fn()
