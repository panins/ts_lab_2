from scipy.stats import norm
from math import e,sqrt


def prob_dist(price):
    price_up = round(price*1.1,2)
    if price_up/price >1.1:
        price_up-=0.01
    price_down = round(price*0.9,2)
    if price_down/price <0.9:
        price_down+=0.01
    price_prob_list = []
    for x in range(int(price_down*100),int(price_up*100),1):
        price_prob_list.append([float(x)/100,
                                norm.cdf(float(x)/100+0.004,price,sqrt(price))
                                -norm.cdf(float(x)/100-0.005,price,sqrt(price))])
    total_prob = sum(list(zip(*price_prob_list))[1])
    print(total_prob)
    for _ in price_prob_list:
        _.append(_[1]/total_prob)
    return price_prob_list

price = 100.
a = prob_dist(price)
for x,y,z in a:
    print(x,y,z)
k,j = 0,0
k = sum(list(zip(*a))[1])
j = sum(list(zip(*a))[2])
print(k,j)
print(norm.cdf(89.995,price,sqrt(price))-norm.cdf(90.004,price,sqrt(price)))