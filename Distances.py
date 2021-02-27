from math import sqrt
import matplotlib.pyplot as plt 
def euc_dist(a,b):
    return sqrt(sum((e1-e2)**2 for e1, e2 in zip(a,b)))

def man_dist(a,b):
    return sum(abs(e1-e2) for e1,e2 in zip(a,b))

def mink_dist(a,b,p):
    if(p==2): return euc_dist(a,b)
    if(p==1): return man_dist(a,b)
    else:
        return 0

def ham_dis(a,b):
    return sum(abs(e1 - e2) for e1,e2 in zip(a,b)) / len(a)

##row1 = [10,20,15,10,5]
##row2 = [12,24,18,8,7]
row1 = (1,3)
row2 = (2,4)
plt.subplot(1,2,1)
plt.plot(row1,row2,linestyle = '--')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.title('Euclidean distance')
plt.subplot(1,2,2)
row3 = (row1[0],row2[1])
print(row3)
print(row2)
plt.plot(row1,row2)
#plt.plot(row1,row3,linestyle = '--')
#plt.plot(row3,row2,linestyle = '--')
plt.show()

dist = euc_dist(row1, row2)
manh = man_dist(row1, row2)
mink = mink_dist(row1, row2, 1)
ham = ham_dis(row1, row2)


print(dist, manh, mink, ham)