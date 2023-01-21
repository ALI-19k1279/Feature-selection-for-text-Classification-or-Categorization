import array as arrr
from turtle import left
import numpy as np
def mergesort(llistt,rlistt):
    res=[]
    while len(llistt)!=0 and len(rlistt)!=0:
        if llistt[0]<rlistt[0]:
            res.append(llistt[0])
            llistt.remove(llistt[0])
        else:
            res.append(rlistt[0])
            rlistt.remove(rlistt[0])
            
    if len(llistt)==0:
        res=res+rlistt
    else :
        res=res+llistt
    return res
def arrayfunc(listt):
    if len(listt) <= 1:
      return listt

    temp=0
    # for i in range(len(listt)):
    #     for j in range(i+1,len(listt)):
    #         if listt[i] > listt[j]:
    #             temp=listt[i]
    #             listt[i]=listt[j]
    #             listt[j]=temp
    middle=len(listt)//2
    leftlistt=listt[:middle]
    rightlistt=listt[middle:]
    print(leftlistt)
    print(rightlistt)
    leftlistt=arrayfunc(leftlistt)
    rightlistt=arrayfunc(rightlistt)
    return list(mergesort(leftlistt,rightlistt))

def convertToRoman(number):
    num = [1, 4, 5, 9, 10, 40, 50, 90,
        100]
    sym = ["I", "IV", "V", "IX", "X", "XL",
        "L", "XC", "C"]
    i = 8
    while number:
        div = number // num[i]
        number %= num[i]
  
        while div:
            print(sym[i], end = "")
            div -= 1
        i -= 1


def merge_sort(unsorted_list):
   if len(unsorted_list) <= 1:
      return unsorted_list
# Find the middle point and devide it
   middle = len(unsorted_list) // 2
   left_list = unsorted_list[:middle]
   right_list = unsorted_list[middle:]

   left_list = merge_sort(left_list)
   right_list = merge_sort(right_list)
   return list(merge(left_list, right_list))

# Merge the sorted halves
def merge(left_half,right_half):
   res = []
   while len(left_half) != 0 and len(right_half) != 0:
      if left_half[0] < right_half[0]:
         res.append(left_half[0])
         left_half.remove(left_half[0])
      else:
         res.append(right_half[0])
         right_half.remove(right_half[0])
   if len(left_half) == 0:
      res = res + right_half
   else:
      res = res + left_half
   return res
unsorted_list = [64, 34, 25, 12, 22, 11, 90]


def checkprime(num):
    flag=True
    for i in range(2,num):
        if num%i==0:
            flag=False
            break
    if flag:
        print("prime")
    else:
        print("not prime")
    
if __name__=="__main__":
    listt = [19,2,31,45,6,11,121,27]
    arr2=[1,22,22,31,1]
    # print(listt[::-1])
    print(set(arr2))
    #checkprime(4)
    # print(merge_sort(unsorted_list))
    # number=int(input('enter'))
    # convertToRoman(number)