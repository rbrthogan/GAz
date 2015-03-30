__author__ = 'Robert Hogan'

from operator import mul
import numpy as np
class hash_table:
   table=[]
   bucketNumber=0

   def __init__(self, buckets=10007):
      self.bucketNumber = buckets
      hash_table.table=[[[],[],[]] for _ in xrange(buckets)]

   def key_gen(self,individual):
        x=np.array(individual)

        key=np.int(np.sum(x)*np.prod(x[x!=0],dtype=np.float32))

        return key%self.bucketNumber


   def insert(self,input):
        individual=input[0]
        key=hash_table.key_gen(self,individual)

        hash_table.table[key][0].append(individual)
        hash_table.table[key][1].append(input[1])
        hash_table.table[key][2].append(input[2])

        return

   def retrieve(self,individual):
       key=hash_table.key_gen(self,individual)
       if len(hash_table.table[key][0])==0:
            return
       if individual in hash_table.table[key][0]:
            i=hash_table.table[key][0].index(individual)
            return  hash_table.table[key][1][i],hash_table.table[key][2][i]
       else:
            return

   def display_table(self):
        i=0
        for row in hash_table.table:
            if len(row[0])!=0:
                print i,row[:2]
            i+=1

   def size(self):
       occupancies=[len(hash_table.table[i][0]) for i in range(self.bucketNumber)]
       return sum(occupancies)

   def bucket_occupancies(self):
       occupancies=[len(hash_table.table[i][0]) for i in range(self.bucketNumber)]
       return occupancies