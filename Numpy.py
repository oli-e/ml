#!/usr/bin/env python
# coding: utf-8

# # Basics of numpy

# ### I'm learning

# In[2]:


import numpy as np

a = np.array([1,2,3])
a


# In[9]:


b = np.zeros([3])
b


# In[15]:


c = a + b
c


# In[13]:


a*c


# In[17]:


np.not_equal(a,c)


# In[21]:


# Get shape
a.shape


# In[25]:


a.nbytes


# In[27]:


a = np.array([[1,2,3,6,5,4,8],[1,2,3,5,4,6,9]])
a


# In[30]:


a[1,6]


# In[31]:


a[:,2]


# In[33]:


a[1,:]


# In[34]:


# Getting fancy [startindex:endindex:stepsize]
a[0, 1:6:2]


# In[39]:


a[1,4]=5
print(a)

a[:,2] = [1,2]
print(a)


# In[52]:


# Get specific element form 3d example = (work outside in)

three_d = np.array([[[1,2],[5,2]],[[1,6],[7,8]]])
print(three_d)


# In[42]:


np.full((7,2,5), 69)


# In[43]:


np.full(a.shape, 6.9)


# In[46]:


np.full_like(c,6, dtype='int32')


# In[47]:


np.random.rand(4,2)


# In[54]:


np.random.random_sample(three_d.shape)


# In[109]:


np.random.randint(8, size=(3,3))


# In[58]:


np.identity(5)


# In[63]:


arr = np.array([[5,6,8]])
r1 = np.repeat(arr, 3, axis=0)
r1


# In[64]:


np.ones((5,5))


# In[79]:


# Task 1
output = np.ones((5,5))
nulls = np.zeros((3,3))
nulls[1,1]= 9

output[1:-1,1:-1]=nulls
print(output)


# ### Copying

# In[81]:


a = np.array([1,2,3])
b = a.copy()
b[2] = 110
print(b)
print(a)


# ## Linear Algebra

# In[82]:


a = np.full((2,3),7)

b = np.full((3,2), 2)

print(a)
print(b)


# In[83]:


# Matrix multiplication
np.matmul(a,b)


# In[85]:


# Find the determinant
c = np.identity(3)
np.linalg.det(c)


# ## Statistics

# In[87]:


stats = np.array([[1,2,3],[4,5,6]])
stats


# In[89]:


np.min(stats)


# In[91]:


np.min(stats, axis=1)


# In[92]:


np.min(stats, axis=0)


# In[93]:


np.max(stats, axis=1)


# In[95]:


np.max(stats)


# In[96]:


np.max(stats, axis=0)


# In[97]:


np.sum(stats)


# # reorganizing arrays

# #### reshape

# In[99]:


before = np.array([[1,2,3,4],[5,6,7,8]])
print(before)

after = before.reshape((8,1))
print(after)

after2 = before.reshape((2,2,2))
print(after2)


# #### vstack

# In[101]:


v1 = np.array([1,2,3,4])
v2 = np.array([5,6,7,8])

np.vstack([v1, v2, v1, v2, v2,v2])


# #### hstack

# In[107]:


h1 = np.ones((2,4))
h2 = np.zeros((2,2))

np.hstack((h1,h2))


# #### miscellaneous

# ##### Load data from file

# In[ ]:


filedata = np.genfromtxt('*.txt', delimiter=',' )
filedata = filedata.astype('int32')


# ##### boolean masking and advanced indexing

# In[113]:


arr = np.random.randint(101, size=(4,9))
arr


# In[115]:


arr [arr > 70]


# In[ ]:





# In[ ]:





# In[ ]:




