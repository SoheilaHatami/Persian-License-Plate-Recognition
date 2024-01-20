# %%
#Section 1
import os

root='LPDdataset/'
rate_train=0.8
f1=open('train.txt','w')
f2=open('valid.txt','w')
a=[root+i for i in os.listdir(root) if i[-1]!='t']
train_num=int(len(a)*rate_train)
for i in range(len(a)):
  if i<train_num:
    f1.write(a[i]+'\n')
  else:
    f2.write(a[i]+'\n')
f1.close()
f2.close()

# %%
# Section 2
root='LPDdataset/'
f=open('names.txt','w')
a='License Plate'
f.write(a)
f.close()

# %%
# Section 3
root='LPDdataset/'
f=open('data.txt','w')
a='classes=1'
b='train=train.txt'
c='valid=valid.txt'
d='names=names.txt'
e='backup=backup'
f.write(a+'\n'+b+'\n'+c+'\n'+d+'\n'+e)
f.close()

# %%
# Section 4
root='IranianLicensePlateDataset/'
f=open('config.txt','w')
a=''
f.write(a)
f.close()



