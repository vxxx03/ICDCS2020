import sys
import tensorflow as tf
from tensorflow import keras
import numpy as np
from scipy import stats

train_data=[]
train_labels=[]
test_data=[]
test_labels=[]

N=128

dataset_dir='./dataset/'
dataset_name=['xbee']

if len(sys.argv)>1:
    dataset_name[0]=sys.argv[1]

for dataset in dataset_name:
    lines=open(dataset_dir+dataset+'_normal.txt','r').readlines()
    for i,line in enumerate(lines):
        strlist=np.array(line.split())
        if i%5==0:
            test_data.append(strlist.astype(np.float))
            test_labels.append(0)         
        else:
            train_data.append(strlist.astype(np.float))
            train_labels.append(0)

for dataset in dataset_name:
    lines=open(dataset_dir+dataset+'_attack.txt','r').readlines()
    for i,line in enumerate(lines):
        strlist=np.array(line.split())
        if i%5==0:
            test_data.append(strlist.astype(np.float))
            test_labels.append(1)         
        else:
            train_data.append(strlist.astype(np.float))
            train_labels.append(1)


train_labels=np.array(train_labels)
train_data=np.array(train_data)
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=0,
                                                        padding='post',
                                                        maxlen=N)
test_labels=np.array(test_labels)
test_data=np.array(test_data)
test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                        value=0,
                                                        padding='post',
                                                        maxlen=N)
#train_data=train_data/255.0
#test_data=test_data/255.0
train_data=train_data.reshape((-1,N,1))/255.0
test_data=test_data.reshape((-1,N,1))/255.0

print('Dataset loaded.')
#########################################################################################


model=keras.models.load_model(dataset_name[0]+'_cnn_model.h5')

# output layer weights by InF-FS
intermediate_layer_model = keras.models.Model(inputs=model.input,
                                 outputs=model.layers[13].output)
intermediate_output = intermediate_layer_model.predict(train_data)

fDist=np.transpose(intermediate_output)
std=np.std(fDist,axis=1)
#print(std)

sigma=np.zeros([50,50])
spear=np.zeros([50,50])
for i in range(50):
    for j in range(50):
        sigma[i][j]=max(std[i],std[j])
        spear[i][j]=1-np.abs(stats.spearmanr(fDist[i],fDist[j])[0])
A = 0.5 * sigma + 0.5 * spear
#print(A)
eigen=np.linalg.eig(A)[0]
rho=np.max( np.abs( eigen ) )
S=np.linalg.inv(np.eye(50)-0.9/rho*A)-np.eye(50)
initial_score=np.sum(S,axis=1)
#print(initial_score)

#########################################################################################

# nisp
nnweights=[]
for i in [0,2,4,6,8,11,13,15]:
        extracted_weights=model.layers[i].get_weights()[0]
        nnweights.append(np.abs(extracted_weights))

nnscores=[]
for i in [0,2,4,6,8,11,13,15]:
    nnscores.append(np.zeros(model.layers[i].output_shape[1:]))

# initial weights
nnscores[6]=nnweights[7].reshape(model.layers[13].output_shape[1:])
#nnscores[6]=initial_score

# propagation in FC
nnscores[5]=np.dot(nnweights[6],nnscores[6])
temp=np.dot(nnweights[5],nnscores[5])
nnscores[4]=temp.reshape(model.layers[8].output_shape[1:])

# propagation in Conv
for i in [3,2,1,0]:
    for j in range(nnscores[i+1].shape[0]):
        nnscores[i][j] = nnscores[i][j] + np.dot(nnweights[i+1][0],nnscores[i+1][j])
        nnscores[i][j+pow(2,i+1)]= nnscores[i][j+pow(2,i+1)] + np.dot(nnweights[i+1][1],nnscores[i+1][j])

# propagation to inputs
cscore=np.zeros(N)
for j in range(nnscores[0].shape[0]):
    cscore[j] = cscore[j] + np.dot(nnweights[0][0],nnscores[0][j])[0]
    cscore[j+1]= cscore[j+1] + np.dot(nnweights[0][1],nnscores[0][j])[0]

# neuron -> substring
ssscore=[]
for i in range(5):
    ssscore.append(np.sum(nnscores[i],1))


# output
f=open(dataset_name[0]+'_importance_scores.txt','w')
scores=[]
for i in [cscore, ssscore[0], ssscore[1], ssscore[2], ssscore[3]]:
    for j in i:
        f.write(str(j)+' ')
    f.write('\n')
    scores.append(i)

######################################################
field_amount = 2
t=2
field_length = pow(2,t)

if len(sys.argv)>2:
    field_amount=int(sys.argv[2])
if len(sys.argv)>3:
    t=int(sys.argv[3])
field_length = pow(2,t)
# dynamic programming to select optimal fields

def DP(amount,position):
    global field_length,t
    if amount==1 or field_length*(amount-1)>position:
        return [ [position] , scores[t][position] ]
    else:
        candidate_scores=[]
        candidate_records=[]
        for p in range(position-field_length+1):
            candidate=DP(amount-1,p)
            candidate_scores.append(candidate[1])
            candidate_records.append(candidate[0])
        temp=np.argmax(np.array(candidate_scores))
        new_record = candidate_records[temp]
        new_record.append(position)
        new_score = candidate_scores[temp] + scores[t][position]
        return [ new_record, new_score]

candidate_scores=[]
candidate_records=[]
for p in range(N-field_length+1):
    candidate=DP(field_amount,p)
    candidate_scores.append(candidate[1])
    candidate_records.append(candidate[0])
temp=np.argmax(np.array(candidate_scores))

best_fields=candidate_records[temp]

# output
fp4=open(dataset_name[0]+'_P4_definition.txt','w')
fp4.write('header intrusion_detection {\n')
if best_fields[0]>0:
    fp4.write('    bit<'+str(best_fields[0])+'> padding0;\n')
for i in range(len(best_fields)):
    if i>0 and best_fields[i]>best_fields[i-1]+field_length:
        fp4.write('    bit<'+str(best_fields[i]-best_fields[i-1]-field_length)+'> padding'+str(i)+';\n')
    fp4.write('    bit<'+str(field_length)+'> matching'+str(i)+';\n')
fp4.write('}')
fp4.close()