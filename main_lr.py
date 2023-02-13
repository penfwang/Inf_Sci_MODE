from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)
from itertools import chain
import array
import random
import json
import numpy as np
from math import sqrt
from deap import algorithms
from deap import base
from deap import benchmarks
from deap.benchmarks.tools import diversity, convergence, hypervolume
from deap import creator
from deap import tools
import alg_NSGA2
from math import sqrt
from operator import mul
from functools import reduce
import numpy.matlib
import operator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from collections import Counter
import geatpy as ea
import math,time,sys,saveFile
from minepy import MINE
import diverse
from used_filter import fisher_score


def uniform(low, up, size=None):####generate a matrix of the range of variables
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

def findindex(org, x):
    result = []
    for k,v in enumerate(org): #k和v分别表示org中的下标和该下标对应的元素
        if v == x:
            result.append(k)
    return result

def find_all(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1: return
        yield start
        start += len(sub)


def fit_train(x1, train_data):
    x = np.zeros((1,len(x1)))
    for ii in range(len(x1)):
        x[0,ii] = x1[ii]
    x = random.choice(x)
    x = 1 * (x >= 0.6)
    if np.count_nonzero(x) == 0:
        f1 = 1###error_rate
        f2 = 1
    else:
     x = "".join(map(str, x))  # transfer the array form to string in order to find the position of 1
     value_position = np.array(list(find_all(x, '1'))) + 1  # cause the label in the first column in training data
     value_position = np.insert(value_position, 0, 0)  # insert the column of label
     tr = train_data[:, value_position]
     #clf = KNeighborsClassifier(n_neighbors = 5)
     clf = LogisticRegression()
     scores = cross_val_score(clf, tr[:,1:],tr[:,0], cv = 5)
     f1 = np.mean(1 - scores)
     f2 = (len(value_position)-1)/(train_data.shape[1] - 1)
     # f2 = len(value_position) - 1
    return f1, f2


def fit_train_group(x1, train_data, space):
    x = np.zeros((1,len(x1)))
    for ii in range(len(x1)):
        x[0,ii] = x1[ii]
    x = random.choice(x)
    x = 1 * (x >= 0.6)
    if np.count_nonzero(x) == 0:
        f1 = 1###error_rate
        f2 = 1
    else:
     x = "".join(map(str, x))
     value_position = np.array(list(find_all(x, '1')))
     training_data = train_data[:,1:]
     training_label = train_data[:,0]
     value_position = [space[i] for i in value_position]
     tr =  training_data[:, value_position]
     #clf = KNeighborsClassifier(n_neighbors = 5)
     clf = LogisticRegression()
     scores = cross_val_score(clf,tr ,training_label, cv = 5)
     f1 = np.mean(1 - scores)
     f2 = len(value_position)/(train_data.shape[1] - 1)
     # f2 = len(value_position) - 1
    return f1, f2



def kNNClassify(newInput, dataSet, labels, k):
    numSamples = dataSet.shape[0]   
    diff = np.tile(newInput, (numSamples, 1)) - dataSet  
    squaredDiff = diff ** 2  
    squaredDist = squaredDiff.sum(axis = 1)   
    distance = squaredDist ** 0.5 
    sortedDistIndices = distance.argsort()
    classCount = {}
    for i in range(k):
        voteLabel = labels[sortedDistIndices[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    sorted_ClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_ClassCount[0][0]




def more_confidence(EXA, index_of_objectives):
    a = 0.6
    cr = np.zeros((len(index_of_objectives),1))
    for i in range(len(index_of_objectives)):###the number of indexes
        temp = 0
        object = EXA[index_of_objectives[i]]
        for ii in range(len(object)):###the number of features
           b = object[ii]
           if b > a:  con = (b - a) / (1 - a)
           else:      con = (a - b) / (a)
           temp = con + temp
        cr[i,0] = temp
    sorting = np.argsort(-cr[:,0])####sorting from maximum to minimum
    index_one = index_of_objectives[sorting[0]]
    return index_one


def delete_duplicate(EXA):####list
    EXA1 = []
    EXA_array = np.array(EXA)
    all_index = []
    for i0 in range(EXA_array.shape[0]):
       x = 1 * (EXA_array[i0,:] >= 0.6)
       x = "".join(map(str, x))  # transfer the array form to string in order to find the position of 1
       all_index.append(x)##store all individuals who have changed to 0 or 1
    single_index = set(all_index)####find the unique combination
    single_index = list(single_index)####translate it's form in order to following operating.
    for i1 in range(len(single_index)):
       index_of_objectives = findindex(all_index, single_index[i1])##find the index of each unique combination
       if len(index_of_objectives) == 1:
          for i2 in range(len(index_of_objectives)):
             EXA1.append(EXA[index_of_objectives[i2]])
       else:####some combination have more than one solutions.here may have duplicated solutions
           #index_one = more_confidence(EXA, index_of_objectives)
           index_one = index_of_objectives[0]
           EXA1.append(EXA[index_one])
    return EXA1


def mutDE4(y, r1, r2, r3, f):###DE/current-to-rand/1
    k_f = random.random()
    for i in range(len(y)):
        y[i] = y[i]+k_f*(r1[i]-y[i])+ f*(r2[i]-r3[i])
    return y



def mutDE_binary_yong(y, a, b, c, f):
    y_new = toolbox.clone(y)
    pa = 0.005
    temp = []
    temp.append(a)
    temp.append(b)
    temp.append(c)
    temp.append(y)
    PF = np.array([ind.fitness.values for ind in temp])
    [levels1, criLevel1] = ea.indicator.ndsortDED(PF, 1)
    x1 = 1 * (levels1 == 1.0)
    x1 = "".join(map(str, x1))
    index_non = np.array(list(find_all(x1, '1')))
    if len(temp)-1 not in index_non:###the target solution is the dominated one
        for i in range(len(y)):###
            c_value = pa
            if c_value < random.uniform(0, 1):
                y_new[i] = temp[index_non[0]][i]
            else:
                y_new[i] = 1- temp[index_non[0]][i]
    else:###the target solution dominates others
        for i in range(len(y)):## c = min(1,f*(a[i]^b[i])+pa)pop_new = diverse.find_and_modify_duplicated(offspring,pop_new)
            xor_value = (1-a[i])*b[i] + a[i]*(1-b[i])
            c_value = min(1,f*xor_value+pa)
            if c_value < random.uniform(0, 1):
                y_new[i] = temp[index_non[0]][i]
            else:
                y_new[i] = 1- temp[index_non[0]][i]
    return y_new



##cxBinomial(offspring[ii],y_new,0.5)###crossover
def cxBinomial(x, y, cr):#####binary crossover
    y_new = toolbox.clone(y)
    size = len(x)
    index = random.randrange(size)
    for i in range(size):
        if i == index or random.uniform(0, 1) <= cr:
            # y_new[i] = y[i]
            y_new[i] = y[i]
        else:
            y_new[i] = x[i]
    return y_new


def first_level_nondominant(x1):
    x = np.array([ind.fitness.values for ind in x1])
    [levels, criLevel] = ea.indicator.ndsortDED(x, 1)
    x = 1 * (levels == 1.0)
    x = "".join(map(str, x))
    index = np.array(list(find_all(x, '1')))
    return PF


def first_nondominated(pop):
    PF = np.array([ind.fitness.values for ind in pop])
    [levels1, criLevel1] = ea.indicator.ndsortDED(PF, 1)
    x1 = 1 * (levels1 == 1.0)
    x1 = "".join(map(str, x1))
    index_non = np.array(list(find_all(x1, '1')))
    pop_non = [pop[i] for i in index_non]
    return pop_non



def continus2binary(x):
    for i in range(len(x)):
            if x[i] >= 0.6:
                x[i] = 1.0
            else:
                x[i] = 0.0
    return x


def hamming_distance(s, s0):
    """Return the Hamming distance between equal-length sequences"""
    s1 = toolbox.clone(s)
    s2 = toolbox.clone(s0)
    s3 = continus2binary(s1)
    s4 = continus2binary(s2)
    if len(s3) != len(s4):
        raise ValueError("Undefined for sequences of unequal length")
    return sum(el1 != el2 for el1, el2 in zip(s3, s4))


def xor(a,b):
    xor_value = (1-a)*b+ a*(1-b)
    return xor_value

def euclidean_distance(x1,x2):
    s1 = toolbox.clone(x1)
    s2 = toolbox.clone(x2)
    s1 = np.array(s1)
    s2 = np.array(s2)
    temp = sum((s1-s2)**2)
    temp1 = np.sqrt(temp)
    return temp1


def get_dis(off):
    dis = np.zeros((len(off), len(off)))
    off_fit = np.array([ind.fitness.values for ind in off])
    norm_fit = normlize(off_fit)
    for i in range(len(norm_fit)):
        for j in range(len(norm_fit)):
            dis[i, j] = np.sqrt(np.sum(np.square(norm_fit[i]-norm_fit[j])))
    return dis


def normlize(data):
    data = np.array(data)
    for i in range(data.shape[1]):
      column = data[:,i]
      mini = min(column)
      maxi = max(column)
      if mini != maxi:
         for j in range(len(column)):
            data[j,i] = (column[j] - mini) / (maxi - mini)
    return data


def get_cos_dis(off):
    dis = np.zeros((len(off), len(off)))
    matrix_cos_simi = np.zeros((len(off), len(off) - 1))
    for i in range(len(off)):
        x = np.array(off[i])
        pop_temp1 = [m for m in off]
        pop_temp1.remove(off[i])
        for j in range(len(pop_temp1)):
            y = np.array(pop_temp1[j])
            matrix_cos_simi[i, j] = np.dot(x, y) / (np.linalg.norm(x) * (np.linalg.norm(y)))
            # matrix_cos_simi[i, j] = sum(map(lambda i, j: abs(i-j),x,y))##
    cos_dis = [np.max(matrix_cos_simi[i, :]) for i in range(len(matrix_cos_simi))]  # the smaller the better
    for i in range(len(cos_dis)):
        for j in range(len(cos_dis)):
            dis[i, j] = np.sqrt(np.sum(np.square(cos_dis[i]-cos_dis[j])))
    return dis



creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))####minimise two objectives
creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)
toolbox = base.Toolbox()

def main(seed,x_train):
    random.seed(seed)
    mine = MINE(alpha=0.6, c=15, est="mic_approx")
    NDIM = x_train.shape[1] - 1
    BOUND_LOW, BOUND_UP = 0.0, 1.0
    NGEN = 100###the number of generation
    if NDIM < 300:
        MU = NDIM  ####the number of particle
    else:
        MU = 300  #####bound to 300
    Max_FES = MU * 100
    #std_value = []
    training_data = x_train[:,1:]
    #for i_in in range(NDIM):  ####just know the dimensinal number
       #scc = np.std(training_data[:,i_in])
       #std_value.append(scc)
    #std_value = np.array(std_value)
    #saved_index = np.argwhere(std_value > 0)
    #spaces  = [i[0] for i in saved_index]
    mic_value = []
    for i_in in range(NDIM):  ####just know the dimensinal number
      mine.compute_score(training_data[:,i_in], x_train[:,0])
      mic_value.append(mine.mic())
    #mic_value = np.array(mic_value)
    score = fisher_score(training_data, x_train[:,0])
    # mic_value = np.array(mic_value)
    mic_value = [(i-np.min(mic_value))/(np.max(mic_value)-np.min(mic_value)) for i in mic_value]
    score = [(i-np.min(score))/(np.max(score)-np.min(score)) for i in score]
    # print(mic_value)
    # print(score)
    mic_value = [0.5*mic_value[iii] + 0.5*score[iii] for iii in range(len(score))]
    rou0 = 0.1 * max(mic_value)
    saved_index = np.argwhere(mic_value > rou0)
    spaces = [i[0] for i in saved_index]
    unique_number = []
    hy = []
    toolbox.register("attr_bool", random.randint, BOUND_LOW, BOUND_UP)  #####dertemine the way of randomly generation and gunrantuu the range
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool,len(spaces))  ###fitness
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)  ##particles
    toolbox.register("evaluate", fit_train_group, train_data= x_train,space = spaces)
    toolbox.register("select", alg_NSGA2.selNS)#
    toolbox.register("select1", diverse.selection_diversity) 
    offspring = toolbox.population(n=MU)
    offspring = diverse.constraint(offspring)
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)#####toolbox.evaluate = fit_train
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    fit_num = len(offspring)
    ######################the first need to store all, otherwise there is not enough solutions to enter into next generation
    offspring = toolbox.select(offspring, len(offspring))
    pop_non = first_nondominated(offspring)##############
    pop_surrogate = delete_duplicate(offspring)
    PF1 = np.array([ind.fitness.values for ind in pop_non])
    hy.append(hypervolume(pop_non, [1, 1]))
    unique_number.append(len(pop_surrogate))
    dis = get_dis(offspring)
    #dis = get_cos_dis(offspring)
    for gen in range(1, NGEN):
        pop_new = toolbox.clone(offspring)
        for ii in range(len(offspring)):  #####upate the whole population
####################################get the nbest from the local neighbood
            #neighbood_size = int(0.2*MU)
            #if neighbood_size< 4:
                #neighbood_size = 4
            #ss_i = np.argsort(dis[ii, :])  ###the index in the whole population
            #neighbood  = [offspring[t] for t in ss_i[1:neighbood_size]]
            #index_non1 = first_nondominated(neighbood)##t
            #temp_individual_fitness = np.array([ind.fitness.values for ind in index_non1])
            #index1 = np.argsort(temp_individual_fitness[:, 0])  ###error
            #index1 = random.choice(index1)
            #nbest = index_non1[index1]
            ###################################get the nbest from the local neighbood
            niche_index = diverse.niche_introduce(dis,ii)
            niche_index.remove(ii)
            if random.uniform(0,1) < 0.8:
                #offspring1 = toolbox.clone(neighbood)
                offspring1 = [offspring[m] for m in niche_index]
            else:
                offspring1 = toolbox.clone(offspring)
                del (offspring1[ii])
            r1,r2,r3 = random.sample(offspring1, 3)####three uniuqe individuals
            y_new = mutDE_binary_yong(offspring[ii], r1, r2, r3, 0.5)###mutation
            pop_new[ii] = cxBinomial(offspring[ii],y_new,0.5)###crossover
            del pop_new[ii].fitness.values###delete the fitness
        #pop_new = diverse.find_and_modify_duplicated(offspring,pop_new)
        ##################################################
        invalid_ind = [ind for ind in pop_new if not ind.fitness.valid]
        fitne = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit1 in zip(invalid_ind, fitne):
            ind.fitness.values = fit1
        # Select the next generation population
        fit_num = fit_num + len(offspring)
        pop_surrogate.extend(delete_duplicate(pop_new))
        pop_surrogate = delete_duplicate(pop_surrogate)
        unique_number.append(len(pop_surrogate))
        pop_mi = pop_new + offspring
        pop1 = delete_duplicate(pop_mi)
        #offspring = toolbox.select(pop1, MU)
        offspring = toolbox.select1(pop1, MU)
        #if len(pop1) == MU:
            #offspring = pop1
        #elif len(pop1) > MU:
            #offspring = grid_dominance(pop1,15,MU)
        pop_non = first_nondominated(offspring)
        hy.append(hypervolume(pop_non, [1, 1]))
        dis = get_dis(offspring)
        #dis = get_cos_dis(offspring)
        if fit_num > Max_FES:
            break
    return offspring,unique_number,hy,spaces



if __name__ == "__main__":
    #seedtt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]###
    dataset_name = str(sys.argv[1])
    seed = str(sys.argv[2])
    folder1 = '/nesi/project/vuw03334/split_73' + '/' + 'train' + str(dataset_name) + ".npy"
    folder2 = '/nesi/project/vuw03334/split_73' + '/' + 'test' + str(dataset_name) + ".npy"
    x_train = np.load(folder1)
    x_test = np.load(folder2)
    start = time.time()
    random_seed = seed
    pop,unique_number,hy,spaces = main(seed,x_train)
    end = time.time()
    running_time = end - start
    pop_non = first_nondominated(pop)
    front_training = np.array([ind.fitness.values for ind in pop_non])
    EXA_array = np.array(pop_non)
    saveFile.saveAllfeature2(random_seed, dataset_name, EXA_array)
    saveFile.saveAllfeature3(random_seed, dataset_name, front_training)
    saveFile.saveAllfeature5(random_seed, dataset_name, unique_number)
    saveFile.saveAllfeature8(random_seed, dataset_name, hy)
    saveFile.saveAllfeature9(random_seed, dataset_name, spaces)
    saveFile.savetime(random_seed, dataset_name, running_time)
    print('End')
