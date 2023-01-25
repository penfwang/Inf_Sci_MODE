from __future__ import division
import bisect
import math
import random
from itertools import chain
from operator import attrgetter, itemgetter
from collections import defaultdict
import numpy as np
import itertools
# from minepy import MINE
from deap import tools, base
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
toolbox = base.Toolbox()
import geatpy as ea


def get_all_index(lst,item):
    tmp = []
    tag = 0
    for i in lst:
        if i == item:
            tmp.append(tag)
        tag +=1
    return tmp

def find_all(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1: return
        yield start
        start += len(sub)

def findindex(org, x):
    result = []
    for k,v in enumerate(org): #k和v分别表示org中的下标和该下标对应的元素
        if v == x:
            result.append(k)
    return result


def ran(number):
    i = 0
    y = np.zeros((number))
    while i< number:
        y[i]=random.random()
        i = i+1
    return y


def add_delete(temp,p_add,p_delete,dim):
    y_add = random.random()
    y_delete = random.random()
    inter = random.randint(0,dim-1)
    if y_add < p_add[inter]:
            temp[inter] = 1
    if y_delete < p_delete[inter]:
            # temp[t2] = random.uniform(0, 0.6)
            temp[inter] = 0
    return temp

def get_whole_01(individuals):
    all_index = []
    individuals_array = np.array(individuals)  ####
    for i0 in range(individuals_array.shape[0]):
        x1 = 1 * (individuals_array[i0, :] >= 0.6)
        x1 = "".join(map(str, x1))  # transfer the array form to string in order to find the position of 1
        all_index.append(x1)  ##store all individuals who have changed to 0 or 1
    return all_index


def constraint(off):
    N = len(off)
    D = len(off[0])
    # print(N,D)
    if 3*N < D:####the number of selected features shod be smaller than 3N
       for i1 in range(N):
            for i2 in range(D):
                off[i1][i2] = 0
       sampled_int = random.sample(range(1,3*N),N)
       for i in range(len(sampled_int)):
           selected_features = sampled_int[i]
           sampled_position = random.sample(range(0, D), selected_features)
           for j in sampled_position:
               off[i][j] = 1
       return off
    else:
       return off



def find_and_modify_duplicated_new(old,pop_new,spaces,pop_non):#####use p_add and p_delete to add or remove features
    dup_index = []
    all_index = get_whole_01(old)
    unique_all_Index = set(all_index)
    unique_all_Index = list(unique_all_Index)
    for i in range(len(pop_new)):
        x = np.array(pop_new[i])
        x1 = 1 * (x >= 0.6)
        x2 = "".join(map(str, x1))
        mm = findindex(unique_all_Index,x2)
        if len(mm) == 0:
            unique_all_Index.append(x2)
        else:
            dup_index.append(i)
    if len(dup_index) != 0:
        matrix_01 = np.zeros((len(pop_non), len(pop_non[0])))
        EXA_array = np.array(pop_non)
        for i0 in range(EXA_array.shape[0]):
            x = 1 * (EXA_array[i0, :] >= 0.6)
            matrix_01[i0, :] = x
        p_add = matrix_01.mean(axis=0) + 0.2
        p_delete = 1 - matrix_01.mean(axis=0) + 0.2
        for i1 in dup_index:
            m = np.array(pop_new[i1])
            m1 = 1 * (m >= 0.6)
            m2 = "".join(map(str,m1))
            value_position = np.array(list(find_all(m2, '1')))
            un_position = np.array(list(find_all(m2, '0')))
            if len(value_position) > 1:
                temp_select = random.sample(list(value_position),1)
                deal_with_position = temp_select[0]
                if random.uniform(0,1) < p_delete[deal_with_position]:
                    pop_new[i1][deal_with_position] = 0
                temp_select1 = random.sample(list(un_position),1)
                deal_with_position1 = temp_select1[0]
                if random.uniform(0,1) < p_add[deal_with_position1]:
                    pop_new[i1][deal_with_position1] = 1
    return pop_new


def find_and_modify_duplicated(old,pop_new):###randomly change
    dup_index = []
    all_index = get_whole_01(old)
    unique_all_Index = set(all_index)
    unique_all_Index = list(unique_all_Index)
    for i in range(len(pop_new)):
        x = np.array(pop_new[i])
        x1 = 1 * (x >= 0.6)
        x2 = "".join(map(str, x1))
        mm = findindex(unique_all_Index,x2)
        if len(mm) == 0:
            unique_all_Index.append(x2)
        else:
            dup_index.append(i)
    for i1 in dup_index:
            m = np.array(pop_new[i1])
            m1 = 1 * (m >= 0.6)
            m2 = "".join(map(str,m1))
            value_position = np.array(list(find_all(m2, '1')))
            un_position = np.array(list(find_all(m2, '0')))
            if len(value_position) != 0 and len(un_position) != 0:
                temp = min(len(value_position),len(un_position))
                move_length = random.randint(1,temp)
                selected_random = random.sample(list(value_position),move_length)
                un_random = random.sample(list(un_position),move_length)
                for j in range(len(selected_random)):
                        pop_new[i1][selected_random[j]] = 0
                        pop_new[i1][un_random[j]] = 1
    return pop_new



def find_and_modify_duplicated_ori(old,pop_new,spaces,mic_value):####use mic to swap
    dup_index = []
    all_index = get_whole_01(old)
    unique_all_Index = set(all_index)
    unique_all_Index = list(unique_all_Index)
    for i in range(len(pop_new)):
        x = np.array(pop_new[i])
        x1 = 1 * (x >= 0.6)
        x2 = "".join(map(str, x1))
        mm = findindex(unique_all_Index,x2)
        if len(mm) == 0:
            unique_all_Index.append(x2)
        else:
            dup_index.append(i)
    for i1 in dup_index:
            m = np.array(pop_new[i1])
            m1 = 1 * (m >= 0.6)
            m2 = "".join(map(str,m1))
            value_position = np.array(list(find_all(m2, '1')))
            if len(value_position) > 1:
                move_length = random.randint(1,len(value_position)-1)
                selected_features = [spaces[j] for j in value_position]
                mic_selected_features = [mic_value[j] for j in selected_features]
                sorted_index = np.argsort(mic_selected_features)
                ty = [value_position[j] for j in sorted_index[0:move_length]]
                for j in ty:
                    pop_new[i1][j] = 0
    return pop_new



def selection_diversity(pop,k):
    levels = pareto_sorting(pop)
    unique_sets = list(set(levels))
    index = [np.argwhere(levels == i) for i in unique_sets]
    mm = 0
    used_index = []
    for f in range(len(index)):
        temp = index[f]
        temp1 = []
        for j in temp:
            temp1.append(j[0])
            used_index.append(j[0])
        index[f] = temp1####change the form
        mm = mm + len(temp)
        if mm < k:
            continue
        else:
            break
    # print('mm',mm,f,len(index[f]))
    if mm == k:
        pop_non = [pop[m] for m in used_index]
        return pop_non
    else:
    ####that means need to delete some solutions from index[f],,,,front groups have mm -len(index[f]),,,
    # therefore need to remove  mm - k  solutions from last group
        need_remove_num = mm - k
        # last_front = index[f]
        pop_last = [pop[m] for m in index[f]]
        matrix_cos_simi = np.zeros((len(pop_last),len(used_index)-1))
        for i in range(len(pop_last)):
            x = np.array(pop_last[i])
            pop_temp1 = [pop[m] for m in used_index]
            pop_temp1.remove(pop_last[i])
            for j in range(len(pop_temp1)):
                y = np.array(pop_temp1[j])
                matrix_cos_simi[i,j] = np.dot(x,y)/(np.linalg.norm(x)*(np.linalg.norm(y)))
                #matrix_cos_simi[i, j] = sum(map(lambda i, j: abs(i-j),x,y))##
        # print(matrix_cos_simi)
        cos_dis = [1-np.max(matrix_cos_simi[i,:]) for i in range(len(matrix_cos_simi))]#the larger the better
        crow_dis = assignCrowdingDist(pop_last)##the larger the better
        # whole_dis = [crow_dis[i]+ cos_dis[i] for i in range(len(crow_dis))]#remove smaller ones
        #####################################use caitong's formula
        #whole_dis = two_CD_one(cos_dis,crow_dis)
        #sort_index = np.argsort(np.array(whole_dis))
        #removed_index = sort_index[0:need_remove_num]
        #remove_pop_index = [index[f][m] for m in removed_index]
        #for tt in remove_pop_index:
            #used_index.remove(tt)
        #pop_out = [pop[m] for m in used_index]
        #####################################use hang's formula
        sort_index = np.argsort(np.array(cos_dis))
        removed_index = sort_index[0:need_remove_num]
        indexx = get_all_index(cos_dis,cos_dis[removed_index[-1]])
        if len(indexx) > 1:### solutions have the same cos value:
            comp = 0###the number of solutions with the same cos outside the removed_index
            for ele in indexx:
                if ele not in removed_index:
                    comp =comp +1
            if comp !=0:####need choose len(indexx) -comp solutions from len(indexx)
                removed_index = list(removed_index)
                for jj in indexx:
                    if jj in removed_index:
                        removed_index.remove(jj)
                chosen_length = len(indexx) -comp
                crow_dis_temp= [crow_dis[mm] for mm in indexx]
                sort_crow_temp = np.argsort(crow_dis_temp)
                indexx_x = [indexx[t] for t in sort_crow_temp[0:chosen_length]]
                removed_index.extend(indexx_x)
        remove_pop_index = [index[f][m] for m in removed_index]
        for tt in remove_pop_index:
            used_index.remove(tt)
        pop_out = [pop[m] for m in used_index]
        return pop_out


def two_CD_one(a,b):
    distances2 = [0.0] * len(a)  #####all is 0
    avg_a = np.mean(a)
    avg_b = np.mean(b)
    for i_i in range(len(a)):
        if a[i_i] > avg_a or b[i_i] > avg_b:
            distances2[i_i] = max(a[i_i],b[i_i])#/(rank +1))
        else:
            distances2[i_i] = min(a[i_i],b[i_i])
    return distances2


def niche_introduce(dis,ii):
    ss_i = list(np.argsort(dis[ii, :]))  ###the index in the whole population from min to max
    niche_index = ss_i[:4]  #####the three having the smallest distance to the current individual, the first one is itself
    three_distance = [dis[ii, iii] for iii in niche_index[1:4]]
    gaussion_mean = np.mean(three_distance)
    gaussion_std = np.std(three_distance)
    range_low = gaussion_mean - 3 * gaussion_std
    range_high = gaussion_mean + 3 * gaussion_std
    for j in ss_i:
        if range_low <= dis[ii, j] <= range_high:
            niche_index.append(j)
    niche_index = list(sorted(set(niche_index)))###sort is to ensure the first ndividual is itself
    return niche_index





def pareto_sorting(pop):
    PF = np.array([ind.fitness.values for ind in pop])
    [levels, criLevel1] = ea.indicator.ndsortDED(PF)
    return levels




def sortNondominated(individuals, k, first_front_only=False):
    if k == 0:
        return []
    map_fit_ind = defaultdict(list)
    for ind in individuals:
        map_fit_ind[ind.fitness].append(ind)
    fits = map_fit_ind.keys()
    current_front = []
    next_front = []
    dominating_fits = defaultdict(int)
    dominated_fits = defaultdict(list)
    # Rank first Pareto front
    for i, fit_i in enumerate(fits):
        fits = list(fits)
        for fit_j in fits[i + 1:]:
            if fit_i.dominates(fit_j):
                dominating_fits[fit_j] += 1
                dominated_fits[fit_i].append(fit_j)
            elif fit_j.dominates(fit_i):
                dominating_fits[fit_i] += 1
                dominated_fits[fit_j].append(fit_i)
        if dominating_fits[fit_i] == 0:
            current_front.append(fit_i)
    fronts = [[]]
    for fit in current_front:
        fronts[-1].extend(map_fit_ind[fit])
    pareto_sorted = len(fronts[-1])
    if not first_front_only:
        N = min(len(individuals), k)
        while pareto_sorted < N:
            fronts.append([])
            for fit_p in current_front:
                for fit_d in dominated_fits[fit_p]:
                    dominating_fits[fit_d] -= 1
                    if dominating_fits[fit_d] == 0:
                        next_front.append(fit_d)
                        pareto_sorted += len(map_fit_ind[fit_d])
                        fronts[-1].extend(map_fit_ind[fit_d])
            current_front = next_front
            next_front = []
    return fronts



def assignCrowdingDist(individuals):
    if len(individuals) == 0:
        return
    distances = [0.0] * len(individuals)
    crowd = [(ind.fitness.values, i) for i, ind in enumerate(individuals)]
    nobj = len(individuals[0].fitness.values)
    for i in range(nobj):
        crowd.sort(key=lambda element: element[0][i])
        # distances[crowd[0][1]] = float("inf")
        # distances[crowd[-1][1]] = float("inf")
        distances[crowd[0][1]] =1
        distances[crowd[-1][1]] =1
        if crowd[-1][0][i] == crowd[0][0][i]:
            continue
        norm = nobj * float(crowd[-1][0][i] - crowd[0][0][i])
        for prev, cur, next in zip(crowd[:-2], crowd[1:-1], crowd[2:]):
            distances[cur[1]] += (next[0][i] - prev[0][i]) / norm
    return distances




def selection_diversity_ori(pop,k):
    pareto_fronts = sortNondominated(pop, k)
    pop_non,pop_temp = pareto_fronts[0],pareto_fronts[0]
    front_num = [len(pop_non)]
    temp = len(pop_non)
    # print('the number of solutions in first front',temp,k)
    for i in range(1,len(pareto_fronts)):
         for j in pareto_fronts[i]:
             pop_temp.append(j)
         front_num.append(len(pareto_fronts[i]))
    if temp == k:
        return pop_non
    if temp > k:
        distances = assignCrowdingDist(pop_non)
        sorted_index = np.argsort(-np.array(distances))
        out = [pop_non[m] for m in sorted_index[0:k]]
        return out
    if temp < k:
        matrix_cos_simi = np.zeros((len(pop_temp),len(pop_temp)))
        for i in range(len(pop_temp)):
            for j in range(len(pop_temp)):
                x,y = np.array(pop_temp[i]),np.array(pop_temp[j])
                matrix_cos_simi[i,j] = np.dot(x,y)/(np.linalg.norm(x)*(np.linalg.norm(y)))
            matrix_cos_simi[i,i] = 0 #####make 1 as 0
                # man_dis = sum(map(lambda i, j: abs(i-j),x,y))##
        cos_dis = [np.max(matrix_cos_simi[i,:]) for i in range(len(matrix_cos_simi))]
        last_solutions,needed = pareto_fronts[-1],len(pareto_fronts[-1])-len(pop_temp)+k
        ###that means need to select needed solutions from last_solutions
        last_dis = assignCrowdingDist(last_solutions)
        last_simi = cos_dis[-len(pareto_fronts[-1]):]###last several ones
        # print(last_dis)
        # print(last_simi)
        whole_dis = [last_dis[i]+ 1-last_simi[i] for i in range(len(last_dis))]##make it larger better
        # print(whole_dis)
        sort_index = np.argsort(-np.array(whole_dis))
        used_index = sort_index[0:needed]
        pop_out = pop_temp[0:np.sum(front_num[0:-1])]
        for mm in used_index:
            pop_out.append(last_solutions[mm])
        print('len(pop_out)',len(pop_out))
        # exit()
        return pop


