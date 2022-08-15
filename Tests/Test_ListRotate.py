#%%
l = [1, 2, 3, 4, 5, 6]
k_elems = int(input('k_elems: '))

def k_fold_list(l, k_elems):
    
    for _ in range(k_elems):
        l = [l.pop()]+l
    return l
# help(list)
l = k_fold_list(l,k_elems)
print(l)
