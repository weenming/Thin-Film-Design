def sort_by_ith_list(*args, index):
    assert index <= len(args)
    # sort acording to initial ot so that plot does not get messy
    res = [
        [
            a[i] for a in sorted(zip(*args), key=lambda lists: lists[index])
        ] 
        for i in range(len(args))
    ]
    return res