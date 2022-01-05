#makes sure that everything in new_batch.csv is marked as used in the master list, so that we don't re-sample things we already sampled in an ongoing batch

new_batch = open("new_batch.csv","r").read().splitlines()
master_list = open("../../master-screen-list/master-list.csv","r").read().splitlines()

master_list = [entry.split(",") for entry in master_list]

master_list2 = []


for m in master_list:
    if m[1] == "1":
        master_list2.append(m[0])

print(len(master_list2))

for l in new_batch:
    if l not in master_list2:
        print(l)

