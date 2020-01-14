import csv
#
# for line in csv.reader(open("caches/cache_request.txt","r"),):
#     if "Love the Way" in line[0]:
#         print(line[0])

# print()
# print()
# for line in open("caches/cache_request.txt","r").readlines():
#     if "Love the Way" in line:
#         print(line[:100])

count = 0
total = 0
for line in csv.reader(open("data_in_use/nn_train_with_drqa_dev.csv","r"),):
    if int(line[-1]) == 1:
        count += 1
    total += 1

print("Dev", count, total, count/total)

count = 0
total = 0
for line in csv.reader(open("data/nn_train_with_drqa_full.csv","r"),):
    if int(line[-1]) == 1:
        count += 1
    total += 1

print("Test", count, total, count/total)

