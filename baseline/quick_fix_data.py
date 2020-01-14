# import csv
# import re
#
#
# p = re.compile("\[_")
# fp = open("caches/_get_noun_phrases.txt")
# for line in csv.reader(fp):
#
#     if p.search(str(line)):
#         print(line[1])

# PATTERN IS THE VALUES YOU WANT TO REMOVE NOT KEEP


def fix_cache(path, pattern):
    with open(path, "r") as f:
        lines = f.readlines()
    with open(path, "w") as f:
        for line in lines:
            if not pattern(line):
                f.write(line)


# fix_cache("caches/_get_drqa_score_doc.txt", lambda x: "-1" in x)
# fix_cache("caches/_get_drqa_score_sent.txt", lambda x: "-1" in x)
fix_cache("caches/_get_pageview_data.txt", lambda x: "-1" in x)
# fix_cache("caches/cache_request.txt", lambda x: x.count(",") != 1)
# fix_cache("caches/_get_pageview_data.txt", lambda x: x.count(",") != 1)


print("DONE")