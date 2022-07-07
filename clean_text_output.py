import re

txt = ""

remove_list = ["PREC@1:", "PREC@3:", "PREC@5:", "PREC@10:",
               "RECALL@1:", "RECALL@3:", "RECALL@5:", "RECALL@10:",
               "MAP@1:", "MAP@3:", "MAP@5:", "MAP@10:",
               "NDCG@1:", "NDCG@3:", "NDCG@5:", "NDCG@10:"]


def clean_text(rgx_list, new_text):
    new_text = re.sub(rgx_list[0], '', new_text)
    for rgx_match in rgx_list:
        new_text = re.sub(rgx_match, ',', new_text)
    return new_text


print(clean_text(remove_list, txt))
