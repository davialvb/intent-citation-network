def sorting(cetralts, n_tops):
    cetralts_tops = sorted(cetralts.items(), key=lambda item: item[1], reverse=True)[:n_tops]
    return cetralts_tops


def infos(centrs_tops, df):
    top_i = 1
    for idd, val in centrs_tops:
        print("Top",str(top_i), ", Id arXiv:", idd, ", Centrality:",round(val,4))
        print(f"Title: {df[df["id"]==idd]["title"].values[0]}", "\n")
        print(f"Discipline: {df[df["id"]==idd]["discipline"].values[0]}", "\n")
        top_i += 1
