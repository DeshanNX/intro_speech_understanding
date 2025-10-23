
def words2characters(words):
    characters = []
    for w in words:
        for ch in str(w):
            characters.append(ch)
    return characters

