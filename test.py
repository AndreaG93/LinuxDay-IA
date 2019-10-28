# scikit-learn, pandas
from sklearn.datasets import load_iris, load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# Validation taglio i dati in due
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


def ddsa():
    dataset = load_iris()

    X = dataset['data']  # input matrice feature
    y = dataset['target']  # output

    X_train, X_test, y_train, y_test = train_test_split(X,y)

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    p_train = model.predict(X_train)
    p_test = model.predict(X_test)

    acc_test = accuracy_score(y_test, p_test)
    acc_train = accuracy_score(y_train, p_train)
    print(acc_test)
    print(acc_train)

    # sul test prestazione minore del train... è normale avare accuratezza inferiore a 1
    # sul train è ovvio aveere acc 1

    # clustering .... raggrupppi in categroia un dataset (PRODUCE LUI LE CATEGORIE)
    # classificatore .... raggruppa in categorie ma gliele devi dare

    # transfer learning !!!!




def main():
    dataset = load_iris()

    X = dataset['data']  # input matrice feature
    y = dataset['target']  # output

    model = DecisionTreeClassifier()   # processo di apprendimento aggiusta coefficienti angolari retta
    # regola delta ... bad progpagation
    # deep learning: tecniche ispirate al tessuto nervoso
    # machine learning: IA a partire dai dati
    # reti neurali --- una via di  mezzo

    model.fit(X, y)   # Addrestramento...

    p = model.predict(X)
    print(p)

    acc = accuracy_score(y, p)
    print(acc) # verrà 1 --- 100% accuratezza

    # generalizza su altri dati????
    # una parte per addestrmento altri per i testi (DIVIDO IN DUE IL DATASET)

    # over fitting (poco training) --- underfitting (troppo ... mi specializzo in pochi esempi non ha capito la struttura)

    # distill.pub // visualizzazion

















#main()
ddsa()