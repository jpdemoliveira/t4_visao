from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
import time
import os

# Valores de componentes principais para teste
valores_pca = [20, 50, 100]

# Distâncias
metricas_distancia = ['euclidean', 'manhattan', 'cosine']

# Divisões de treino/validação/teste
divisoes = [
    (0.6, 0.2, 0.2),
    (0.7, 0.15, 0.15),
    (0.8, 0.1, 0.1)
]

# Carregar MNIST
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist.data, mnist.target.astype('int')

# Criar diretorio de resultados
raiz_saida = "resultados_multiplos_pca"
os.makedirs(raiz_saida, exist_ok=True)

# Loop pelos valores de PCA
for n_pca in valores_pca:

    # Reduzir dimensionalidade com PCA
    # Aplicar PCA
    pca = PCA(n_components=n_pca)
    X_reduzido = pca.fit_transform(X)
    output_dir = os.path.join(raiz_saida, f"pca_{n_pca}")
    os.makedirs(output_dir, exist_ok=True)

    for div in divisoes:
        treino_p, valid_p, teste_p = div
        X_temp, X_test, y_temp, y_test = train_test_split(X_reduzido, y, test_size=teste_p, random_state=42)
        valid_ratio = valid_p / (1 - teste_p)
        X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=valid_ratio, random_state=42)

        for metrica in metricas_distancia:

            # kNN com métrica de distância
            knn = KNeighborsClassifier(n_neighbors=3, metric=metrica)

            t_ini_treino = time.time()
            knn.fit(X_train, y_train)
            t_fim_treino = time.time()

            t_ini_pred = time.time()
            y_valid_pred = knn.predict(X_valid)
            y_test_pred = knn.predict(X_test)
            t_fim_pred = time.time()

            acc_valid = accuracy_score(y_valid, y_valid_pred)
            acc_test = accuracy_score(y_test, y_test_pred)
            tempo_treino = t_fim_treino - t_ini_treino
            tempo_pred = t_fim_pred - t_ini_pred

            filename = f"knn_{metrica}_{int(treino_p*100)}-{int(valid_p*100)}-{int(teste_p*100)}.txt"
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'w') as f:
                f.write(f"[CLASSIFICADOR: kNN]\n")
                f.write(f"PCA: {n_pca} componentes\n")
                f.write(f"Distância: {metrica}\n")
                f.write(f"Divisão treino/val/teste: {int(treino_p*100)}/{int(valid_p*100)}/{int(teste_p*100)}\n\n")
                f.write(f"Acurácia Validação: {acc_valid:.2%}\n")
                f.write(f"Acurácia Teste: {acc_test:.2%}\n")
                f.write(f"Tempo de Treinamento: {tempo_treino:.2f} segundos\n")
                f.write(f"Tempo de Predição: {tempo_pred:.2f} segundos\n\n")
                f.write("Relatório de Classificação (Teste):\n")
                f.write(classification_report(y_test, y_test_pred))

        ### Classificador Linear ###
        clf = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial')

        t_ini_treino = time.time()
        clf.fit(X_train, y_train)
        t_fim_treino = time.time()

        t_ini_pred = time.time()
        y_valid_pred = clf.predict(X_valid)
        y_test_pred = clf.predict(X_test)
        t_fim_pred = time.time()

        acc_valid = accuracy_score(y_valid, y_valid_pred)
        acc_test = accuracy_score(y_test, y_test_pred)
        tempo_treino = t_fim_treino - t_ini_treino
        tempo_pred = t_fim_pred - t_ini_pred

        filename = f"linear_{int(treino_p*100)}-{int(valid_p*100)}-{int(teste_p*100)}.txt"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w') as f:
            f.write(f"[CLASSIFICADOR: Regressão Logística]\n")
            f.write(f"PCA: {n_pca} componentes\n")
            f.write(f"Divisão treino/val/teste: {int(treino_p*100)}/{int(valid_p*100)}/{int(teste_p*100)}\n\n")
            f.write(f"Acurácia Validação: {acc_valid:.2%}\n")
            f.write(f"Acurácia Teste: {acc_test:.2%}\n")
            f.write(f"Tempo de Treinamento: {tempo_treino:.2f} segundos\n")
            f.write(f"Tempo de Predição: {tempo_pred:.2f} segundos\n\n")
            f.write("Relatório de Classificação (Teste):\n")
            f.write(classification_report(y_test, y_test_pred))