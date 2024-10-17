from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def pipeline_1(preproceso):
    pipeline = Pipeline([
        ('preproceso', preproceso),
        ('modelo', LogisticRegression(max_iter=1000))
    ])
    return pipeline

def pipeline_2(X_grupo, y_grupo, preproceso):
    pipeline = create_pipeline(preproceso)
    pipeline.fit(X_grupo, y_grupo)
    return pipeline

def pipeline_3(pipeline, X_prueba, y_prueba):
    y_predicion = pipeline.predict(X_prueba)
    precision = accuracy_score(y_prueba, y_predicion)
    print(f"Precision: {precision}")
