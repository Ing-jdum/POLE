import pandas as pd
import ast

### Resultados de correr la funcion que obtiene la mejor combinacion de parametros

centrality_measures = """
Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.7567567567567568
Accuracy on Training Set: 0.8061224489795918
['clustering']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 4, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8243243243243243
Accuracy on Training Set: 0.9421768707482994
['pagerank']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.7702702702702703
Accuracy on Training Set: 0.8979591836734694
['pagerank', 'clustering']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 6, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.7027027027027027
Accuracy on Training Set: 0.7687074829931972
['degree']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 8, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.7162162162162162
Accuracy on Training Set: 0.7619047619047619
['degree', 'clustering']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 4, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8243243243243243
Accuracy on Training Set: 0.9183673469387755
['degree', 'pagerank']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 6, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.7567567567567568
Accuracy on Training Set: 0.8333333333333334
['degree', 'pagerank', 'clustering']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 8, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.7837837837837838
Accuracy on Training Set: 0.7959183673469388
['closeness']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 8, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8513513513513513
Accuracy on Training Set: 0.8537414965986394
['closeness', 'clustering']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 8, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8648648648648649
Accuracy on Training Set: 0.9013605442176871
['closeness', 'pagerank']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 8, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8378378378378378
Accuracy on Training Set: 0.8741496598639455
['closeness', 'pagerank', 'clustering']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 8, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8378378378378378
Accuracy on Training Set: 0.8707482993197279
['closeness', 'degree']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 8, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8108108108108109
Accuracy on Training Set: 0.8435374149659864
['closeness', 'degree', 'clustering']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 6, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8378378378378378
Accuracy on Training Set: 0.9047619047619048
['closeness', 'degree', 'pagerank']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 8, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8378378378378378
Accuracy on Training Set: 0.9013605442176871
['closeness', 'degree', 'pagerank', 'clustering']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 6, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.7432432432432432
Accuracy on Training Set: 0.8333333333333334
['betweenness']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 8, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.7702702702702703
Accuracy on Training Set: 0.8435374149659864
['betweenness', 'clustering']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 8, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.7837837837837838
Accuracy on Training Set: 0.8639455782312925
['betweenness', 'pagerank']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 4, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8108108108108109
Accuracy on Training Set: 0.95578231292517
['betweenness', 'pagerank', 'clustering']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 6, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.7702702702702703
Accuracy on Training Set: 0.8129251700680272
['betweenness', 'degree']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 6, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.7702702702702703
Accuracy on Training Set: 0.8367346938775511
['betweenness', 'degree', 'clustering']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 4, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8108108108108109
Accuracy on Training Set: 0.8945578231292517
['betweenness', 'degree', 'pagerank']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 8, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.7702702702702703
Accuracy on Training Set: 0.8639455782312925
['betweenness', 'degree', 'pagerank', 'clustering']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 8, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8783783783783784
Accuracy on Training Set: 0.9013605442176871
['betweenness', 'closeness']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 8, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8378378378378378
Accuracy on Training Set: 0.8843537414965986
['betweenness', 'closeness', 'clustering']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 8, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8648648648648649
Accuracy on Training Set: 0.9149659863945578
['betweenness', 'closeness', 'pagerank']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 8, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8648648648648649
Accuracy on Training Set: 0.9217687074829932
['betweenness', 'closeness', 'pagerank', 'clustering']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8783783783783784
Accuracy on Training Set: 0.9319727891156463
['betweenness', 'closeness', 'degree']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 4, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8648648648648649
Accuracy on Training Set: 0.9387755102040817
['betweenness', 'closeness', 'degree', 'clustering']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 6, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8513513513513513
Accuracy on Training Set: 0.9285714285714286
['betweenness', 'closeness', 'degree', 'pagerank']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 6, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8783783783783784
Accuracy on Training Set: 0.9455782312925171
['betweenness', 'closeness', 'degree', 'pagerank', 'clustering']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 8, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.5405405405405406
Accuracy on Training Set: 0.54421768707483
['triangles']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 8, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.7162162162162162
Accuracy on Training Set: 0.7551020408163265
['triangles', 'clustering']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 8, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.7027027027027027
Accuracy on Training Set: 0.7346938775510204
['triangles', 'pagerank']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 6, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.7297297297297297
Accuracy on Training Set: 0.8197278911564626
['triangles', 'pagerank', 'clustering']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 8, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.6891891891891891
Accuracy on Training Set: 0.717687074829932
['triangles', 'degree']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 8, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.7162162162162162
Accuracy on Training Set: 0.7619047619047619
['triangles', 'degree', 'clustering']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 6, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.7297297297297297
Accuracy on Training Set: 0.826530612244898
['triangles', 'degree', 'pagerank']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 8, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.7432432432432432
Accuracy on Training Set: 0.8333333333333334
['triangles', 'degree', 'pagerank', 'clustering']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 8, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8243243243243243
Accuracy on Training Set: 0.8605442176870748
['triangles', 'closeness']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 8, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.7837837837837838
Accuracy on Training Set: 0.8231292517006803
['triangles', 'closeness', 'clustering']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 8, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8108108108108109
Accuracy on Training Set: 0.8775510204081632
['triangles', 'closeness', 'pagerank']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 8, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8513513513513513
Accuracy on Training Set: 0.8979591836734694
['triangles', 'closeness', 'pagerank', 'clustering']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 8, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8378378378378378
Accuracy on Training Set: 0.8571428571428571
['triangles', 'closeness', 'degree']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 8, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8513513513513513
Accuracy on Training Set: 0.9013605442176871
['triangles', 'closeness', 'degree', 'clustering']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 6, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8783783783783784
Accuracy on Training Set: 0.9115646258503401
['triangles', 'closeness', 'degree', 'pagerank']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 8, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8648648648648649
Accuracy on Training Set: 0.8809523809523809
['triangles', 'closeness', 'degree', 'pagerank', 'clustering']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 6, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8108108108108109
Accuracy on Training Set: 0.8877551020408163
['triangles', 'betweenness']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 8, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.7297297297297297
Accuracy on Training Set: 0.7993197278911565
['triangles', 'betweenness', 'clustering']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 8, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.7297297297297297
Accuracy on Training Set: 0.8299319727891157
['triangles', 'betweenness', 'pagerank']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 8, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.7567567567567568
Accuracy on Training Set: 0.8639455782312925
['triangles', 'betweenness', 'pagerank', 'clustering']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 8, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.7432432432432432
Accuracy on Training Set: 0.7993197278911565
['triangles', 'betweenness', 'degree']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 8, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.7567567567567568
Accuracy on Training Set: 0.8299319727891157
['triangles', 'betweenness', 'degree', 'clustering']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 8, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.7702702702702703
Accuracy on Training Set: 0.8503401360544217
['triangles', 'betweenness', 'degree', 'pagerank']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 8, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.7702702702702703
Accuracy on Training Set: 0.8537414965986394
['triangles', 'betweenness', 'degree', 'pagerank', 'clustering']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 8, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8783783783783784
Accuracy on Training Set: 0.8809523809523809
['triangles', 'betweenness', 'closeness']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 8, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8513513513513513
Accuracy on Training Set: 0.9149659863945578
['triangles', 'betweenness', 'closeness', 'clustering']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8918918918918919
Accuracy on Training Set: 0.9625850340136054
['triangles', 'betweenness', 'closeness', 'pagerank']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 8, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8783783783783784
Accuracy on Training Set: 0.891156462585034
['triangles', 'betweenness', 'closeness', 'pagerank', 'clustering']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 8, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8648648648648649
Accuracy on Training Set: 0.9081632653061225
['triangles', 'betweenness', 'closeness', 'degree']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 6, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8108108108108109
Accuracy on Training Set: 0.8945578231292517
['triangles', 'betweenness', 'closeness', 'degree', 'clustering']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 6, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8783783783783784
Accuracy on Training Set: 0.9319727891156463
['triangles', 'betweenness', 'closeness', 'degree', 'pagerank']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 8, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8378378378378378
Accuracy on Training Set: 0.8775510204081632
['triangles', 'betweenness', 'closeness', 'degree', 'pagerank', 'clustering']
"""

problem_knowledge_features = """
Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 6, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.9054054054054054
Accuracy on Training Set: 0.9149659863945578
['lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 8, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8513513513513513
Accuracy on Training Set: 0.8673469387755102
['total_crimes_count']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 8, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8918918918918919
Accuracy on Training Set: 0.8877551020408163
['connected_criminals_count']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 8, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8918918918918919
Accuracy on Training Set: 0.8877551020408163
['connected_criminals_count', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8783783783783784
Accuracy on Training Set: 0.9081632653061225
['connected_criminals_count', 'total_crimes_count']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.9054054054054054
Accuracy on Training Set: 0.8843537414965986
['connected_criminals_count', 'total_crimes_count', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 8, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8513513513513513
Accuracy on Training Set: 0.8605442176870748
['pagerank', 'total_crimes_count']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 6, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8513513513513513
Accuracy on Training Set: 0.8469387755102041
['pagerank', 'total_crimes_count', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 8, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8513513513513513
Accuracy on Training Set: 0.8707482993197279
['pagerank', 'connected_criminals_count']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 6, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.9054054054054054
Accuracy on Training Set: 0.8571428571428571
['pagerank', 'connected_criminals_count', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 8, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8648648648648649
Accuracy on Training Set: 0.8571428571428571
['pagerank', 'connected_criminals_count', 'total_crimes_count']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 8, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8783783783783784
Accuracy on Training Set: 0.8401360544217688
['pagerank', 'connected_criminals_count', 'total_crimes_count', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 4, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8783783783783784
Accuracy on Training Set: 0.9115646258503401
['pagerank', 'clustering', 'total_crimes_count']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 6, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8648648648648649
Accuracy on Training Set: 0.8775510204081632
['pagerank', 'clustering', 'total_crimes_count', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 8, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8648648648648649
Accuracy on Training Set: 0.8707482993197279
['pagerank', 'clustering', 'connected_criminals_count', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 8, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8783783783783784
Accuracy on Training Set: 0.8639455782312925
['pagerank', 'clustering', 'connected_criminals_count', 'total_crimes_count']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 8, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8783783783783784
Accuracy on Training Set: 0.8605442176870748
['pagerank', 'clustering', 'connected_criminals_count', 'total_crimes_count', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8513513513513513
Accuracy on Training Set: 0.9319727891156463
['pagerank', 'clustering', 'nearest_path', 'connected_criminals_count', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 6, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8513513513513513
Accuracy on Training Set: 0.8299319727891157
['degree', 'clustering', 'nearest_path', 'total_crimes_count', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 6, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8648648648648649
Accuracy on Training Set: 0.8877551020408163
['degree', 'pagerank', 'total_crimes_count']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 8, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8648648648648649
Accuracy on Training Set: 0.8605442176870748
['degree', 'pagerank', 'total_crimes_count', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 6, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8648648648648649
Accuracy on Training Set: 0.8775510204081632
['degree', 'pagerank', 'connected_criminals_count']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8513513513513513
Accuracy on Training Set: 0.9319727891156463
['degree', 'pagerank', 'connected_criminals_count', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 6, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8648648648648649
Accuracy on Training Set: 0.891156462585034
['degree', 'pagerank', 'connected_criminals_count', 'total_crimes_count']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 8, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8648648648648649
Accuracy on Training Set: 0.8571428571428571
['degree', 'pagerank', 'connected_criminals_count', 'total_crimes_count', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8513513513513513
Accuracy on Training Set: 0.9217687074829932
['degree', 'pagerank', 'nearest_path', 'connected_criminals_count']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 8, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8513513513513513
Accuracy on Training Set: 0.8537414965986394
['degree', 'pagerank', 'nearest_path', 'connected_criminals_count', 'total_crimes_count', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 6, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8648648648648649
Accuracy on Training Set: 0.9013605442176871
['degree', 'pagerank', 'clustering', 'total_crimes_count']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 8, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8513513513513513
Accuracy on Training Set: 0.8435374149659864
['degree', 'pagerank', 'clustering', 'total_crimes_count', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 4, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8648648648648649
Accuracy on Training Set: 0.9115646258503401
['degree', 'pagerank', 'clustering', 'connected_criminals_count']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 6, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8648648648648649
Accuracy on Training Set: 0.8877551020408163
['degree', 'pagerank', 'clustering', 'connected_criminals_count', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 8, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8513513513513513
Accuracy on Training Set: 0.8469387755102041
['degree', 'pagerank', 'clustering', 'connected_criminals_count', 'total_crimes_count']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8783783783783784
Accuracy on Training Set: 0.9115646258503401
['degree', 'pagerank', 'clustering', 'connected_criminals_count', 'total_crimes_count', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8513513513513513
Accuracy on Training Set: 0.9217687074829932
['degree', 'pagerank', 'clustering', 'nearest_path', 'connected_criminals_count', 'total_crimes_count', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 6, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8783783783783784
Accuracy on Training Set: 0.8571428571428571
['closeness', 'connected_criminals_count']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 6, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.918918918918919
Accuracy on Training Set: 0.8809523809523809
['closeness', 'connected_criminals_count', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 4, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8648648648648649
Accuracy on Training Set: 0.9149659863945578
['closeness', 'connected_criminals_count', 'total_crimes_count', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 8, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8513513513513513
Accuracy on Training Set: 0.891156462585034
['closeness', 'clustering']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 6, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8783783783783784
Accuracy on Training Set: 0.8979591836734694
['closeness', 'clustering', 'total_crimes_count']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 6, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8513513513513513
Accuracy on Training Set: 0.8843537414965986
['closeness', 'clustering', 'total_crimes_count', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 6, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8918918918918919
Accuracy on Training Set: 0.8877551020408163
['closeness', 'clustering', 'connected_criminals_count']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 6, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.9054054054054054
Accuracy on Training Set: 0.8741496598639455
['closeness', 'clustering', 'connected_criminals_count', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8783783783783784
Accuracy on Training Set: 0.9251700680272109
['closeness', 'clustering', 'connected_criminals_count', 'total_crimes_count']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8648648648648649
Accuracy on Training Set: 0.8979591836734694
['closeness', 'clustering', 'connected_criminals_count', 'total_crimes_count', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8783783783783784
Accuracy on Training Set: 0.8979591836734694
['closeness', 'clustering', 'nearest_path', 'total_crimes_count', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 6, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8918918918918919
Accuracy on Training Set: 0.8605442176870748
['closeness', 'clustering', 'nearest_path', 'connected_criminals_count']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8513513513513513
Accuracy on Training Set: 0.9115646258503401
['closeness', 'clustering', 'nearest_path', 'connected_criminals_count', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 6, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8648648648648649
Accuracy on Training Set: 0.9115646258503401
['closeness', 'pagerank']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8918918918918919
Accuracy on Training Set: 0.8843537414965986
['closeness', 'pagerank', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 8, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8783783783783784
Accuracy on Training Set: 0.9183673469387755
['closeness', 'pagerank', 'total_crimes_count', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 8, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8783783783783784
Accuracy on Training Set: 0.9047619047619048
['closeness', 'pagerank', 'connected_criminals_count']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 6, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.9054054054054054
Accuracy on Training Set: 0.9251700680272109
['closeness', 'pagerank', 'connected_criminals_count', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 8, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8783783783783784
Accuracy on Training Set: 0.9149659863945578
['closeness', 'pagerank', 'connected_criminals_count', 'total_crimes_count']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 6, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8918918918918919
Accuracy on Training Set: 0.9115646258503401
['closeness', 'pagerank', 'connected_criminals_count', 'total_crimes_count', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 8, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8783783783783784
Accuracy on Training Set: 0.8639455782312925
['closeness', 'pagerank', 'nearest_path']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8918918918918919
Accuracy on Training Set: 0.9115646258503401
['closeness', 'pagerank', 'nearest_path', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 8, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8648648648648649
Accuracy on Training Set: 0.8945578231292517
['closeness', 'pagerank', 'nearest_path', 'total_crimes_count']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 8, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8648648648648649
Accuracy on Training Set: 0.8809523809523809
['closeness', 'pagerank', 'nearest_path', 'connected_criminals_count']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 8, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8648648648648649
Accuracy on Training Set: 0.8945578231292517
['closeness', 'pagerank', 'nearest_path', 'connected_criminals_count', 'total_crimes_count']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 6, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8648648648648649
Accuracy on Training Set: 0.8945578231292517
['closeness', 'pagerank', 'nearest_path', 'connected_criminals_count', 'total_crimes_count', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 8, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8648648648648649
Accuracy on Training Set: 0.8979591836734694
['closeness', 'pagerank', 'clustering', 'total_crimes_count']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 8, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8648648648648649
Accuracy on Training Set: 0.8945578231292517
['closeness', 'pagerank', 'clustering', 'total_crimes_count', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 6, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8918918918918919
Accuracy on Training Set: 0.9183673469387755
['closeness', 'pagerank', 'clustering', 'connected_criminals_count']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 8, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8648648648648649
Accuracy on Training Set: 0.8979591836734694
['closeness', 'pagerank', 'clustering', 'connected_criminals_count', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 8, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8648648648648649
Accuracy on Training Set: 0.9013605442176871
['closeness', 'pagerank', 'clustering', 'connected_criminals_count', 'total_crimes_count']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 8, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8783783783783784
Accuracy on Training Set: 0.8809523809523809
['closeness', 'pagerank', 'clustering', 'connected_criminals_count', 'total_crimes_count', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8513513513513513
Accuracy on Training Set: 0.8979591836734694
['closeness', 'pagerank', 'clustering', 'nearest_path']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 8, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8648648648648649
Accuracy on Training Set: 0.8877551020408163
['closeness', 'pagerank', 'clustering', 'nearest_path', 'total_crimes_count']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 8, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8513513513513513
Accuracy on Training Set: 0.8809523809523809
['closeness', 'pagerank', 'clustering', 'nearest_path', 'total_crimes_count', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 8, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8783783783783784
Accuracy on Training Set: 0.891156462585034
['closeness', 'pagerank', 'clustering', 'nearest_path', 'connected_criminals_count']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8648648648648649
Accuracy on Training Set: 0.9285714285714286
['closeness', 'pagerank', 'clustering', 'nearest_path', 'connected_criminals_count', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 8, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8783783783783784
Accuracy on Training Set: 0.9013605442176871
['closeness', 'pagerank', 'clustering', 'nearest_path', 'connected_criminals_count', 'total_crimes_count']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 6, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8648648648648649
Accuracy on Training Set: 0.9013605442176871
['closeness', 'pagerank', 'clustering', 'nearest_path', 'connected_criminals_count', 'total_crimes_count', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 6, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8783783783783784
Accuracy on Training Set: 0.8979591836734694
['closeness', 'degree', 'total_crimes_count']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 8, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.9054054054054054
Accuracy on Training Set: 0.8775510204081632
['closeness', 'degree', 'connected_criminals_count']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 8, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8783783783783784
Accuracy on Training Set: 0.9047619047619048
['closeness', 'degree', 'connected_criminals_count', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 8, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.9054054054054054
Accuracy on Training Set: 0.8877551020408163
['closeness', 'degree', 'connected_criminals_count', 'total_crimes_count']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8783783783783784
Accuracy on Training Set: 0.9183673469387755
['closeness', 'degree', 'connected_criminals_count', 'total_crimes_count', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 8, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8513513513513513
Accuracy on Training Set: 0.8129251700680272
['closeness', 'degree', 'nearest_path']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 8, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8648648648648649
Accuracy on Training Set: 0.8197278911564626
['closeness', 'degree', 'nearest_path', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 6, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8513513513513513
Accuracy on Training Set: 0.8605442176870748
['closeness', 'degree', 'nearest_path', 'total_crimes_count']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 6, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8648648648648649
Accuracy on Training Set: 0.8775510204081632
['closeness', 'degree', 'nearest_path', 'total_crimes_count', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 8, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8513513513513513
Accuracy on Training Set: 0.7993197278911565
['closeness', 'degree', 'nearest_path', 'connected_criminals_count']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 6, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8648648648648649
Accuracy on Training Set: 0.8469387755102041
['closeness', 'degree', 'nearest_path', 'connected_criminals_count', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 8, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8783783783783784
Accuracy on Training Set: 0.8401360544217688
['closeness', 'degree', 'nearest_path', 'connected_criminals_count', 'total_crimes_count']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 6, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8918918918918919
Accuracy on Training Set: 0.8469387755102041
['closeness', 'degree', 'nearest_path', 'connected_criminals_count', 'total_crimes_count', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 8, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8513513513513513
Accuracy on Training Set: 0.8741496598639455
['closeness', 'degree', 'clustering', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 8, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8648648648648649
Accuracy on Training Set: 0.8639455782312925
['closeness', 'degree', 'clustering', 'total_crimes_count']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 8, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8513513513513513
Accuracy on Training Set: 0.8401360544217688
['closeness', 'degree', 'clustering', 'total_crimes_count', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 8, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8918918918918919
Accuracy on Training Set: 0.891156462585034
['closeness', 'degree', 'clustering', 'connected_criminals_count']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8918918918918919
Accuracy on Training Set: 0.9183673469387755
['closeness', 'degree', 'clustering', 'connected_criminals_count', 'total_crimes_count']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 6, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8513513513513513
Accuracy on Training Set: 0.8401360544217688
['closeness', 'degree', 'clustering', 'connected_criminals_count', 'total_crimes_count', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 6, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8513513513513513
Accuracy on Training Set: 0.8741496598639455
['closeness', 'degree', 'clustering', 'nearest_path', 'total_crimes_count', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 6, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8648648648648649
Accuracy on Training Set: 0.8945578231292517
['closeness', 'degree', 'clustering', 'nearest_path', 'connected_criminals_count']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 6, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8783783783783784
Accuracy on Training Set: 0.8843537414965986
['closeness', 'degree', 'clustering', 'nearest_path', 'connected_criminals_count', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 6, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8783783783783784
Accuracy on Training Set: 0.8537414965986394
['closeness', 'degree', 'clustering', 'nearest_path', 'connected_criminals_count', 'total_crimes_count']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 6, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8918918918918919
Accuracy on Training Set: 0.8843537414965986
['closeness', 'degree', 'clustering', 'nearest_path', 'connected_criminals_count', 'total_crimes_count', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 6, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8648648648648649
Accuracy on Training Set: 0.9217687074829932
['closeness', 'degree', 'pagerank', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 8, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8783783783783784
Accuracy on Training Set: 0.8945578231292517
['closeness', 'degree', 'pagerank', 'total_crimes_count']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 4, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8918918918918919
Accuracy on Training Set: 0.9455782312925171
['closeness', 'degree', 'pagerank', 'total_crimes_count', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 6, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8918918918918919
Accuracy on Training Set: 0.9319727891156463
['closeness', 'degree', 'pagerank', 'connected_criminals_count']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 8, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8918918918918919
Accuracy on Training Set: 0.9081632653061225
['closeness', 'degree', 'pagerank', 'connected_criminals_count', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 8, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8918918918918919
Accuracy on Training Set: 0.8877551020408163
['closeness', 'degree', 'pagerank', 'connected_criminals_count', 'total_crimes_count']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 6, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8918918918918919
Accuracy on Training Set: 0.9251700680272109
['closeness', 'degree', 'pagerank', 'connected_criminals_count', 'total_crimes_count', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 6, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8648648648648649
Accuracy on Training Set: 0.8775510204081632
['closeness', 'degree', 'pagerank', 'nearest_path']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 8, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8783783783783784
Accuracy on Training Set: 0.8537414965986394
['closeness', 'degree', 'pagerank', 'nearest_path', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 8, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8513513513513513
Accuracy on Training Set: 0.8877551020408163
['closeness', 'degree', 'pagerank', 'nearest_path', 'total_crimes_count']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 8, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8918918918918919
Accuracy on Training Set: 0.891156462585034
['closeness', 'degree', 'pagerank', 'nearest_path', 'total_crimes_count', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 8, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8783783783783784
Accuracy on Training Set: 0.891156462585034
['closeness', 'degree', 'pagerank', 'nearest_path', 'connected_criminals_count']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 8, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8783783783783784
Accuracy on Training Set: 0.8503401360544217
['closeness', 'degree', 'pagerank', 'nearest_path', 'connected_criminals_count', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 8, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8783783783783784
Accuracy on Training Set: 0.8843537414965986
['closeness', 'degree', 'pagerank', 'nearest_path', 'connected_criminals_count', 'total_crimes_count']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 8, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8648648648648649
Accuracy on Training Set: 0.891156462585034
['closeness', 'degree', 'pagerank', 'nearest_path', 'connected_criminals_count', 'total_crimes_count', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 8, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8513513513513513
Accuracy on Training Set: 0.9013605442176871
['closeness', 'degree', 'pagerank', 'clustering']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 6, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8918918918918919
Accuracy on Training Set: 0.935374149659864
['closeness', 'degree', 'pagerank', 'clustering', 'total_crimes_count']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 8, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8918918918918919
Accuracy on Training Set: 0.9251700680272109
['closeness', 'degree', 'pagerank', 'clustering', 'total_crimes_count', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 8, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8783783783783784
Accuracy on Training Set: 0.8877551020408163
['closeness', 'degree', 'pagerank', 'clustering', 'connected_criminals_count']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 8, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8648648648648649
Accuracy on Training Set: 0.891156462585034
['closeness', 'degree', 'pagerank', 'clustering', 'connected_criminals_count', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 8, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8648648648648649
Accuracy on Training Set: 0.8775510204081632
['closeness', 'degree', 'pagerank', 'clustering', 'connected_criminals_count', 'total_crimes_count']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 8, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8783783783783784
Accuracy on Training Set: 0.8945578231292517
['closeness', 'degree', 'pagerank', 'clustering', 'connected_criminals_count', 'total_crimes_count', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 8, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8918918918918919
Accuracy on Training Set: 0.8843537414965986
['closeness', 'degree', 'pagerank', 'clustering', 'nearest_path', 'total_crimes_count']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 8, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8783783783783784
Accuracy on Training Set: 0.8775510204081632
['closeness', 'degree', 'pagerank', 'clustering', 'nearest_path', 'total_crimes_count', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 8, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8783783783783784
Accuracy on Training Set: 0.8673469387755102
['closeness', 'degree', 'pagerank', 'clustering', 'nearest_path', 'connected_criminals_count']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 8, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8783783783783784
Accuracy on Training Set: 0.8741496598639455
['closeness', 'degree', 'pagerank', 'clustering', 'nearest_path', 'connected_criminals_count', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 8, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8918918918918919
Accuracy on Training Set: 0.8945578231292517
['closeness', 'degree', 'pagerank', 'clustering', 'nearest_path', 'connected_criminals_count', 'total_crimes_count']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 8, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8783783783783784
Accuracy on Training Set: 0.8741496598639455
['closeness', 'degree', 'pagerank', 'clustering', 'nearest_path', 'connected_criminals_count', 'total_crimes_count', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 6, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8783783783783784
Accuracy on Training Set: 0.8571428571428571
['betweenness', 'connected_criminals_count', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 6, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8783783783783784
Accuracy on Training Set: 0.8707482993197279
['betweenness', 'connected_criminals_count', 'total_crimes_count', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 8, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8783783783783784
Accuracy on Training Set: 0.8503401360544217
['betweenness', 'nearest_path', 'connected_criminals_count']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 6, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8513513513513513
Accuracy on Training Set: 0.8333333333333334
['betweenness', 'nearest_path', 'connected_criminals_count', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 8, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8513513513513513
Accuracy on Training Set: 0.8571428571428571
['betweenness', 'clustering', 'connected_criminals_count', 'total_crimes_count', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 6, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8513513513513513
Accuracy on Training Set: 0.8979591836734694
['betweenness', 'clustering', 'nearest_path', 'total_crimes_count', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8648648648648649
Accuracy on Training Set: 0.9047619047619048
['betweenness', 'clustering', 'nearest_path', 'connected_criminals_count', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 6, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8648648648648649
Accuracy on Training Set: 0.8571428571428571
['betweenness', 'clustering', 'nearest_path', 'connected_criminals_count', 'total_crimes_count']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 4, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8648648648648649
Accuracy on Training Set: 0.9047619047619048
['betweenness', 'clustering', 'nearest_path', 'connected_criminals_count', 'total_crimes_count', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 4, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8513513513513513
Accuracy on Training Set: 0.9115646258503401
['betweenness', 'pagerank', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 6, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8783783783783784
Accuracy on Training Set: 0.9013605442176871
['betweenness', 'pagerank', 'connected_criminals_count']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 8, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8513513513513513
Accuracy on Training Set: 0.8945578231292517
['betweenness', 'pagerank', 'connected_criminals_count', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 8, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8783783783783784
Accuracy on Training Set: 0.9013605442176871
['betweenness', 'pagerank', 'connected_criminals_count', 'total_crimes_count', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 8, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8648648648648649
Accuracy on Training Set: 0.8979591836734694
['betweenness', 'pagerank', 'nearest_path', 'total_crimes_count']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 8, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8648648648648649
Accuracy on Training Set: 0.8809523809523809
['betweenness', 'pagerank', 'nearest_path', 'total_crimes_count', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 8, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8513513513513513
Accuracy on Training Set: 0.8945578231292517
['betweenness', 'pagerank', 'nearest_path', 'connected_criminals_count']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 8, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.9054054054054054
Accuracy on Training Set: 0.8945578231292517
['betweenness', 'pagerank', 'nearest_path', 'connected_criminals_count', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 8, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8648648648648649
Accuracy on Training Set: 0.8843537414965986
['betweenness', 'pagerank', 'nearest_path', 'connected_criminals_count', 'total_crimes_count']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 6, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8783783783783784
Accuracy on Training Set: 0.8945578231292517
['betweenness', 'pagerank', 'nearest_path', 'connected_criminals_count', 'total_crimes_count', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 6, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8783783783783784
Accuracy on Training Set: 0.9115646258503401
['betweenness', 'pagerank', 'clustering', 'connected_criminals_count']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 8, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8513513513513513
Accuracy on Training Set: 0.8775510204081632
['betweenness', 'pagerank', 'clustering', 'connected_criminals_count', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 6, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8783783783783784
Accuracy on Training Set: 0.8945578231292517
['betweenness', 'pagerank', 'clustering', 'connected_criminals_count', 'total_crimes_count', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 8, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8513513513513513
Accuracy on Training Set: 0.8741496598639455
['betweenness', 'pagerank', 'clustering', 'nearest_path', 'total_crimes_count']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 8, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8648648648648649
Accuracy on Training Set: 0.8843537414965986
['betweenness', 'pagerank', 'clustering', 'nearest_path', 'total_crimes_count', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 6, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8648648648648649
Accuracy on Training Set: 0.9183673469387755
['betweenness', 'pagerank', 'clustering', 'nearest_path', 'connected_criminals_count']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 8, 'max_depth': 7, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8918918918918919
Accuracy on Training Set: 0.8877551020408163
['betweenness', 'pagerank', 'clustering', 'nearest_path', 'connected_criminals_count', 'lives_with_criminal']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 8, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8783783783783784
Accuracy on Training Set: 0.9081632653061225
['betweenness', 'pagerank', 'clustering', 'nearest_path', 'connected_criminals_count', 'total_crimes_count']

Best Parameters: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 8, 'max_depth': 6, 'class_weight': 'balanced'}
Accuracy on Test Set: 0.8783783783783784
Accuracy on Training Set: 0.891156462585034
['betweenness', 'pagerank', 'clustering', 'nearest_path', 'connected_criminals_count', 'total_crimes_count', 'lives_with_criminal']

"""


def string_to_dict(param_string):
    # Properly format the string for dictionary conversion
    param_string = param_string.replace("'", "\"")
    try:
        return ast.literal_eval(param_string)
    except:
        return {}



def parse_cluster(cluster):
    lines = cluster.split('\n')
    params = string_to_dict(lines[0].split('Best Parameters: ')[1])
    test_acc = float(lines[1].split(': ')[1])
    train_acc = float(lines[2].split(': ')[1])
    features = ast.literal_eval(lines[3])
    return {**params, 'Test Set Accuracy': test_acc, 'Training Set Accuracy': train_acc, 'Features': features}


# Parse each cluster and create a list of dictionaries
def result_to_csv(data, filename="default.csv"):
    # Split the data into clusters
    clusters = data.strip().split('\n\n')
    # Function to clean and convert parameter string to a dictionary
    parsed_data = [parse_cluster(cluster) for cluster in clusters]

    # Create a DataFrame
    df = pd.DataFrame(parsed_data)
    df = df.sort_values(by='Test Set Accuracy', ascending=False)
    # Write DataFrame to CSV

    csv_filename = filename
    df.to_csv(csv_filename, index=False)

    print(f"Data has been written to {csv_filename}")

result_to_csv(problem_knowledge_features, "problem_knowledge_features.csv")
